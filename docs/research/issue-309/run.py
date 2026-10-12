"""Issue #309 guarded host runner; uses #318 guards and #307 diagnostics.

No build, model download, baseline rewrite, publication or automatic retries.
complete.json means collection finished, never acceptance or adoption.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
BOUNDARIES = {"boundary128": [127, 128, 129], "boundary512": [511, 512, 513],
              "boundary2048": [2047, 2048, 2049]}
CASES = ("same", "mixed", *BOUNDARIES, "limit", "empty")
TEST = "reranker::mlx::bucket_probe::observe_length_batches"
REVISION = "bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3"
MAX_ABS = 2.5e-5


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


host = load("host318", ROOT / "docs/research/issue-318/run.py")
compare = load("compare307", ROOT / "docs/research/issue-307/compare.py")


def validate(records, case, mode, models):
    inputs = [r for r in records if r.get("event") == "input"]
    calls = [r for r in records if r.get("event") == "call"]
    compare.require(len(inputs) == 1, "incomplete libtest records")
    header = inputs[0]
    compare.require(header["workload"] == case and header["revision"] == REVISION
                    and header["model"] == "cl-nagoya/ruri-v3-reranker-310m"
                    and header["dtype"] == "F32", "wrong model/workload")
    compare.require(sorted(Path(p).resolve() for p in header["model_files"]) == sorted(p.resolve() for p in models),
                    "observed model paths differ from manifest")
    validate_calls(header, calls, case, mode)
    return header, calls


def validate_calls(header, calls, case, mode):
    """Validate raw and public call contents independently of model paths."""
    compare.require(len(calls) == 4, "incomplete libtest records")
    count = {"same": 50, "mixed": 50, **{name: 3 for name in BOUNDARIES}, "limit": 3, "empty": 0}[case]
    compare.require(len(header["pairs"]) == count, "wrong input count")
    for repeat, row in enumerate(calls):
        compare.require(row["repeat"] == repeat and row["state"] == ("cold" if repeat == 0 else "warm")
                        and row["candidate"] == (mode == "candidate"), "wrong mode/state")
        compare.require(len(row["lengths"]) == len(row["logits"]) == len(row["scores"]) == count,
                        "lost/extra output rows")
        compare.require(row["wall_ns"] >= row["encode_ns"] >= 0 and row["metal_peak_bytes"] >= 0,
                        "invalid observation units")
        compare.require(sorted(row["order"]) == list(range(count)), "missing ranked identities")
        compare.require(row["order"] == sorted(range(count), key=lambda i: -row["scores"][i]),
                        "production tie order changed")
        indices = []
        padded = 0
        for f in row["forwards"]:
            compare.require(f["bucket"] in (128, 512, 2048, 8192)
                            and 0 < f["rows"] <= 256000 // f["bucket"]
                            and f["rows"] == len(f["indices"]), "bucket/budget violated")
            compare.require(all(0 <= i < count and row["lengths"][i] <= f["bucket"] for i in f["indices"]),
                            "wrong forwarded identities/shape")
            if mode == "candidate":
                compare.require(all(f["bucket"] == next(b for b in (128, 512, 2048, 8192) if b >= row["lengths"][i])
                                    for i in f["indices"]), "candidate did not use smallest fixed bucket")
            else:
                compare.require(f["bucket"] == next(b for b in (128, 512, 2048, 8192) if b >= max(row["lengths"])),
                                "reference no longer uses global maximum")
            indices.extend(f["indices"])
            padded += f["rows"] * f["bucket"]
        compare.require(sorted(indices) == list(range(count)) and padded == row["padded_tokens"]
                        and sum(row["lengths"]) == row["real_tokens"], "incomplete forward accounting")
        if count:
            compare.finite(row["logits"])
            compare.finite(row["scores"])
            compare.require(all(0 <= v <= 1 for v in row["scores"]), "score range")
            compare.require(row["padding_ratio"] == padded / row["real_tokens"], "padding units")
        if case in BOUNDARIES:
            compare.require(row["lengths"] == BOUNDARIES[case],
                            "pair encoding no longer spans actual boundaries")
        if case == "limit":
            compare.require(row["lengths"] == [8191, 8192, 8192], "upper-limit truncation changed")



def distribution(values):
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def pair_diff(reference, candidate):
    values = compare.reranker_diff(reference, candidate)
    values["within_criteria"] = values["logit_abs"] <= MAX_ABS and values["score_abs"] <= MAX_ABS
    return values


def analyze(groups):
    result = {}
    for case in CASES:
        diagnostics = []
        modes = {}
        for trial in range(3):
            a_header, a = groups[(case, trial, "reference")]
            b_header, b = groups[(case, trial, "candidate")]
            compare.require(a_header == b_header, "different comparison inputs/model")
            for old, new in zip(a, b):
                compare.require(old["lengths"] == new["lengths"], "pair encoding mismatch")
                differences = [pair_diff([x, s], [y, t]) for x, s, y, t in
                               zip(old["logits"], old["scores"], new["logits"], new["scores"])]
                rank = compare.ranking_diff(list(range(len(old["scores"]))), old["logits"], new["logits"]) if old["scores"] else None
                diagnostics.append({"trial": trial, "repeat": old["repeat"], "differences": differences,
                                    "rank": rank, "exact_score_order": old["order"] == new["order"]})
        for mode in ("reference", "candidate"):
            runs = [groups[(case, trial, mode)][1] for trial in range(3)]
            modes[mode] = {
                "cold_wall_ns": distribution([r[0]["wall_ns"] for r in runs]),
                # Use a per-process median; warm calls within one process are correlated.
                "warm_process_median_wall_ns": distribution([statistics.median(r[i]["wall_ns"] for i in (1, 2, 3)) for r in runs]),
                "metal_peak_bytes": distribution([max(c["metal_peak_bytes"] for c in r) for r in runs]),
                "forward_count": [len(r[0]["forwards"]) for r in runs],
                "padded_tokens": [r[0]["padded_tokens"] for r in runs],
                "real_tokens": [r[0]["real_tokens"] for r in runs],
            }
        old = modes["reference"]["warm_process_median_wall_ns"]
        new = modes["candidate"]["warm_process_median_wall_ns"]
        noise = max(old["max"] - old["min"], new["max"] - new["min"])
        difference = new["median"] - old["median"]
        disjoint = old["max"] < new["min"] or new["max"] < old["min"]
        result[case] = {"diagnostics": diagnostics, "measurements": modes,
                        "warm_delta_ns": difference, "observed_range_noise_ns": noise,
                        "speed_signal": "not_applicable" if case == "empty" else "benefit" if disjoint and difference < -noise else "regression" if disjoint and difference > noise else "unresolved",
                        "within_diagnostics": all(d["exact_score_order"] and (d["rank"] is None or d["rank"]["reference_order"] == d["rank"]["actual_order"])
                            and all(v["within_criteria"] for v in d["differences"]) for d in diagnostics)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("context", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    context = json.loads(args.context.read_text())
    for key in ("machine", "ram_bytes", "os", "rust", "cargo", "xcode", "metal", "build", "load_conditions"):
        compare.require(bool(context.get(key)), f"missing host condition: {key}")
    models = [Path(p) for p in context["model_files"]]
    compare.require(all(p.is_absolute() and p.is_file() for p in models), "use absolute existing snapshot paths")
    compare.require(len(models) == 3 and {p.name for p in models} == {"model.safetensors", "config.json", "tokenizer.json"}
                    and all(p.parent.name == REVISION for p in models), "use three fixed reranker snapshot files")
    compare.require(all(v["dtype"] == "F32" for v in host.weight_header(next(p for p in models if p.name == "model.safetensors")).values()), "weights must be F32")
    compare.require(not args.output.resolve().is_relative_to(ROOT), "evidence must be outside checkout")
    sources = sorted(ROOT.joinpath("src").rglob("*.rs")) + sorted(ROOT.joinpath("crates").rglob("*.rs")) + sorted(ROOT.joinpath("vendor/hf-hub/src").rglob("*.rs"))
    sources += [ROOT / p for p in ("Cargo.toml", "Cargo.lock", "build.rs", "docs/research/issue-307/inputs.json",
                                   "docs/research/issue-307/compare.py", "docs/research/issue-307/results/numerical-rechecked/models.json", "docs/research/issue-318/run.py",
                                   "docs/research/issue-309/run.py", "crates/rurico-ffi/Cargo.toml", "vendor/hf-hub/Cargo.toml")]
    files = sources + models + [binary]
    before = {str(p): host.digest(p) for p in files}
    known = json.loads((ROOT / "docs/research/issue-307/results/numerical-rechecked/models.json").read_text())["reranker"]
    compare.require(known["revision"] == REVISION and known["repository"] == "cl-nagoya/ruri-v3-reranker-310m"
                    and all(before[str(p)] == known["files"][p.name]["sha256"] for p in models), "fixed model content differs from #307")
    diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
    context.update(start_commit=host.command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
                   tracked_diff_sha256=hashlib.sha256(diff).hexdigest(), files=before,
                   diagnostics={"logit_abs": MAX_ABS, "score_abs": MAX_ABS, "logit_order": "exact", "score_order": "exact", "tie_order": "original index"},
                   guards={"seconds": 1800, "rss_bytes": 8*1024**3, "swap_increase_bytes": 256*1024**2, "pressure": 1})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "context.json").write_text(json.dumps(context, indent=2) + "\n")
    deadline = time.monotonic() + 1800
    baseline = host.snapshot(-1)
    compare.require(baseline["pressure"] == 1, "pressure abnormal before measurement")
    groups, samples = {}, []
    for case in CASES:
        for trial in range(3):
            for mode in (("reference", "candidate") if trial % 2 == 0 else ("candidate", "reference")):
                compare.require(time.monotonic() < deadline, "measurement deadline before next process")
                name = f"{case}-{trial}-{mode}"
                status = {"case": case, "trial": trial, "mode": mode, "complete": False}
                with (args.output / f"{name}.stdout").open("w") as stdout, (args.output / f"{name}.stderr").open("w") as stderr:
                    process = subprocess.Popen([str(binary), TEST, "--exact", "--ignored", "--nocapture", "--test-threads=1"],
                        env=dict(os.environ, RURICO_309_MODE=mode, RURICO_309_WORKLOAD=case),
                        stdout=stdout, stderr=stderr, start_new_session=True)
                    try:
                        while process.poll() is None:
                            row = host.snapshot(process.pid)
                            row.update(name=name, monotonic=time.monotonic())
                            samples.append(row)
                            compare.require(time.monotonic() < deadline and row["rss_bytes"] <= 8*1024**3
                                            and row["pressure"] == 1 and row["swap_bytes"] - baseline["swap_bytes"] <= 256*1024**2,
                                            "measurement guard reached; not OOM/leak proof")
                            time.sleep(.1)
                        status["exit_code"] = process.returncode
                        compare.require(process.returncode == 0, f"{name}: libtest failed")
                    finally:
                        try:
                            host.reclaim(process)
                            status["group_absent"] = True
                        finally:
                            (args.output / "monitor.json").write_text(json.dumps(samples, indent=2) + "\n")
                            (args.output / f"{name}.status.json").write_text(json.dumps(status, indent=2) + "\n")
                records = [json.loads(line.removeprefix("bucket309 ")) for line in (args.output / f"{name}.stderr").read_text().splitlines()
                           if line.startswith("bucket309 ")]
                groups[(case, trial, mode)] = validate(records, case, mode, models)
                (args.output / f"{name}.json").write_text(json.dumps(records, indent=2) + "\n")
                status.update(complete=True, sampled_process_rss_max_bytes=max(row["rss_bytes"] for row in samples if row["name"] == name))
                (args.output / f"{name}.status.json").write_text(json.dumps(status, indent=2) + "\n")
    compare.require(before == {str(p): host.digest(p) for p in files}
                    and diff == subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT), "source/model/binary changed")
    summary = analyze(groups)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    passed = all(row["within_diagnostics"] for row in summary.values())
    (args.output / "complete.json").write_text(json.dumps({"collection_complete": True, "within_diagnostics": passed,
        "acceptance": "requires independent analysis and recommendation; no adoption",
        "files": {p.name: host.digest(p) for p in args.output.iterdir() if p.is_file()}, "after": host.snapshot(-1)}, indent=2) + "\n")
    if not passed:
        raise SystemExit("collection completed with numerical/rank disagreement; retain evidence")


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}; retain incomplete evidence")
    signal.signal(signal.SIGTERM, interrupt)
    main()
