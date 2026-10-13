"""Issue #317 explicit fixed-model lane; standard check remains unchanged."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
# Reuse existing host process ownership and OS memory observations.
spec = importlib.util.spec_from_file_location("host318", ROOT / "docs/research/issue-318/run.py")
host318 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host318)
TEST = "host317::observe_query317"
CASES = [("query", c) for c in ("single", "multiple", "boundary")] + [
    ("pair", c) for c in ("single", "multiple", "common", "boundary")]


def validate(records, domain, case, mode):
    if [r["event"] for r in records] != ["context", "load"] + ["call"] * 4 + ["comparison"]:
        raise ValueError("missing or excess observation records")
    if records[0] != {"event": "context", "domain": domain, "case": case, "mode": mode}:
        raise ValueError("wrong experiment identity")
    counts = {("query", "single"): 1, ("query", "multiple"): 32, ("query", "boundary"): 4,
              ("pair", "single"): 1, ("pair", "multiple"): 32, ("pair", "common"): 256,
              ("pair", "boundary"): 4}
    calls = records[2:6]
    for trial, row in enumerate(calls):
        if row["trial"] != trial or row["cold"] != (trial == 0):
            raise ValueError("wrong cold/warm or trial ordering")
        values = row["values"] if domain == "query" else row["scores"]
        if len(values) != counts[domain, case]:
            raise ValueError("missing or excess outputs")
        flat = [x for vector in values for x in vector] if domain == "query" else values
        if not flat or not all(math.isfinite(x) for x in flat):
            raise ValueError("nonfinite or missing output")
        metrics = row["metrics"]
        forwards = row["forwards"] if domain == "query" else metrics["forwards"]
        if sum(f["batch_size"] for f in forwards) != counts[domain, case]:
            raise ValueError("wrong forward rows")
        for forward in forwards:
            if forward["sequence_length"] not in (128, 512, 2048, 8192):
                raise ValueError("unexpected production bucket")
        if row["metal_peak_bytes"] < 0:
            raise ValueError("invalid peak")
    return calls


def compare_queries(a, b):
    maximum, minimum = 0.0, 1.0
    if len(a) != len(b):
        raise ValueError("query count differs")
    for x, y in zip(a, b):
        if len(x) != 768 or len(y) != 768:
            raise ValueError("fixed310m dimension differs")
        norm = math.sqrt(sum(t*t for t in x) * sum(t*t for t in y))
        if not norm:
            raise ValueError("zero norm")
        maximum = max(maximum, max(abs(s-t) for s, t in zip(x, y)))
        minimum = min(minimum, sum(s*t for s, t in zip(x, y))/norm)
    if maximum > 1e-5 or minimum < .99999:
        raise ValueError("Issue #307 same-token diagnostic failed")
    return {"max_abs": maximum, "min_cosine": minimum}


def summaries(results):
    output = []
    for domain, case in CASES:
        baseline = results[domain, case, "baseline"]
        candidate = results[domain, case, "candidate"]
        if domain == "query":
            diagnostic = compare_queries(baseline[0]["values"], candidate[0]["values"])
        else:
            if baseline[0]["scores"] != candidate[0]["scores"]:
                raise ValueError("pair reuse changed scores/order")
            diagnostic = {"exact_scores": True}
        entry = {"domain": domain, "case": case, "diagnostic": diagnostic}
        for mode, calls in (("baseline", baseline), ("candidate", candidate)):
            def fields(row):
                if domain == "query":
                    return {"wall_ns": row["wall_ns"], "tokenize_ns": row["tokenize_ns"],
                            "forward_count": len(row["forwards"]), "metal_peak_bytes": row["metal_peak_bytes"]}
                return {"wall_ns": row["metrics"]["wall_ns"], "tokenize_ns": row["metrics"]["tokenize_ns"],
                        "forward_count": len(row["metrics"]["forwards"]), "metal_peak_bytes": row["metal_peak_bytes"]}
            extracted = [fields(row) for row in calls]
            entry[mode] = {"cold": extracted[0], "warm": {
                key: {"min": min(values), "median": statistics.median(values), "max": max(values)}
                for key in extracted[0]
                for values in [[row[key] for row in extracted[1:]]]}}
        output.append(entry)
    return output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("context", type=Path)
    parser.add_argument("output", type=Path)
    # Required explicit experiment budget. It is not a product resource limit.
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("positive measurement budget required")
    context = json.loads(args.context.read_text())
    for key in ("machine", "ram_bytes", "os", "rust", "cargo", "xcode", "metal", "build", "load_conditions", "embedding_snapshot", "reranker_snapshot", "parent_issue_evidence"):
        if not context.get(key):
            raise ValueError(f"missing host context: {key}")
    snapshots = [Path(context[key]).resolve(strict=True) for key in ("embedding_snapshot", "reranker_snapshot")]
    revisions = ("18b60fb8c2b9df296fb4212bb7d23ef94e579cd3", "bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3")
    if any(path.name != revision for path, revision in zip(snapshots, revisions)):
        raise ValueError("fixed HF snapshot revision required")
    models = [path/name for path in snapshots for name in ("model.safetensors", "config.json", "tokenizer.json")]
    binary = args.binary.resolve(strict=True)
    out = args.output.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError("use new evidence directory outside checkout")
    source = [ROOT/"Cargo.lock", ROOT/"Cargo.toml", ROOT/"build.rs"]
    source += sorted((ROOT/"src").rglob("*.rs")) + sorted((ROOT/"docs/research/issue-317").glob("*.*"))
    # Also retain all injected/existing test source compiled into this libtest.
    source += [ROOT/p for p in subprocess.check_output(["git", "ls-files", "docs", "tests"], cwd=ROOT, text=True).splitlines() if p.endswith((".rs", ".json"))]
    paths = sorted(set(source + models + [binary, args.context.resolve()]))
    before = {str(p): host318.digest(p) for p in paths}
    diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
    out.mkdir(parents=True, exist_ok=False)
    (out/"source.patch").write_bytes(diff)
    headers = {str(p):host318.weight_header(p) for p in models if p.name=="model.safetensors"}
    if any(tensor["dtype"] != "F32" for header in headers.values() for tensor in header.values()):
        raise ValueError("fixed F32 experiment required")
    context.update(start_commit=host318.command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
                   source_diff_sha256=hashlib.sha256(diff).hexdigest(), files=before,
                   measurement_seconds=args.seconds, weight_headers=headers)
    (out/"context.json").write_text(json.dumps(context, indent=2)+"\n")
    # Preserve the new prototype source, which git diff alone omits before add.
    for path in (ROOT/"docs/research/issue-317").glob("*.*"):
        (out/path.name).write_bytes(path.read_bytes())
    deadline = time.monotonic() + args.seconds
    results, samples = {}, []
    for ordinal, (domain, case) in enumerate(CASES):
        modes = ("baseline", "candidate") if ordinal%2==0 else ("candidate", "baseline")
        for mode in modes:
            if time.monotonic() >= deadline:
                raise RuntimeError("measurement budget reached; incomplete evidence retained")
            key = f"{domain}-{case}-{mode}"
            record_path = out/f"{key}.jsonl"
            record_path.touch(exist_ok=False)
            env = dict(os.environ, RURICO_317_DOMAIN=domain, RURICO_317_CASE=case, RURICO_317_MODE=mode,
                       RURICO_317_OUTPUT=str(record_path), RURICO_307_MODEL_DIR=str(snapshots[domain=="pair"]))
            argv = [str(binary), TEST, "--exact", "--ignored", "--nocapture", "--test-threads=1"]
            status = {"argv": argv, "complete": False}
            with (out/f"{key}.stdout").open("w") as stdout, (out/f"{key}.stderr").open("w") as stderr:
                process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                try:
                    while process.poll() is None:
                        observation = host318.snapshot(process.pid)
                        observation.update(case=key, monotonic=time.monotonic())
                        samples.append(observation)
                        if time.monotonic() >= deadline:
                            raise RuntimeError("measurement budget reached")
                        time.sleep(.1)
                    status["exit_code"] = process.returncode
                    if process.returncode:
                        raise RuntimeError(f"{key} exited {process.returncode}")
                finally:
                    try:
                        host318.reclaim(process)
                        status["group_absent"] = True
                    finally:
                        (out/f"{key}.status.json").write_text(json.dumps(status, indent=2)+"\n")
                        (out/"monitor.json").write_text(json.dumps(samples, indent=2)+"\n")
            rows = [json.loads(line) for line in record_path.read_text().splitlines()]
            results[domain, case, mode] = validate(rows, domain, case, mode)
            status["complete"] = True
            (out/f"{key}.status.json").write_text(json.dumps(status, indent=2)+"\n")
    after = {str(p):host318.digest(p) for p in paths}
    if before != after or diff != subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT):
        raise RuntimeError("source/model/binary/context changed")
    (out/"summary.json").write_text(json.dumps(summaries(results), indent=2)+"\n")
    (out/"complete.json").write_text(json.dumps({"collection_complete":True,"acceptance":"requires independent analysis", "files": {p.name:host318.digest(p) for p in out.iterdir() if p.is_file()}},indent=2)+"\n")


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}; incomplete evidence retained")
    signal.signal(signal.SIGTERM, interrupt)
    main()
