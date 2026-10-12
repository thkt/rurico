"""Host-only #318 evidence runner. No builds, downloads, retries or publication.

Reuses #364's measurement guards, NOT a product resource/deadline contract.
Run a prebuilt libtest binary in separate owned process groups; preserve partial
evidence on every failure. Complete means collection finished, not acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


CASES = ("retention", "shapes", "forward", "overlap", "control", "fairness", "ownership", "mixed")
ALL_CASES = (*CASES, "deadline")
TEST = "runtime_spike::observe_runtime"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weight_header(path):
    with path.open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if not 0 < length <= 16*1024**2:
            raise ValueError("invalid safetensors header size")
        header = json.loads(stream.read(length))
    return {key:{"dtype":value["dtype"], "shape":value["shape"]}
            for key, value in header.items() if key != "__metadata__"}


def command(*args):
    return subprocess.check_output(args, text=True, timeout=2).strip()


def snapshot(pgid):
    rows = command("ps", "-axo", "pid=,pgid=,rss=").splitlines()
    rss = sum(int(row.split()[2]) * 1024 for row in rows if int(row.split()[1]) == pgid)
    pressure = int(command("sysctl", "-n", "kern.memorystatus_vm_pressure_level"))
    swap = command("sysctl", "-n", "vm.swapusage")
    match = re.search(r"used\s*=\s*([\d.]+)([MG])", swap)
    if not match:
        raise RuntimeError(f"unrecognized swap observation: {swap}")
    used = float(match[1]) * (1024**2 if match[2] == "M" else 1024**3)
    return {"rss_bytes": rss, "pressure": pressure, "swap_bytes": used}


def reclaim(process):
    # Keep ownership until process and its ordinary descendants are absent.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass
    deadline = time.monotonic() + 2
    while True:
        process.poll()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            process.wait(timeout=2)
            return
        except PermissionError:
            pass  # macOS zombie-only groups may report EPERM; not proof of exit.
        if time.monotonic() >= deadline:
            raise RuntimeError("owned group disappearance not confirmed")
        time.sleep(.01)


def validate(case, records):
    contexts = [r for r in records if r.get("event") == "context"]
    if len(contexts) != 1 or contexts[0]["case"] != case:
        raise RuntimeError("missing/incorrect experiment context")
    counts = {event: sum(r.get("event") == event for r in records)
              for event in ("retention", "shape_case", "direct", "comparison", "stream", "deadline", "deadline_forward", "deadline_result", "trial", "forward", "fairness", "ownership", "ownership_peak", "mixed", "mixed_peak")}
    required = {
        "retention": {"retention": 3}, "forward": {"direct":3, "comparison": 12, "stream": 12},
        "overlap": {"direct":3, "comparison": 6, "stream": 6}, "control": {"stream": 6, "deadline": 3},
        "fairness": {"fairness": 96}, "ownership": {"ownership": 48, "ownership_peak": 6},
        "mixed": {"mixed": 12, "mixed_peak": 6},
        "shapes": {"shape_case":4},
        "deadline": {"trial":6, "stream":6, "forward":6, "deadline_forward":6, "deadline_result":6},
    }[case]
    if any(counts[key] != value for key, value in required.items()):
        raise RuntimeError(f"incomplete observations: {counts}; required {required}")
    if case == "deadline":
        trials = [i for i, row in enumerate(records) if row.get("event") == "trial"]
        expected = [(trial, overlap) for trial in range(3) for overlap in (False, True)]
        for index, (trial, overlap) in enumerate(expected):
            begin = trials[index]
            end = trials[index + 1] if index + 1 < len(trials) else len(records)
            group = records[begin:end]
            if (group[0]["trial"], group[0]["overlap"]) != (trial, overlap):
                raise RuntimeError("incorrect deadline trial/overlap")
            selected = {}
            for event in ("deadline_forward", "forward", "stream", "deadline_result"):
                rows = [r for r in group if r.get("event") == event]
                if len(rows) != 1:
                    raise RuntimeError("deadline must stop before a second forward")
                selected[event] = rows[0]
            timing, forward, stream, result = (selected[event] for event in
                ("deadline_forward", "forward", "stream", "deadline_result"))
            if (timing["remaining_at_start_ns"] <= 0 or timing["overrun_at_end_ns"] < 0
                or timing["forward_wall_ns"] != timing["remaining_at_start_ns"] + timing["overrun_at_end_ns"]
                or timing["rows"] != 10 or forward["rows"] != 10
                or stream["requested"] != 50 or stream["completed"] != 10
                or stream["cancelled"] is not False or stream["overlap"] != overlap
                or result["completed"] != 10 or result["complete_vec_rejected"] is not True
                or (result["trial"], result["overlap"]) != (trial, overlap)
                or result["measurement_deadline_ms"] != 10):
                raise RuntimeError("deadline boundary conditions not observed")
    if case == "shapes":
        tensors = [r for r in records if r.get("event") == "tensor"]
        for rows in (1, 10, 25, 50):
            for stage, width in (("embedding_hidden", 768), ("qkv_projection", 2304), ("glu_projection_before_split", 6144)):
                matches = [r for r in tensors if r.get("stage") == stage and r.get("shape") == [rows, 128, width]]
                if not matches or any(r.get("dtype") != "Float32" for r in matches):
                    raise RuntimeError("expected shape/dtype metadata was not observed")
    if case == "retention":
        samples = [r for r in records if r.get("event") == "retention"]
        if [r["inputs"] for r in samples] != [50, 500, 5000]:
            raise RuntimeError("incorrect retention input sizes")
        if len({r["prepared_peak_bytes"] for r in samples}) != 1:
            raise RuntimeError("prepared payload did not converge")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("context", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", nargs="+", choices=ALL_CASES, default=ALL_CASES)
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases):
        raise ValueError("duplicate experiment cases")
    context = json.loads(args.context.read_text())
    for key in ("machine", "ram_bytes", "os", "rust", "cargo", "xcode", "metal", "build", "load_conditions"):
        if not context.get(key):
            raise ValueError(f"missing host condition: {key}")
    # Caller lists the actual fixed-cache model/config/tokenizer paths. No
    # search of credentials, model download, or mutation of outside controls.
    model_paths = [Path(p) for p in context["model_files"]]
    if any(not p.is_absolute() for p in model_paths):
        raise ValueError("use absolute fixed-snapshot model paths")
    if len(model_paths) != 6 or len(set(model_paths)) != 6:
        raise ValueError("list embed310m and reranker310m's six distinct files")
    if sum(p.name == "model.safetensors" for p in model_paths) != 2:
        raise ValueError("both fixed model.safetensors files are required")
    root = Path(__file__).resolve().parents[3]
    binary = args.binary.resolve(strict=True)
    source_paths = [root / p for p in (
        "Cargo.lock", "src/reranker/mlx.rs", "docs/research/issue-318/host.rs",
        "src/research.rs", "src/research/runtime318.rs", "src/modernbert/model.rs",
        "docs/research/issue-318/planner.rs", "docs/research/issue-318/run.py")]
    source_paths += sorted((root / "docs/research/issue-318/probe").rglob("*.rs"))
    source_paths += [root / "docs/research/issue-318/probe" / name for name in ("Cargo.toml", "Cargo.lock")]
    tracked_diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)
    if args.output.resolve().is_relative_to(root):
        raise ValueError("evidence must be outside checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    before = {str(p): digest(p) for p in source_paths + model_paths + [binary]}
    context.update(start_commit=command("git", "-C", str(root), "rev-parse", "HEAD"),
                   source_diff_sha256=hashlib.sha256(tracked_diff).hexdigest(), files=before,
                   weight_headers={str(p):weight_header(p) for p in model_paths if p.name == "model.safetensors"},
                   binary=str(binary), guards={"seconds":1800, "rss_bytes":8*1024**3,
                       "swap_increase_bytes":256*1024**2, "pressure":1})
    context["cases"] = list(args.cases)
    (args.output / "context.json").write_text(json.dumps(context, indent=2) + "\n")
    deadline = time.monotonic() + 1800
    baseline = snapshot(-1)
    if baseline["pressure"] != 1:
        raise RuntimeError("host pressure is not normal before measurement")
    samples = []
    for case in args.cases:
        if time.monotonic() >= deadline:
            raise RuntimeError("measurement time limit reached before next case")
        env = dict(os.environ, RURICO_318_CASE=case)
        status = {"case":case, "complete":False}
        with (args.output / f"{case}.stdout").open("w") as stdout, (args.output / f"{case}.stderr").open("w") as stderr:
            process = subprocess.Popen([str(binary), TEST, "--exact", "--ignored", "--nocapture", "--test-threads=1"],
                                       env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                while process.poll() is None:
                    observation = snapshot(process.pid)
                    observation.update(case=case, monotonic=time.monotonic())
                    samples.append(observation)
                    if (time.monotonic() >= deadline or observation["rss_bytes"] > 8*1024**3
                        or observation["pressure"] != 1 or observation["swap_bytes"]-baseline["swap_bytes"] > 256*1024**2):
                        status["stop_reason"] = "measurement guard reached; not OOM/leak proof"
                        raise RuntimeError(status["stop_reason"])
                    time.sleep(.1)
                status["exit_code"] = process.returncode
                if process.returncode:
                    raise RuntimeError(f"{case} exited {process.returncode}")
            finally:
                try:
                    reclaim(process)
                    status["group_absent"] = True
                finally:
                    (args.output / "monitor.json").write_text(json.dumps(samples, indent=2) + "\n")
                    (args.output / f"{case}.status.json").write_text(json.dumps(status, indent=2) + "\n")
        records = [json.loads(line.removeprefix("runtime318 ")) for line in (args.output / f"{case}.stderr").read_text().splitlines()
                   if line.startswith("runtime318 ")]
        validate(case, records)
        (args.output / f"{case}.json").write_text(json.dumps(records, indent=2) + "\n")
        status["complete"] = True
        (args.output / f"{case}.status.json").write_text(json.dumps(status, indent=2) + "\n")
    after = {str(p): digest(p) for p in source_paths + model_paths + [binary]}
    after_diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)
    if before != after or tracked_diff != after_diff:
        raise RuntimeError("source/model/binary changed during observations")
    (args.output / "complete.json").write_text(json.dumps({
        "collection_complete":True, "acceptance":"requires independent analysis",
        "files":{p.name:digest(p) for p in args.output.iterdir() if p.is_file()},
        "after":snapshot(-1)}, indent=2) + "\n")


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}; retain incomplete evidence")
    signal.signal(signal.SIGTERM, interrupt)
    main()
