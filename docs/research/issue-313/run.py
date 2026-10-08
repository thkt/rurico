#!/usr/bin/env python3
"""CPU research measurement; output must be new and outside the checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
PROBE = Path(__file__).resolve().parent / "probe"
VARIANTS = ("rrf_map", "rrf_slots", "identity_borrowed", "identity_owned", "topk_sort", "topk_select")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = [ROOT / "Cargo.lock", ROOT / "src/retrieval.rs", ROOT / "src/retrieval/tests.rs",
             PROBE / "Cargo.toml", PROBE / "Cargo.lock", PROBE / "src/main.rs", Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def command(args, env, output):
    result = subprocess.run(args, cwd=ROOT, env=env, text=True, capture_output=True, timeout=540)
    output.write_text(result.stdout + result.stderr)
    result.check_returncode()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    args = parser.parse_args()
    out, target = args.output.resolve(), args.target_dir.resolve()
    if out.is_relative_to(ROOT) or target.is_relative_to(ROOT):
        parser.error("output and target-dir must be outside checkout")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("this RSS measurement command requires macOS Apple Silicon")
    if target.exists():
        parser.error("target-dir must be new: relative source paths must not reuse another checkout build")
    out.mkdir(parents=True, exist_ok=False)
    target.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    manifest = ["--offline", "--locked", "--manifest-path", str(PROBE / "Cargo.toml")]
    before = source_hashes()
    start = time.time()
    context = {
        "start_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": before, "platform": platform.platform(), "machine": platform.machine(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "cargo": subprocess.check_output(["cargo", "--version"], text=True).strip(),
        "profile": "release (standalone Cargo defaults, not product LTO)",
        "timing": "counting System allocator; includes counter overhead; 1 warmup then 7 calls per process",
        "peak": "additional live requested heap bytes per call; process max RSS separately includes setup/warmup",
        "input": "small:10 parents x 3 chunks; large:1000 x 100; 2 sources; k=3; deterministic tied dyadic scores",
        "ownership": "equal prepared input Vec for both identity paths; input preparation and result drop excluded",
        "isolation": "not established by runner; host must record concurrent load and reservation separately",
        "build_command": ["cargo", "build", *manifest, "--release"],
    }
    (out / "context.json").write_text(json.dumps(context, indent=2) + "\n")
    command(["cargo", "test", *manifest], env, out / "tests.log")
    command(["cargo", "clippy", *manifest, "--all-targets", "--", "-D", "warnings"], env, out / "clippy.log")
    command(context["build_command"], env, out / "build.log")
    binary = target / "release/retrieval-spike-313"
    command([str(binary), "--verify"], env, out / "equivalence.log")
    records, processes = [], []
    # Reverse full process order in round 2. Keep all rounds, including noise.
    for round_id in range(3):
        routes = [(w, v) for w in ("small", "large") for v in VARIANTS]
        if round_id % 2:
            routes.reverse()
        for workload, variant in routes:
            name = f"{round_id}-{workload}-{variant}"
            result = subprocess.run(["/usr/bin/time", "-l", str(binary), workload, variant],
                                    cwd=ROOT, env=env, text=True, capture_output=True, timeout=540)
            (out / f"{name}.stderr").write_text(result.stderr)
            (out / f"{name}.jsonl").write_text(result.stdout)
            result.check_returncode()
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            if len(rows) != 7 or [r["repeat"] for r in rows] != list(range(1, 8)):
                raise ValueError(f"incomplete measurement: {name}")
            for row in rows:
                if row["workload"] != workload or row["variant"] != variant:
                    raise ValueError(f"wrong measurement target: {name}")
                row["round"] = round_id
            records.extend(rows)
            rss = re.search(r"(\d+)\s+maximum resident set size", result.stderr)
            if rss is None:
                raise ValueError(f"RSS absent: {name}")
            processes.append({"round": round_id, "workload": workload, "variant": variant,
                              "max_rss_bytes": int(rss[1])})
    after = source_hashes()
    if before != after:
        raise ValueError("source changed during measurement; retain failed evidence")
    summary = []
    for workload in ("small", "large"):
        for variant in VARIANTS:
            rows = [r for r in records if r["workload"] == workload and r["variant"] == variant]
            entry = {"workload": workload, "variant": variant, "samples": len(rows)}
            for key in ("ns", "alloc_calls", "requested_bytes", "additional_peak_bytes"):
                values = [r[key] for r in rows]
                entry[key] = {"min": min(values), "median": statistics.median(values), "max": max(values)}
            rss_values = [r["max_rss_bytes"] for r in processes if r["workload"] == workload and r["variant"] == variant]
            entry["process_max_rss_bytes"] = {"samples": 3, "min": min(rss_values),
                                              "median": statistics.median(rss_values), "max": max(rss_values)}
            summary.append(entry)
    (out / "raw.json").write_text(json.dumps({"calls": records, "processes": processes}, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "complete.json").write_text(json.dumps({"source_after": after, "binary_sha256": digest(binary),
        "elapsed_seconds": time.time() - start, "calls": len(records), "processes": len(processes),
        "summary_sha256": digest(out / "summary.json"), "raw_sha256": digest(out / "raw.json")}, indent=2) + "\n")
    print(out / "summary.json")


if __name__ == "__main__":
    main()
