"""Host-only three-table comparison. No publication, adoption or automatic retry.

Prepare isolated source copies first; measure prebuilt binaries on Apple Silicon.
Partial logs survive interruption. No new resource hard limit is introduced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import compare

ROOT = compare.ROOT
host = compare.load("host318", ROOT / "docs/research/issue-318/run.py")
processes = compare.load("processes307", ROOT / "docs/research/issue-307/processes.py")
TESTS = {"embedding": "embed::mlx::bucket_probe::observe_bucket_tables",
         "reranker": "reranker::mlx::bucket_probe::observe_length_batches"}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def source_files():
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    return sorted({p.decode() for p in paths.split(b"\0") if p and (ROOT / p.decode()).is_file()})


def manifest(root, paths):
    return {p: host.digest(root / p) for p in paths}


def patch(root, table):
    """Only experimental copies change layout; original four-entry API stays fixed."""
    bounds = compare.TABLES[table]
    n = len(bounds)
    path = root / "src/model_io.rs"
    source = path.read_text()
    old = "pub(crate) const BUCKET_BOUNDS: [usize; 4] = [128, 512, 2048, MAX_SEQ_LEN];"
    compare.oracle.require(source.count(old) == 1, "bucket source changed; inspect before patching")
    source = source.replace(old, f"pub(crate) const BUCKET_BOUNDS: [usize; {n}] = [{', '.join(map(str, bounds[:-1]))}, MAX_SEQ_LEN];")
    path.write_text(source)
    for name in ("src/embed/processing.rs", "src/embed/metrics.rs", "src/reranker/bucket_spike.rs"):
        path = root / name
        source = path.read_text().replace("; 4]", f"; {n}]")
        source = source.replace("[Vec::new(), Vec::new(), Vec::new(), Vec::new()]", "std::array::from_fn(|_| Vec::new())")
        path.write_text(source)
    # libtest compiles existing tests too. Preserve their four-index log fixture,
    # extending only unused entries. Those fixed-table tests run in original check.
    path = root / "src/embed/metrics/tests.rs"
    source = path.read_text()
    old = "m.bucket_hist = [3, 5, 2, 1];"
    compare.oracle.require(source.count(old) == 1, "metrics fixture changed")
    path.write_text(source.replace(old, f"m.bucket_hist = [{', '.join(map(str, [3, 5, 2, 1] + [0]*(n-4)))}];"))
    path = root / "src/embed/processing/tests.rs"
    source = path.read_text()
    old = "assert_eq!(metrics.bucket_hist, [5, 3, 1, 1]);"
    compare.oracle.require(source.count(old) == 1, "fixed-table assertion changed")
    # Preserve the original four-entry expectation, but permit libtest to compile
    # with a different array length. This test is not selected in the copies.
    path.write_text(source.replace(old, "assert_eq!(metrics.bucket_hist.as_slice(), &[5, 3, 1, 1]);"))


def prepare(output, cache_base=None):
    compare.oracle.require(not output.is_relative_to(ROOT), "output must be outside checkout")
    output.mkdir(parents=True, exist_ok=False)
    paths = source_files()
    before = manifest(ROOT, paths)
    plan = {"start_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source": before, "source_sha256": hashlib.sha256(json.dumps(before, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "tables": compare.TABLES, "binaries": {}, "variant_sources": {}}
    save(output / "plan.json", plan)
    for table in compare.TABLES:
        clone = output / table
        for p in paths:
            target = clone / p
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / p, target)
        if cache_base is not None:
            compare.oracle.require(cache_base != output and not cache_base.is_relative_to(output), "cache source overlaps output")
            shutil.copytree(cache_base / table / "target", clone / "target")
        if table != "current":
            patch(clone, table)
        plan["variant_sources"][table] = manifest(clone, paths)
        save(output / "plan.json", plan)
        with (output / f"{table}-build.jsonl").open("w") as stdout, (output / f"{table}-build.stderr").open("w") as stderr:
            processes.owned_run(["cargo", "test", "--locked", "--release", "--lib", "--features", "test-mlx,smoke",
                                 "--no-run", "--message-format=json"], cwd=clone, stdout=stdout, stderr=stderr, check=True,
                                env=dict(os.environ, CARGO_TARGET_DIR=str(clone / "target")))
        artifacts = [json.loads(line) for line in (output / f"{table}-build.jsonl").read_text().splitlines() if line.startswith("{")]
        binaries = [r["executable"] for r in artifacts if r.get("reason") == "compiler-artifact" and r.get("executable")
                    and r["target"]["name"] == "rurico" and r["target"]["kind"] == ["lib"] and r["profile"]["test"]]
        compare.oracle.require(len(set(binaries)) == 1, "ambiguous/missing libtest")
        binary = Path(binaries[0])
        plan["binaries"][table] = {"path": str(binary), "sha256": host.digest(binary)}
        save(output / "plan.json", plan)
    verify_sources(output, plan)
    save(output / "prepared.json", {"complete": True, "plan_sha256": host.digest(output / "plan.json")})
    verify_prepared(output)


def verify_sources(prepared, plan):
    """Fresh post-build observation, also repeated before/after measurement."""
    compare.oracle.require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == plan["start_commit"], "HEAD changed")
    compare.oracle.require(json.loads((prepared / "plan.json").read_text()) == plan, "plan changed")
    compare.oracle.require(plan["tables"] == compare.TABLES and plan["source"] == manifest(ROOT, source_files()), "checkout changed; prepare new copies")
    for table, binary in plan["binaries"].items():
        compare.oracle.require(host.digest(Path(binary["path"])) == binary["sha256"] and
                               manifest(prepared / table, plan["source"]) == plan["variant_sources"][table], "variant source/binary changed")


def verify_prepared(prepared):
    record = json.loads((prepared / "prepared.json").read_text())
    compare.oracle.require(record["complete"] is True and host.digest(prepared / "plan.json") == record["plan_sha256"], "preparation incomplete or plan changed")


def verify_plan(prepared, plan):
    verify_prepared(prepared)
    verify_sources(prepared, plan)


def observe(binary, kind, case, repeats, budget, directory, name, model_dirs, baseline, deadline):
    prefix = "bucket308 " if kind == "embedding" else "bucket309 "
    env = dict(os.environ, RURICO_308_CASE=case, RURICO_308_BUDGET=str(budget), RURICO_308_REPEATS=str(repeats),
               RURICO_309_MODE="reference", RURICO_309_WORKLOAD=case)
    compare.oracle.require(time.monotonic() < deadline, "measurement deadline before next process")
    samples, status = [], {"complete": False}
    with (directory / f"{name}.stdout").open("w") as stdout, (directory / f"{name}.stderr").open("w") as stderr:
        process = subprocess.Popen([str(binary), TESTS[kind], "--exact", "--ignored", "--nocapture", "--test-threads=1"],
                                   stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        try:
            while process.poll() is None:
                row = host.snapshot(process.pid)
                row["elapsed_seconds"] = time.monotonic() - started
                samples.append(row)
                compare.oracle.require(time.monotonic() < deadline and row["rss_bytes"] <= 8*1024**3
                    and row["pressure"] == 1 and row["swap_bytes"] - baseline["swap_bytes"] <= 256*1024**2,
                    "measurement guard reached; not OOM/leak proof")
                if len(samples) % 10 == 0:
                    row["processes"] = host.command("ps", "-axo", "pid=,ppid=,pcpu=,comm=")
                time.sleep(.1)
            status["exit_code"] = process.returncode
            compare.oracle.require(process.returncode == 0, "libtest failed; retain partial evidence")
        finally:
            try:
                host.reclaim(process)
                status["group_absent"] = True
            finally:
                save(directory / f"{name}.monitor.json", samples)
                save(directory / f"{name}.status.json", status)
    records = [json.loads(line.removeprefix(prefix)) for line in (directory / f"{name}.stderr").read_text().splitlines() if line.startswith(prefix)]
    save(directory / f"{name}.json", records)
    header = records[0]
    compare.oracle.require(sorted(Path(p).resolve() for p in header["model_files"]) ==
                           sorted((model_dirs[kind] / p).resolve() for p in ("model.safetensors", "config.json", "tokenizer.json")), "observed model differs")
    status.update(complete=True, sampled_process_rss_max_bytes=max(r["rss_bytes"] for r in samples))
    save(directory / f"{name}.status.json", status)
    return records


def select_cases(kinds, cases):
    available = {"embedding": compare.EMBED_CASES, "reranker": compare.RERANK_CASES}
    compare.oracle.require(bool(kinds) and len(set(kinds)) == len(kinds)
                           and set(kinds) <= available.keys(), "invalid/duplicate kinds")
    if cases is not None:
        compare.oracle.require(bool(cases) and len(set(cases)) == len(cases), "empty/duplicate cases")
        compare.oracle.require(set(cases) <= {case for kind in kinds for case in available[kind]},
                               "cases unavailable for selected kinds")
    selected = {kind: [case for case in available[kind] if cases is None or case in cases] for kind in kinds}
    compare.oracle.require(all(selected.values()), "selected kind has no cases")
    return selected


def measure(prepared, context_path, output, repeats, sustained_repeats, budget, kinds, cases, trials):
    selected = select_cases(kinds, cases)
    plan = json.loads((prepared / "plan.json").read_text())
    verify_plan(prepared, plan)
    context = json.loads(context_path.read_text())
    for key in ("machine", "ram_bytes", "os", "rust", "cargo", "xcode", "metal", "build", "load_conditions"):
        compare.oracle.require(bool(context.get(key)), f"missing host condition: {key}")
    compare.oracle.require(repeats >= 4 and sustained_repeats >= 4 and budget > 0 and trials in (1, 3), "invalid observation plan")
    known = json.loads((ROOT / "docs/research/issue-307/results/numerical-rechecked/models.json").read_text())
    model_dirs = {k: Path(context["model_dirs"][k]).resolve() for k in TESTS}
    models = {}
    for kind, directory in model_dirs.items():
        compare.oracle.require(directory.name == known[kind]["revision"], "wrong snapshot revision")
        for name in ("model.safetensors", "config.json", "tokenizer.json"):
            p = directory / name
            models[str(p)] = host.digest(p)
            compare.oracle.require(models[str(p)] == known[kind]["files"][name]["sha256"], "model differs from #307")
    compare.oracle.require(not output.is_relative_to(ROOT) and not output.is_relative_to(prepared), "use new external evidence directory")
    output.mkdir(parents=True, exist_ok=False)
    context.update(plan_sha256=host.digest(prepared / "plan.json"), source_sha256=plan["source_sha256"], source=plan["source"], models=models,
                   trials=trials, repeats=repeats, sustained_repeats=sustained_repeats, embedding_budget=budget, pause_ms=1,
                   reranker_budget=256000, reranker_mode="reference", resource_limits="no new limit; host execution contract remains applicable")
    context.update(kinds=kinds, cases=cases, guards={"seconds":1800,"rss_bytes":8*1024**3,
        "swap_increase_bytes":256*1024**2,"pressure":1},
        diagnostics={"logit_abs":2.5e-5,"score_abs":2.5e-5,"rank":"exact","tie_order":"original index"})
    save(output / "context.json", context)
    global started
    started = time.monotonic()
    summaries = {}
    deadline = started + 1800
    baseline = host.snapshot(-1)
    try:
        compare.oracle.require(baseline["pressure"] == 1, "pressure abnormal before measurement")
        for kind in TESTS:
            if kind not in kinds:
                continue
            for case in selected[kind]:
                count = sustained_repeats if case in ("mixed", "multichunk") else repeats
                groups, records = {}, {}
                for trial in range(trials):
                    tables = list(compare.TABLES)
                    tables = tables[trial:] + tables[:trial]
                    for table in tables:
                        name = f"{kind}-{case}-{trial}-{table}"
                        data = observe(Path(plan["binaries"][table]["path"]), kind, case, count, budget, output, name, model_dirs, baseline, deadline)
                        records[(trial, table)] = data
                        groups[(trial, table)] = compare.validate(data, kind, case, compare.TABLES[table], count)
                diagnostics = {}
                for table in list(compare.TABLES)[1:]:
                    diagnostics[table] = [compare.difference(groups[(i, "current")], groups[(i, table)], kind) for i in range(trials)]
                    if kind == "embedding":
                        diagnostics[table].append({"queries": [compare.oracle.vector_diff(u, v) for i in range(trials) for u, v in
                            zip(records[(i, "current")][-1]["values"], records[(i, table)][-1]["values"])]})
                summaries[f"{kind}/{case}"] = {"diagnostics": diagnostics,
                    "measurements": {table: compare.timings([groups[(i, table)][1] for i in range(trials)], kind) for table in compare.TABLES}}
                save(output / "summary.json", summaries)
        verify_plan(prepared, plan)
        compare.oracle.require(models == {p: host.digest(Path(p)) for p in models}, "model changed")
        save(output / "complete.json", {"collection_complete": True,
             "local_mask_capacity_estimate_mib": {"current": 273.0625, "4096": 337.0625, "1024-4096": 341.0625},
             "acceptance": "requires numerical analysis, speed/memory assessment and recommendation; reranker diagnostics are not agreed product tolerances or adoption authorization",
             "files": {p.name: host.digest(p) for p in output.iterdir() if p.is_file()}})
    except BaseException as error:
        save(output / "incomplete.json", {"collection_complete": False, "reason": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("output", type=Path)
    prep.add_argument("--cache-base", type=Path)
    run = sub.add_parser("measure")
    run.add_argument("prepared", type=Path)
    run.add_argument("context", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--repeats", type=int, default=4)
    run.add_argument("--sustained-repeats", type=int, default=31)
    run.add_argument("--budget", type=int, default=8192)
    run.add_argument("--trials", type=int, choices=(1, 3), default=3)
    run.add_argument("--kinds", nargs="+", choices=tuple(TESTS), default=list(TESTS))
    run.add_argument("--cases", nargs="+", choices=tuple(dict.fromkeys((*compare.EMBED_CASES, *compare.RERANK_CASES))))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output.resolve(), args.cache_base.resolve() if args.cache_base else None)
    else:
        measure(args.prepared.resolve(), args.context, args.output.resolve(), args.repeats, args.sustained_repeats, args.budget, args.kinds, args.cases, args.trials)


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}; retain incomplete evidence")
    signal.signal(signal.SIGTERM, interrupt)
    main()
