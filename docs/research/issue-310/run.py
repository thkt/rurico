"""Host-only #310 comparison. Reuse #308 ownership, input and numerical validators.

No adoption, publication, new product hard limit or baseline rewrite.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load("compare310", ROOT / "docs/research/issue-308/compare.py")
patches = load("patch310", Path(__file__).with_name("patch.py"))
# Each module is private to this adapter, leaving #308's CLI and tables intact.
compare.TABLES = {variant: [128, 512, 2048, 8192] for variant in patches.VARIANTS}
previous_compare = sys.modules.get("compare")
sys.modules["compare"] = compare
try:
    runner = load("runner310", ROOT / "docs/research/issue-308/run.py")
finally:
    if previous_compare is None:
        del sys.modules["compare"]
    else:
        sys.modules["compare"] = previous_compare
runner.patch = patches.patch


def validate(data, kind, case, repeats):
    header, calls = compare.validate(data, kind, case, compare.TABLES["current"], repeats)
    if kind == "embedding":
        for row in calls:
            m = row["metrics"]
            compare.oracle.require(m["readback_elements"] == [f["batch_size"] * 768 for f in m["forwards"]],
                                   "actual readback differs from pooled output")
    return header, calls


def parse_observations(stderr):
    return [json.loads(line.removeprefix("mask310 ")) for line in
            stderr.splitlines() if line.startswith("mask310 ")]


def verify_cleanup(events):
    actions = set()
    failures = 0
    for event in events:
        if event["event"] in ("buffer_clear", "compile_clear"):
            actions.add(event["event"])
        elif event["event"] == "cleanup":
            if event["failed"]:
                compare.oracle.require(actions == {"buffer_clear", "compile_clear"},
                                       "failure did not clear both caches")
                failures += 1
            actions.clear()
    compare.oracle.require(failures == 72, "missing injected cleanup failures")


def model_conditions(context_path):
    context = json.loads(context_path.read_text())
    for key in ("machine", "ram_bytes", "os", "rust", "cargo", "xcode", "metal", "build", "load_conditions"):
        compare.oracle.require(bool(context.get(key)), f"missing host condition: {key}")
    known = json.loads((ROOT / "docs/research/issue-307/results/numerical-rechecked/models.json").read_text())
    model_dirs = {kind: Path(context["model_dirs"][kind]).resolve() for kind in runner.TESTS}
    models = {}
    for kind, directory in model_dirs.items():
        compare.oracle.require(directory.name == known[kind]["revision"], "wrong snapshot revision")
        for name in ("model.safetensors", "config.json", "tokenizer.json"):
            path = directory / name
            models[str(path)] = runner.host.digest(path)
            compare.oracle.require(models[str(path)] == known[kind]["files"][name]["sha256"], "model differs from #307")
    return context, model_dirs, models


def guarded_verification(args, cwd, output, name, baseline, started, deadline):
    """Apply the existing measurement guards to the owned verification group."""
    compare.oracle.require(time.monotonic() < deadline, "verification deadline before next process")
    samples, status = [], {"complete": False}
    with (output / f"{name}.stdout").open("w") as stdout, (output / f"{name}.stderr").open("w") as stderr:
        process = runner.subprocess.Popen(args, cwd=cwd, env=dict(os.environ),
                                          stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while process.poll() is None:
                row = runner.host.snapshot(process.pid)
                row["elapsed_seconds"] = time.monotonic() - started
                samples.append(row)
                compare.oracle.require(time.monotonic() < deadline and row["rss_bytes"] <= 8*1024**3
                    and row["pressure"] == 1 and row["swap_bytes"] - baseline["swap_bytes"] <= 256*1024**2,
                    "verification guard reached; retain partial evidence")
                time.sleep(.1)
            status["exit_code"] = process.returncode
            compare.oracle.require(process.returncode == 0, "libtest failed; retain partial evidence")
            status["complete"] = True
        except BaseException as error:
            status["reason"] = str(error)
            raise
        finally:
            # A second interrupt must not abandon descendants or evidence writes.
            handlers = {sig:signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                try:
                    runner.host.reclaim(process)
                    status["group_absent"] = True
                except BaseException as error:
                    status.update(complete=False, reason=str(error))
                    raise
                finally:
                    status["exit_code"] = process.poll()
                    runner.save(output / f"{name}.monitor.json", samples)
                    runner.save(output / f"{name}.status.json", status)
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)


def verify(prepared, context_path, output):
    context, model_dirs, models = model_conditions(context_path)
    plan, plan_sha256 = runner.verify_plan(prepared)
    compare.oracle.require(not output.is_relative_to(ROOT) and not output.is_relative_to(prepared),
                           "use new external verification directory")
    output.mkdir(parents=True, exist_ok=False)
    runner.save(output / "context.json", dict(context, models=models, plan_sha256=plan_sha256,
                guards={"seconds":1800,"rss_bytes":8*1024**3,"swap_increase_bytes":256*1024**2,"pressure":1}))
    started = time.monotonic()
    deadline = started + 1800
    try:
        baseline = runner.host.snapshot(-1)
        runner.save(output / "baseline.json", baseline)
        compare.oracle.require(baseline["pressure"] == 1, "pressure abnormal before verification")
        for variant, binary in plan["binaries"].items():
            tests = ["mlx_cache::testing::runtime::real_model_cleanup_memory"]
            if variant == "current":
                tests.append("modernbert::model::spike310::all_mask_elements_and_costs")
            for test in tests:
                name = f"{variant}-{'cleanup' if 'cleanup' in test else 'mask'}"
                guarded_verification(["/usr/bin/time", "-l", binary["path"], test,
                                      "--exact", "--ignored", "--nocapture", "--test-threads=1"],
                                     prepared / variant, output, name, baseline, started, deadline)
                compare.oracle.require("1 passed; 0 failed; 0 ignored" in (output / f"{name}.stdout").read_text(),
                                       "missing selected test execution")
                events = parse_observations((output / f"{name}.stderr").read_text())
                if "cleanup" in test:
                    verify_cleanup(events)
                    paths = [e for e in events if e["event"] == "model_files"]
                    compare.oracle.require(len(paths) == 2 and {e["kind"] for e in paths} == set(runner.TESTS), "missing cleanup model identity")
                    for entry in paths:
                        compare.oracle.require(sorted(Path(p).resolve() for p in entry["paths"]) ==
                            sorted((model_dirs[entry["kind"]] / p).resolve() for p in ("model.safetensors", "config.json", "tokenizer.json")),
                            "cleanup model differs from hashed context")
                else:
                    masks = [e for e in events if e["event"] == "mask_elements"]
                    compare.oracle.require(len(masks) == 48, "incomplete full-mask verification")
                runner.save(output / f"{name}.json", events)
        runner.verify_plan(prepared, plan)
        compare.oracle.require(models == {p:runner.host.digest(Path(p)) for p in models}, "model changed")
        runner.save(output / "complete.json", {"verification_complete":True,
                    "files":{p.name:runner.host.digest(p) for p in output.iterdir() if p.is_file()}})
    except BaseException as error:
        runner.save(output / "incomplete.json", {"verification_complete":False,"reason":str(error)})
        raise


def measure(prepared, context_path, output, args):
    selected = runner.select_cases(args.kinds, args.cases)
    compare.oracle.require(args.trials == 1 or args.trial_start == 0, "three trials must start at zero")
    trials = range(args.trial_start, args.trial_start + args.trials)
    plan, plan_sha256 = runner.verify_plan(prepared)
    compare.oracle.require(args.repeats >= 9 and args.sustained_repeats >= 9 and args.budget > 0,
                           "observe at least one eight-forward compile epoch")
    context, model_dirs, models = model_conditions(context_path)
    compare.oracle.require(not output.is_relative_to(ROOT) and not output.is_relative_to(prepared), "use new external evidence directory")
    output.mkdir(parents=True, exist_ok=False)
    context.update(plan_sha256=plan_sha256, models=models,
                   kinds=args.kinds, cases=selected, repeats=args.repeats,
                   sustained_repeats=args.sustained_repeats, trials=list(trials),
                   embedding_budget=args.budget, reranker_budget=256000, pause_ms=1,
                   variants=list(compare.TABLES), buffer_candidate_bytes=64*1024**2,
                   compile_candidate_forward_interval=8,
                   tolerances="embedding existing fixture criteria; reranker differences are diagnostics, no #308 tolerance transfer",
                   guards={"seconds":1800,"rss_bytes":8*1024**3,"swap_increase_bytes":256*1024**2,"pressure":1})
    runner.save(output / "context.json", context)
    runner.started = time.monotonic()
    deadline = runner.started + 1800
    baseline = runner.host.snapshot(-1)
    summaries = {}
    try:
        compare.oracle.require(baseline["pressure"] == 1, "pressure abnormal before measurement")
        for kind, cases in selected.items():
            for case in cases:
                count = args.sustained_repeats if case in ("mixed", "multichunk") else args.repeats
                groups, records, extra, monitors = {}, {}, {}, {}
                for trial in trials:
                    variants = list(compare.TABLES)
                    # Shift by two across the five alternatives; save actual order.
                    shift = (2 * trial) % len(variants)
                    for variant in variants[shift:] + variants[:shift]:
                        name = f"{kind}-{case}-{trial}-{variant}"
                        data, events = runner.observe(Path(plan["binaries"][variant]["path"]), kind, case, count,
                                              args.budget, output, name, model_dirs, baseline, deadline,
                                              event_prefix="mask310 ")
                        groups[trial, variant] = validate(data, kind, case, count)
                        records[trial, variant] = data
                        extra[trial, variant] = events
                        monitors[trial, variant] = json.loads((output / f"{name}.monitor.json").read_text())
                        if case != "empty":
                            compare.oracle.require(any(r["event"] == "mask_graph" for r in extra[trial, variant]), "missing mask observations")
                            compare.oracle.require(any(r["event"] == "cleanup" for r in extra[trial, variant]), "missing cleanup observations")
                diagnostics = {}
                for variant in list(compare.TABLES)[1:]:
                    diagnostics[variant] = [compare.difference(groups[i, "current"], groups[i, variant], kind) for i in trials]
                    if kind == "embedding":
                        diagnostics[variant].append({"queries": [compare.oracle.vector_diff(u, v) for i in trials for u, v in
                            zip(records[i, "current"][-1]["values"], records[i, variant][-1]["values"])]})
                summaries[f"{kind}/{case}"] = {"diagnostics":diagnostics,
                    "measurements":{variant:compare.timings([groups[i, variant][1] for i in trials], kind) for variant in compare.TABLES},
                    "mask_cleanup_series":{variant:[extra[i, variant] for i in trials] for variant in compare.TABLES},
                    "host_series":{variant:[monitors[i, variant] for i in trials] for variant in compare.TABLES}}
                runner.save(output / "summary.json", summaries)
        runner.verify_plan(prepared, plan)
        compare.oracle.require(models == {p:runner.host.digest(Path(p)) for p in models}, "model changed")
        runner.save(output / "complete.json", {"collection_complete":True,
            "acceptance":"requires full-mask/cleanup verification, numerical assessment, bottleneck and recommendation; not adoption",
            "files":{p.name:runner.host.digest(p) for p in output.iterdir() if p.is_file()}})
    except BaseException as error:
        runner.save(output / "incomplete.json", {"collection_complete":False, "reason":str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("output", type=Path)
    prep.add_argument("--cache-base", type=Path)
    validation = sub.add_parser("verify")
    validation.add_argument("prepared", type=Path)
    validation.add_argument("context", type=Path)
    validation.add_argument("output", type=Path)
    run = sub.add_parser("measure")
    run.add_argument("prepared", type=Path)
    run.add_argument("context", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--repeats", type=int, default=9)
    run.add_argument("--sustained-repeats", type=int, default=31)
    run.add_argument("--budget", type=int, default=8192)
    run.add_argument("--trials", type=int, choices=(1, 3), default=3)
    run.add_argument("--trial-start", type=int, choices=(0, 1, 2), default=0)
    run.add_argument("--kinds", nargs="+", choices=tuple(runner.TESTS), default=list(runner.TESTS))
    run.add_argument("--cases", nargs="+", choices=tuple(dict.fromkeys((*compare.EMBED_CASES,*compare.RERANK_CASES))))
    args = parser.parse_args()
    if args.mode == "prepare":
        runner.prepare(args.output.resolve(), args.cache_base.resolve() if args.cache_base else None, patch_current=True)
    elif args.mode == "verify":
        verify(args.prepared.resolve(), args.context, args.output.resolve())
    else:
        measure(args.prepared.resolve(), args.context, args.output.resolve(), args)


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}; retain partial evidence")
    signal.signal(signal.SIGTERM, interrupt)
    main()
