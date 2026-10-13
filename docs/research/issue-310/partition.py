"""Collect one #310 candidate per guarded phase, then verify the complete matrix."""
import argparse
import hashlib
import importlib.util
import json
import signal
import time
from pathlib import Path

VARIANTS = ("current", "shared", "gpu", "buffer", "compile")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(repo):
    spec = importlib.util.spec_from_file_location("partition310_run", repo / "docs/research/issue-310/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def condition_key(context):
    return {k: v for k, v in context.items() if k not in ("variant", "trial", "name")}


def collect(api, args):
    compare, runner = api.compare, api.runner
    runner.select_cases([args.kind], [args.case])
    compare.oracle.require(args.repeats >= 9 and args.budget > 0, "invalid observation plan")
    plan, plan_sha256 = runner.verify_plan(args.prepared)
    context, model_dirs, models = api.model_conditions(args.context)
    compare.oracle.require(not args.output.is_relative_to(api.ROOT) and not args.output.is_relative_to(args.prepared), "use new external output")
    args.output.mkdir(parents=True, exist_ok=False)
    name = f"{args.kind}-{args.case}-{args.trial}-{args.variant}"
    context.update(plan_sha256=plan_sha256, collector_sha256=digest(Path(__file__)),
                   models=models, kind=args.kind, case=args.case, repeats=args.repeats,
                   budget=args.budget, variant=args.variant, trial=args.trial, name=name,
                   guards={"seconds":1800,"rss_bytes":8*1024**3,"pressure":1,"swap_increase_bytes":256*1024**2})
    runner.save(args.output / "context.json", context)
    runner.started = time.monotonic()
    baseline = runner.host.snapshot(-1)
    try:
        compare.oracle.require(baseline["pressure"] == 1, "pressure abnormal before phase")
        data, events = runner.observe(Path(plan["binaries"][args.variant]["path"]), args.kind, args.case,
                              args.repeats, args.budget, args.output, name, model_dirs,
                              baseline, runner.started + 1800, event_prefix="mask310 ")
        api.validate(data, args.kind, args.case, args.repeats)
        if args.case != "empty":
            compare.oracle.require(any(r["event"] == "mask_graph" for r in events), "missing mask observations")
            compare.oracle.require(any(r["event"] == "cleanup" for r in events), "missing cleanup observations")
        runner.verify_plan(args.prepared, plan)
        compare.oracle.require(models == {p: digest(Path(p)) for p in models}, "model changed")
        runner.save(args.output / "phase-complete.json", {
            "phase_complete":True, "acceptance":"one candidate only; not complete comparison or adoption",
            "files":{p.name:digest(p) for p in args.output.iterdir() if p.is_file()}})
    except BaseException as error:
        runner.save(args.output / "incomplete.json", {"phase_complete":False,"reason":str(error)})
        raise


def combine(api, phases, output):
    compare, runner = api.compare, api.runner
    records, events, monitors, groups, identity, evidence = {}, {}, {}, {}, None, []
    for directory in phases:
        complete_bytes = (directory / "phase-complete.json").read_bytes()
        complete = json.loads(complete_bytes)
        complete_sha256 = hashlib.sha256(complete_bytes).hexdigest()
        del complete_bytes
        compare.oracle.require(complete["phase_complete"] is True, "incomplete phase")
        def checked_bytes(name):
            compare.oracle.require(Path(name).name == name, "phase evidence changed")
            compare.oracle.require(name in complete["files"], "missing hashed phase evidence")
            content = (directory / name).read_bytes()
            compare.oracle.require(hashlib.sha256(content).hexdigest() == complete["files"][name], "phase evidence changed")
            return content

        context = json.loads(checked_bytes("context.json"))
        runner.select_cases([context["kind"]], [context["case"]])
        compare.oracle.require(type(context["repeats"]) is int and context["repeats"] >= 9
                               and context["budget"] > 0, "invalid phase plan")
        compare.oracle.require(type(context["trial"]) is int and context["trial"] in range(3)
                               and context["variant"] in VARIANTS, "invalid phase identity")
        current = condition_key(context)
        if identity is None:
            identity = current
        compare.oracle.require(current == identity, "different phase conditions")
        key = context["trial"], context["variant"]
        compare.oracle.require(key not in groups, "duplicate phase")
        name = context["name"]
        required = {"context.json", *(f"{name}.{suffix}" for suffix in ("json","stdout","stderr","monitor.json","status.json"))}
        compare.oracle.require(required <= set(complete["files"]), "missing hashed phase evidence")
        status = json.loads(checked_bytes(f"{name}.status.json"))
        compare.oracle.require(status.get("complete") is True and status.get("exit_code") == 0
                               and status.get("group_absent") is True, "phase process not complete or reclaimed")
        data = json.loads(checked_bytes(f"{name}.json"))
        groups[key] = api.validate(data, context["kind"], context["case"], context["repeats"])
        records[key] = data
        events[key] = api.parse_observations(checked_bytes(f"{name}.stderr").decode())
        monitors[key] = json.loads(checked_bytes(f"{name}.monitor.json"))
        # Hash-only files (stdout and any additional evidence) need no retention.
        parsed = {"context.json", *(f"{name}.{suffix}" for suffix in ("json", "status.json", "stderr", "monitor.json"))}
        for filename in complete["files"].keys() - parsed:
            checked_bytes(filename)
        evidence.append({"directory":str(directory),"phase_complete_sha256":complete_sha256})
    expected = {(trial, variant) for trial in range(3) for variant in VARIANTS}
    compare.oracle.require(set(groups) == expected, "require all five candidates and all three trials")
    kind, case = identity["kind"], identity["case"]
    diagnostics = {}
    for variant in VARIANTS[1:]:
        diagnostics[variant] = [compare.difference(groups[i,"current"], groups[i,variant], kind) for i in range(3)]
        if kind == "embedding":
            diagnostics[variant].append({"queries":[compare.oracle.vector_diff(u,v) for i in range(3) for u,v in
                zip(records[i,"current"][-1]["values"], records[i,variant][-1]["values"])]})
    output.mkdir(parents=True, exist_ok=False)
    runner.save(output / "context.json", dict(identity, phases=evidence))
    runner.save(output / "summary.json", {f"{kind}/{case}":{
        "diagnostics":diagnostics,
        "measurements":{v:compare.timings([groups[i,v][1] for i in range(3)],kind) for v in VARIANTS},
        "mask_cleanup_series":{v:[events[i,v] for i in range(3)] for v in VARIANTS},
        "host_series":{v:[monitors[i,v] for i in range(3)] for v in VARIANTS}}})
    runner.save(output / "complete.json", {"collection_complete":True,
        "acceptance":"complete partitioned collection; numerical assessment and adoption remain separate",
        "files":{p.name:digest(p) for p in output.iterdir() if p.is_file()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    modes = parser.add_subparsers(dest="mode", required=True)
    phase = modes.add_parser("collect")
    for name in ("prepared", "context", "output"):
        phase.add_argument(name, type=Path)
    phase.add_argument("--kind", choices=("embedding","reranker"), required=True)
    phase.add_argument("--case", required=True)
    phase.add_argument("--variant", choices=VARIANTS, required=True)
    phase.add_argument("--trial", type=int, choices=range(3), required=True)
    phase.add_argument("--repeats", type=int, default=31)
    phase.add_argument("--budget", type=int, default=8192)
    merged = modes.add_parser("combine")
    merged.add_argument("output", type=Path)
    merged.add_argument("phases", type=Path, nargs="+")
    args = parser.parse_args()
    api = load(args.repo.resolve())
    if args.mode == "collect":
        args.prepared, args.output = args.prepared.resolve(), args.output.resolve()
        collect(api,args)
    else:
        compare = api.compare
        output = args.output.resolve()
        compare.oracle.require(not output.is_relative_to(api.ROOT), "use external output")
        combine(api, [p.resolve() for p in args.phases], output)


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}; retain partial evidence")
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    main()
