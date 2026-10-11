"""Explicit #366 host lane; reuse #307 inference and amici-owned evaluation."""
import argparse
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

from processes import owned_run
from compare import require, SCORE_MAX_ABS
from run import HERE, ROOT, command, sha, source, write
from search_quality import run as search


def numerical(out, root, candidate):
    """Release reference-process memory before building the consumer."""
    code = ("import sys; from pathlib import Path; from run import numerical; "
            "numerical(Path(sys.argv[1]), root=Path(sys.argv[2]), candidate=sys.argv[3])")
    owned_run([os.sys.executable, "-c", code, str(out), str(root), candidate], cwd=HERE, check=True)


def identity(root):
    files = source(root)
    return {"head":command(["git","rev-parse","HEAD"], cwd=root),
            "dirty":bool(command(["git","status","--porcelain"], cwd=root)),
            "source":files,
            "source_sha256":hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "cargo_lock_sha256":files["Cargo.lock"]}


def prepare(candidate, out, root=ROOT):
    require(re.fullmatch(r"[0-9a-f]{40}", candidate) is not None, "specify a full candidate commit SHA")
    try:
        commit = command(["git","rev-parse","--verify","--quiet",candidate+"^{commit}"], cwd=root)
    except subprocess.CalledProcessError as error:
        raise ValueError("candidate is not a commit") from error
    require(commit == candidate, "candidate is not a commit")
    checkout = out/"candidate"
    # Local clone copies committed files only. Never modify the original checkout.
    subprocess.run(["git","clone","--quiet","--no-hardlinks","--no-checkout",str(root),str(checkout)], check=True)
    subprocess.run(["git","checkout","--quiet","--detach",candidate], cwd=checkout, check=True)
    observed = identity(checkout)
    require(observed["head"] == candidate and not observed["dirty"], "candidate checkout differs")
    require(observed["source"]["docs/research/issue-307/inputs.json"] == sha(HERE/"inputs.json"),
            "candidate input definition differs; investigate without replacing the fixed inputs")
    return checkout, observed


def diagnostics(value):
    """Summarize existing diagnostics, without a new product-quality tolerance."""
    if isinstance(value, dict):
        flags = [v for k,v in value.items()
                 if k in ("within_criteria", "raw_equal", "wrapper_equal", "tokens_equal")]
        for key in ("native_score_abs", "public_vs_direct_score_abs", "score_abs"):
            if key in value:
                scores = value[key] if isinstance(value[key], list) else [value[key]]
                flags.append(bool(scores) and all(score <= SCORE_MAX_ABS for score in scores))
        return all(flags) and all(diagnostics(v) for v in value.values())
    if isinstance(value, list):
        return all(diagnostics(v) for v in value)
    return True


def numerical_result(stage, kind):
    result = json.loads((stage/f"comparison-{kind}.json").read_text())
    expected = {"embedding":(23,27,10,12), "reranker":(12,16,6,6)}[kind]
    counts = (len(result["same_tokens"]), sum(len(b["rows"]) for b in result["same_tokens"]),
              len(result["tokenization"]), len(result["padding"]))
    require(counts == expected, "fixed numerical observation count differs")
    return diagnostics(result)


def finish_stage(stage):
    completion = json.loads((stage/"complete.json").read_text())
    require(completion["execution"] == "complete" and completion["artifacts"], "stage incomplete")
    for name,digest in completion["artifacts"].items():
        require(sha(stage/name) == digest, "stage artifact changed")


def execute(candidate, out, lane, root=ROOT):
    harness = identity(root)
    write(out/"started.json", {"execution":"incomplete", "candidate_sha":candidate, "lane":lane,
                              "started_utc":datetime.now(timezone.utc).isoformat(), "harness":harness})
    stage = "prepare"
    try:
        checkout, snapshot = prepare(candidate, out, root)
        write(out/"candidate.json", {"candidate_sha":candidate, "snapshot":snapshot,
                                   "inputs_sha256":snapshot["source"]["docs/research/issue-307/inputs.json"],
                                   "requirements_sha256":sha(HERE/"requirements.txt")})
        if lane == "light":
            stage = "model-free-tests"
            with (out/"light.private.log").open("x") as log:
                for location, pattern in ((str(HERE), "test_*.py"), ("scripts", "test_candidate_quality.py")):
                    owned_run([os.sys.executable,str(HERE/"processes.py"),"discover","-s",location,"-p",pattern],
                              cwd=root, stdout=log, stderr=log, check=True, graceful=True)
            agreement = None
        else:
            require(os.environ.get("CODEX_SANDBOX") != "seatbelt", "run on unsandboxed host")
            require(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple Silicon host required")
            stage = "numerical"
            numeric = out/stage
            numeric.mkdir()
            numerical(numeric, root=checkout, candidate=candidate)
            finish_stage(numeric)
            stage = "amici"
            consumer = out/stage
            consumer.mkdir()
            search(consumer, candidate=candidate, root=checkout, verify_source=True)
            finish_stage(consumer)
            agreement = {kind:numerical_result(numeric, kind) for kind in ("embedding", "reranker")}
            agreement["amici_baseline"] = json.loads((consumer/"search-comparison.json").read_text())["verify_existing_exit_code"] == 0
        require(identity(checkout) == snapshot and identity(root) == harness, "source changed during validation")
        write(out/"complete.json", {"execution":"complete", "candidate_sha":candidate, "lane":lane,
                                   "model_measurement":lane == "model", "agreement":agreement,
                                   "finished_utc":datetime.now(timezone.utc).isoformat(),
                                   "artifacts":{name:sha(out/name) for name in
                                                (["started.json","candidate.json"] +
                                                 (["numerical/complete.json","amici/complete.json"] if lane == "model" else []))},
                                   "scope":"Independent fixed-input numerics and amici reference composition; full application quality unverified."})
        return 1 if agreement is not None and not all(agreement.values()) else 0
    except BaseException as error:
        (out/"failure.private.log").write_text(traceback.format_exc())
        write(out/"incomplete.json", {"execution":"incomplete", "candidate_sha":candidate, "lane":lane,
                                     "stage":stage, "error_type":type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="Full committed SHA available locally and from public origin for amici")
    parser.add_argument("--lane", choices=("light", "model"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory outside checkout")
    args = parser.parse_args()
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT), "evidence directory must be outside checkout")
    out.mkdir(parents=True, exist_ok=False)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("validation interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(execute(args.candidate, out, args.lane))


if __name__ == "__main__":
    main()
