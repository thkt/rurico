"""Collect #316 context/rerank observations; no builds, downloads or adoption."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("host318", ROOT / "docs/research/issue-318/run.py")
host318 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host318)
STRATEGIES = ("baseline", "query-prefix", "chunk-neighbors", "windows")
REVISION = "bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3"


def quality(order, relevance):
    """Rerank-only metrics over the same full candidate set; NOT amici retrieval."""
    total = sum(relevance.values())
    if not total:
        raise ValueError("query has no relevant candidate")
    result = {}
    for k in (1, 5, 10):
        grades = [relevance[key] for key in order[:k]]
        dcg = sum((2**grade-1)/math.log2(i+2) for i, grade in enumerate(grades))
        ideal = sorted(relevance.values(), reverse=True)[:k]
        idcg = sum((2**grade-1)/math.log2(i+2) for i, grade in enumerate(ideal))
        result[f"Recall@{k}"] = sum(grade > 0 for grade in grades)/total
        result[f"nDCG@{k}"] = dcg/idcg
    return result


def summarize(records, inputs, strategy, mode):
    if len(records) != 1 + len(inputs["queries"])*(4 if mode == "model" else 1):
        raise ValueError("missing or excess call records")
    context, *calls = records
    if (context["event"], context["mode"], context["strategy"]) != ("context", mode, strategy):
        raise ValueError("wrong experiment identity")
    ids = [doc["id"] for doc in inputs["documents"]]
    result = []
    for query_index, query in enumerate(inputs["queries"]):
        count = 4 if mode == "model" else 1
        group = calls[query_index*count:(query_index+1)*count]
        for trial, call in enumerate(group):
            if (call["event"], call["query"], call["trial"], call["cold"]) != ("call", query["id"], trial, trial == 0):
                raise ValueError("wrong query/trial order")
            if [doc["document"] for doc in call["documents"]] != ids:
                raise ValueError("candidate set/order differs")
            windows = [w for doc in call["documents"] for w in doc["windows"]]
            if not windows or any(not w["ids"] or len(w["ids"]) != len(w["mask"]) or len(w["ids"]) > 8192 for w in windows):
                raise ValueError("invalid token/mask lengths")
            for doc in call["documents"]:
                for window in doc["windows"]:
                    length = len(window["ids"])
                    if any(len(window[key]) != length for key in ("sequence_ids", "offsets_bytes", "special_tokens_mask")):
                        raise ValueError("missing token boundary observations")
                    if strategy != "baseline":
                        expected = {i for i, key in enumerate(("raw_query_tokens", "raw_document_tokens")) if doc[key]}
                        if not expected <= set(window["sequence_ids"]):
                            raise ValueError("candidate lost query/document boundary")
                        if (window["ids"][0], window["ids"][-1]) != (doc["raw_ids"][0], doc["raw_ids"][-1]):
                            raise ValueError("candidate changed BOS/EOS")
                        if sum(window["special_tokens_mask"]) != sum(doc["raw_special_tokens_mask"]):
                            raise ValueError("candidate changed special token count")
            if mode == "model":
                if len(call["forwards"]) != len(windows) or any(not math.isfinite(w["score"]) for w in windows):
                    raise ValueError("missing forward/nonfinite score")
                if any(f["rows"] != 1 or f["bucket"] != next(
                        bucket for bucket in (128,512,2048,8192) if len(w["ids"]) <= bucket)
                       for f, w in zip(call["forwards"], windows)):
                    raise ValueError("padding/forward policy differs")
            elif call["forwards"] or call["metal_peak_bytes"] is not None:
                raise ValueError("plan cannot claim model execution")
        entry = {"strategy": strategy, "query": query["id"], "model_measured": mode == "model",
                 "retained": [{"document": d["document"], "raw_query_tokens":d["raw_query_tokens"],
                    "raw_document_tokens":d["raw_document_tokens"],
                    "evidence_any_window": any(w["evidence_retained"] for w in d["windows"]),
                    "windows": len(d["windows"])} for d in group[0]["documents"]]}
        if mode == "model":
            observations = []
            for call in group:
                aggregations = {}
                for aggregate in (("max", "mean") if strategy == "windows" else ("single",)):
                    scores = []
                    for doc in call["documents"]:
                        values = [w["score"] for w in doc["windows"]]
                        scores.append(max(values) if aggregate == "max" else statistics.mean(values))
                    indices = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
                    order = [ids[i] for i in indices]
                    aggregations[aggregate] = {"scores":dict(zip(ids, scores)), "order":order,
                        "ranks":{key:order.index(key)+1 for key in ids}, "quality":quality(order, query["relevance"])}
                observations.append({"trial":call["trial"], "cold":call["cold"], "wall_ns":call["wall_ns"],
                    "forward_count":len(call["forwards"]), "metal_peak_bytes":call["metal_peak_bytes"],
                    "aggregations":aggregations})
            entry["observations"] = observations
            entry["warm"] = {key:{"min":min(values), "median":statistics.median(values), "max":max(values)}
                for key in ("wall_ns", "forward_count", "metal_peak_bytes")
                for values in [[r[key] for r in observations[1:]]]}
        result.append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("context", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("plan", "model"), default="model")
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("positive experiment budget required; not a product limit")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("Apple Silicon host required")
    if args.mode == "model" and os.environ.get("CODEX_SANDBOX") == "seatbelt":
        raise ValueError("model mode requires unsandboxed Metal")
    context = json.loads(args.context.read_text())
    for key in ("machine", "ram_bytes", "os", "rust", "cargo", "xcode", "metal", "build", "load_conditions", "reranker_snapshot", "parent_issue_evidence"):
        if not context.get(key):
            raise ValueError(f"missing host context: {key}")
    snapshot = Path(context["reranker_snapshot"]).resolve(strict=True)
    if snapshot.name != REVISION:
        raise ValueError("fixed snapshot revision required")
    binary = args.binary.resolve(strict=True)
    out = args.output.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError("use new evidence directory outside checkout AND old run")
    model_files = [snapshot/name for name in ("model.safetensors", "config.json", "tokenizer.json")]
    source = [ROOT/"Cargo.toml", ROOT/"Cargo.lock", ROOT/"build.rs", args.context.resolve()]
    source += sorted((ROOT/"src").rglob("*.rs"))
    source += sorted((ROOT/"docs/research/issue-316").glob("*.*"))
    source += [ROOT/"tests/fixtures/long_context/generate.py", ROOT/"docs/research/issue-318/run.py"]
    paths = sorted(set(source + model_files + [binary]))
    before = {str(path):host318.digest(path) for path in paths}
    diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
    out.mkdir(parents=True, exist_ok=False)
    (out/"source.patch").write_bytes(diff)
    for path in (ROOT/"docs/research/issue-316").glob("*.*"):
        (out/path.name).write_bytes(path.read_bytes())
    header = host318.weight_header(model_files[0])
    if not header or any(v["dtype"] != "F32" for v in header.values()):
        raise ValueError("fixed F32 weights required")
    # Invoke the independent fixture producer unchanged, with this same interpreter/env.
    argv = [sys.executable, str(ROOT/"tests/fixtures/long_context/generate.py")]
    generated = subprocess.check_output(argv, timeout=30)
    (out/"inputs.json").write_bytes(generated)
    inputs = json.loads(generated)
    context.update(start_commit=host318.command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        source_diff_sha256=hashlib.sha256(diff).hexdigest(), files=before,
        inputs_sha256=hashlib.sha256(generated).hexdigest(), mode=args.mode,
        python={"executable":sys.executable,"version":platform.python_version(),"fixture_argv":argv},
        measurement_seconds=args.seconds, weight_header=header)
    (out/"context.json").write_text(json.dumps(context, indent=2)+"\n")
    deadline = time.monotonic()+args.seconds
    summary, samples, reference_tokens = [], [], None
    for strategy in STRATEGIES:
        if time.monotonic() >= deadline:
            raise RuntimeError("budget reached; retain incomplete evidence")
        record = out/f"{strategy}.jsonl"
        argv = [str(binary), args.mode, str(out/"inputs.json"), strategy, str(record)]
        env = dict(os.environ, RURICO_316_MODEL_DIR=str(snapshot))
        status = {"argv":argv,"complete":False}
        with (out/f"{strategy}.stdout").open("x") as stdout, (out/f"{strategy}.stderr").open("x") as stderr:
            process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                while process.poll() is None:
                    observation = host318.snapshot(process.pid)
                    observation.update(strategy=strategy, monotonic=time.monotonic())
                    samples.append(observation)
                    if time.monotonic() >= deadline:
                        raise RuntimeError("budget reached; retain incomplete evidence")
                    time.sleep(.1)
                status["exit_code"] = process.returncode
                if process.returncode:
                    raise RuntimeError(f"{strategy} exited {process.returncode}")
            finally:
                try:
                    host318.reclaim(process)
                    status["group_absent"] = True
                finally:
                    (out/f"{strategy}.status.json").write_text(json.dumps(status, indent=2)+"\n")
                    (out/"monitor.json").write_text(json.dumps(samples, indent=2)+"\n")
        rows = [json.loads(line) for line in record.read_text().splitlines()]
        entries = summarize(rows, inputs, strategy, args.mode)
        raw_tokens = [[(d["raw_ids"], d["raw_mask"], d["raw_sequence_ids"], d["raw_offsets_bytes"], d["raw_special_tokens_mask"])
                       for d in call["documents"]] for call in rows[1:]]
        if reference_tokens is None:
            reference_tokens = raw_tokens
        elif raw_tokens != reference_tokens:
            raise ValueError("raw tokenization differs across strategies; comparison refused")
        process_rss_max_bytes = max((s["rss_bytes"] for s in samples if s["strategy"] == strategy), default=None)
        for entry in entries:
            entry["process_rss_max_bytes"] = process_rss_max_bytes
        summary.extend(entries)
        status["complete"] = True
        (out/f"{strategy}.status.json").write_text(json.dumps(status, indent=2)+"\n")
    after = {str(path):host318.digest(path) for path in paths}
    if before != after or diff != subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT):
        raise RuntimeError("source/model/binary/context changed")
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (out/"complete.json").write_text(json.dumps({"collection_complete":True,
        "model_measured":args.mode == "model", "acceptance":"requires independent analysis; no thresholds adopted",
        "files":{p.name:host318.digest(p) for p in out.iterdir() if p.is_file()}},indent=2)+"\n")


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}; incomplete evidence retained")
    signal.signal(signal.SIGTERM, interrupt)
    main()
