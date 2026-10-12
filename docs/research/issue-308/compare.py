"""Compare #308 embedding public-path and reranker internal-path records; no adoption or baseline rewrite."""
import importlib.util
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load("oracle307", ROOT / "docs/research/issue-307/compare.py")
length309 = load("length309", ROOT / "docs/research/issue-309/run.py")
TABLES = {"current": [128, 512, 2048, 8192],
          "4096": [128, 512, 2048, 4096, 8192],
          "1024-4096": [128, 512, 1024, 2048, 4096, 8192]}
EMBED_CASES = tuple(f"boundary{b}" for b in (128, 512, 1024, 2048, 4096)) + ("limit", "mixed", "multichunk")
RERANK_CASES = EMBED_CASES[:-1] + ("same", "empty")


def ns(duration):
    return duration["secs"] * 10**9 + duration["nanos"]


def distribution(values):
    oracle.require(bool(values), "missing observations")
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def validate(records, kind, case, bounds, repeats):
    headers = [r for r in records if r.get("event") == "input"]
    calls = [r for r in records if r.get("event") == "call"]
    oracle.require(len(headers) == 1 and len(calls) == repeats, "incomplete records")
    header = headers[0]
    oracle.require(header.get("case", header.get("workload")) == case, "wrong workload")
    for i, row in enumerate(calls):
        oracle.require(row["repeat"] == i, "missing/duplicate repeat")
        oracle.require(row["metal_peak_bytes"] >= 0, "invalid peak")
        if kind == "embedding":
            oracle.require(header["bounds"] == bounds, "wrong table")
            oracle.require(len(row["documents"]) == len(header["chunk_counts"]), "lost document")
            for index, (doc, count) in enumerate(zip(row["documents"], header["chunk_counts"])):
                oracle.require(doc["id"] == index and doc["chunk_ids"] == [f"c{j}" for j in range(count)]
                               and len(doc["values"]) == count, "document/chunk identity")
                for vector in doc["values"]:
                    oracle.require(len(vector) == 768, "dimension")
                    oracle.finite(vector)
            m = row["metrics"]
            lengths = [len(c) for c in header["chunks"]]
            hist = [sum(next(b for b in bounds if b >= n) == bound for n in lengths) for bound in bounds]
            oracle.require(sum(header["chunk_counts"]) == len(lengths) == m["num_chunks"]
                           and m["bucket_hist"] == hist and m["real_tokens"] == sum(lengths), "planner accounting")
            padded = sum(n * b for n, b in zip(hist, bounds))
            shapes = []
            budget = header["options"]["token_budget"]
            for count, bound in zip(hist, bounds):
                size = max(1, budget // bound)
                while count:
                    rows = min(size, count)
                    shapes.append((rows, bound))
                    count -= rows
            oracle.require([(f["batch_size"], f["sequence_length"]) for f in m["forwards"]] == shapes
                           and m["padded_tokens"] == padded and m["pause_count"] == len(shapes), "forward/pause accounting")
            oracle.require(ns(m["wall"]) >= ns(m["pause"]) >= 0, "invalid wall/pause")
        else:
            count = len(header["pairs"])
            oracle.require(row["candidate"] is False and row["state"] == ("cold" if i == 0 else "warm"), "reranker route/state")
            oracle.require(len(row["lengths"]) == len(row["logits"]) == len(row["scores"]) == count, "lost pair")
            oracle.require(row["order"] == sorted(range(count), key=lambda j: -row["scores"][j]), "rank identities")
            if count:
                oracle.finite(row["logits"])
                oracle.finite(row["scores"])
                bound = next(b for b in bounds if b >= max(row["lengths"]))
                indices = []
                for f in row["forwards"]:
                    oracle.require(f["bucket"] == bound and f["rows"] == len(f["indices"])
                                   and 0 < f["rows"] <= max(1, 256000 // bound), "global maximum route changed")
                    indices.extend(f["indices"])
                oracle.require(indices == list(range(count)) and row["real_tokens"] == sum(row["lengths"])
                               and row["padded_tokens"] == count * bound, "pair/forward accounting")
            else:
                oracle.require(not row["forwards"] and row["real_tokens"] == row["padded_tokens"] == 0, "empty accounting")
    if kind == "embedding":
        queries = [r for r in records if r.get("event") == "queries"]
        oracle.require(len(queries) == 1 and len(queries[0]["values"]) == len(header["queries"]), "lost queries")
        for vector in queries[0]["values"]:
            oracle.require(len(vector) == 768, "query dimension")
            oracle.finite(vector)
    return header, calls


def difference(old, new, kind):
    a, b = old[0], new[0]
    # Only the fixed table differs. Tokens include prefix/overlap and reading order.
    oracle.require({k: v for k, v in a.items() if k != "bounds"} ==
                   {k: v for k, v in b.items() if k != "bounds"}, "different comparison input/options/model")
    differences = []
    for x, y in zip(old[1], new[1]):
        if kind == "embedding":
            for p, q in zip(x["documents"], y["documents"]):
                oracle.require(p["id"] == q["id"] and p["chunk_ids"] == q["chunk_ids"], "changed identity")
                differences.extend(oracle.vector_diff(u, v) for u, v in zip(p["values"], q["values"]))
        elif x["scores"]:
            oracle.require(x["lengths"] == y["lengths"], "pair tokens changed")
            differences.append({"pairs": [length309.pair_diff([u, s], [v, t]) for u, s, v, t in
                                          zip(x["logits"], x["scores"], y["logits"], y["scores"])],
                                "rank": oracle.ranking_diff(list(range(len(x["scores"]))), x["logits"], y["logits"]),
                                "exact_score_order": x["order"] == y["order"],
                                "exact_logit_order": sorted(range(len(x["logits"])), key=lambda i: -x["logits"][i]) == sorted(range(len(y["logits"])), key=lambda i: -y["logits"][i])})
    return differences


def timings(runs, kind):
    def wall(row):
        return ns(row["metrics"]["wall"]) if kind == "embedding" else row["wall_ns"]
    return {"cold_wall_ns": distribution([wall(run[0]) for run in runs]),
            "warm_process_median_wall_ns": distribution([statistics.median(wall(r) for r in run[1:]) for run in runs]),
            "metal_peak_bytes": distribution([max(r["metal_peak_bytes"] for r in run) for run in runs]),
            "series": runs}
