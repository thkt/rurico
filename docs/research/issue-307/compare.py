"""Issue #307 diagnostics. Criteria are fixed before host observations.

Malformed/incomplete records raise ValueError. Numerical disagreements remain
measurements, never become a new tolerance or a replacement baseline.
"""
import argparse
import itertools
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(values):
    require(bool(values) and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                and math.isfinite(x) for x in values), "non-finite/empty output")


def vector_diff(reference, actual):
    require(len(reference) == len(actual), "vector dimension mismatch")
    finite(reference)
    finite(actual)
    nr = math.sqrt(sum(x*x for x in reference))
    na = math.sqrt(sum(x*x for x in actual))
    require(nr > 0 and na > 0, "zero vector")
    cosine = sum(x*y for x, y in zip(reference, actual)) / nr / na
    maximum = max(abs(x-y) for x, y in zip(reference, actual))
    return {"cosine": cosine, "max_abs": maximum, "reference_norm": nr, "actual_norm": na,
            "within_criteria": cosine >= .99999 and maximum <= 1e-5}


def reranker_diff(reference, actual):
    require(len(reference) == len(actual) == 2, "expected logit and score")
    finite(reference)
    finite(actual)
    require(0 <= reference[1] <= 1 and 0 <= actual[1] <= 1, "score outside [0,1]")
    logit, score = (abs(x-y) for x, y in zip(reference, actual))
    return {"logit_abs": logit, "score_abs": score,
            "within_criteria": logit <= 1e-4 and score <= 2.5e-5}


def ranking_diff(keys, reference, actual, actual_order=None):
    require(len(keys) == len(reference) == len(actual) and len(set(keys)) == len(keys), "rank count/order")
    finite(reference)
    finite(actual)
    ro = sorted(range(len(keys)), key=lambda i: -reference[i])
    ao = sorted(range(len(keys)), key=lambda i: -actual[i])
    if actual_order is not None:
        require(sorted(actual_order) == sorted(keys), "returned rank identities")
        ao = [keys.index(k) for k in actual_order]
    near, discordant = [], []
    for i, j in itertools.combinations(range(len(keys)), 2):
        gap = abs(reference[i]-reference[j])
        # Actual ranks, including stable input-order ties, must remain visible.
        reversed_order = (ro.index(i)-ro.index(j)) * (ao.index(i)-ao.index(j)) < 0
        pair = {"keys":[keys[i],keys[j]], "reference_gap":gap, "reversed":reversed_order}
        if gap <= 2e-4:
            near.append(pair)
        elif reversed_order:
            discordant.append(pair)
    return {"reference_order":[keys[i] for i in ro], "actual_order":[keys[i] for i in ao],
            "near_pairs":near, "discordant_pairs":discordant, "within_criteria":not discordant}


def validate_batch(batch, kind):
    # Output values are checked numerically by the comparison's diff function.
    keys, ids, mask, values = (batch[k] for k in ("keys", "ids", "mask", "values"))
    require(len(keys) > 0 and len(set(keys)) == len(keys), "empty/duplicate row keys")
    require(len(keys) == len(ids) == len(mask) == len(values), "batch row count")
    length = len(ids[0])
    require(0 < length <= 8192, "invalid sequence length")
    for row, m, v in zip(ids, mask, values):
        require(len(row) == len(m) == length, "ragged batch")
        require(all(type(x) is int and 0 <= x < 102400 for x in row), "invalid token ID")
        require(all(x in (0, 1) for x in m) and any(m), "invalid/empty mask")
        require(len(v) == (768 if kind == "embedding" else 2), "output dimension")
    probes = batch["hidden_probes"]
    require(len(probes) == (len(keys) if kind == "embedding" else 0), "hidden probe row count")
    for row in probes:
        require(len(row) == 3, "hidden probe count")
        for v in row:
            require(len(v) == 768, "hidden probe dimension")
            finite(v)


def validate_tokens(record):
    for prefix in ("raw", "model"):
        ids, mask = record[prefix+"_ids"], record[prefix+"_mask"]
        require(len(ids) == len(mask) > 0, "token/mask length")
        require(all(type(x) is int and 0 <= x < 102400 for x in ids), "token ID")
        require(all(type(x) is int and x in (0,1) for x in mask) and any(mask), "token mask")
    require(len(record["model_ids"]) <= 8192, "truncation limit")


def ordered(records, expected, key="id"):
    require([r[key] for r in records] == expected, "missing/duplicate/reordered records")


def compare(rurico, reference, inputs):
    kind = rurico["kind"]
    require(kind in ("embedding", "reranker") and reference["kind"] == kind, "model kind")
    require(rurico["producer"] == "rurico" and reference["producer"] == "transformers", "producer")
    schema = rurico["schema"]
    require(schema in (1,2) and reference["schema"] == schema, "schema")
    cases = inputs[kind]
    case_ids = [x["id"] for x in cases]
    ordered(rurico["tokenization"], case_ids)
    ordered(reference["tokenization"], case_ids)
    token_diffs = []
    expected_keys = []
    token_rows = {}
    for r, t in zip(rurico["tokenization"], reference["tokenization"]):
        validate_tokens(r)
        validate_tokens(t)
        token_diffs.append({"id":r["id"], "raw_equal":r["raw_ids"] == t["raw_ids"] and r["raw_mask"] == t["raw_mask"],
                            "wrapper_equal":r["model_ids"] == t["model_ids"] and r["model_mask"] == t["model_mask"],
                            "raw_lengths":[len(r["raw_ids"]),len(t["raw_ids"])],
                            "model_lengths":[len(r["model_ids"]),len(t["model_ids"])]})
        key = r["id"] + "/text"
        expected_keys.append(key)
        token_rows[key] = (r["model_ids"], r["model_mask"])
        if kind == "embedding" and len(r["chunks"]) > 1:
            for i, row in enumerate(r["chunks"]):
                key = f'{r["id"]}/chunk-{i}'
                expected_keys.append(key)
                token_rows[key] = (row, [1]*len(row))
    expected_batches, conditions, observations = [], {}, {}
    for key in expected_keys:
        length = len(token_rows[key][0])
        shared = schema == 2 and length in (128,512,2048,8192)
        for mode in ("exact", "bucket"):
            observation = f"{key}/{'exact' if shared else mode}"
            observations[f"{key}/{mode}"] = observation
            if observation not in conditions:
                expected_batches.append(observation)
                conditions[observation] = []
            conditions[observation].append(mode)
    expected_batches.append("mixed/bucket")
    conditions["mixed/bucket"] = ["bucket"]
    ordered(rurico["batches"], expected_batches)
    ordered(reference["batches"], expected_batches)
    diff = vector_diff if kind == "embedding" else reranker_diff
    numerical, padding, ranking = [], [], []
    rb, tb = {}, {}
    for r, t in zip(rurico["batches"], reference["batches"]):
        validate_batch(r, kind)
        validate_batch(t, kind)
        if schema == 2:
            require(r.get("conditions") == t.get("conditions") == conditions[r["id"]], "batch conditions")
        require(all(r[k] == t[k] for k in ("id", "keys", "ids", "mask")), "same-token comparison input mismatch")
        if r["id"] == "mixed/bucket":
            keys = [k for k in expected_keys if len(token_rows[k][0]) <= 512][:5]
        else:
            keys = [r["id"].rsplit("/", 1)[0]]
        require(r["keys"] == keys, "batch input order mismatch")
        raw_length = max(len(token_rows[k][0]) for k in keys)
        length = raw_length if r["id"].endswith("/exact") else next(n for n in (128,512,2048,8192) if n >= raw_length)
        for key, ids, mask in zip(keys, r["ids"], r["mask"]):
            base_ids, base_mask = token_rows[key]
            require(ids == base_ids + [0]*(length-len(base_ids)) and mask == base_mask + [0]*(length-len(base_mask)), "padding/token provenance")
        rows = []
        for i, (rv, tv) in enumerate(zip(r["values"], t["values"])):
            result = {"key":r["keys"][i], **diff(tv, rv)}
            if kind == "embedding":
                result["hidden_probe_max_abs"] = max(abs(x-y) for rp,tp in zip(r["hidden_probes"][i], t["hidden_probes"][i]) for x,y in zip(rp,tp))
            rows.append(result)
        numerical.append({"batch":r["id"], "shape":[len(keys),length], "rows":rows})
        rb[r["id"]], tb[t["id"]] = r, t
    # Resolve logical conditions only after validating every actual observation.
    # Both implementations still supply their own independently computed values.
    for condition, observation in observations.items():
        rb[condition], tb[condition] = rb[observation], tb[observation]
    for key in expected_keys:
        result = {"key":key, "rurico":diff(rb[key+"/exact"]["values"][0],rb[key+"/bucket"]["values"][0]),
                  "reference":diff(tb[key+"/exact"]["values"][0],tb[key+"/bucket"]["values"][0])}
        if schema == 2:
            result["observations"] = {mode:observations[f"{key}/{mode}"] for mode in ("exact", "bucket")}
            result["same_observation"] = result["observations"]["exact"] == result["observations"]["bucket"]
        padding.append(result)
    if kind == "reranker":
        for mode in ("exact", "bucket", "mixed"):
            for query in dict.fromkeys(x["query"] for x in cases):
                members = [x["id"]+"/text" for x in cases if x["query"] == query]
                if mode == "mixed":
                    keys = rb["mixed/bucket"]["keys"]
                    members = [k for k in members if k in keys]
                    rv = [rb["mixed/bucket"]["values"][keys.index(k)][0] for k in members]
                    tv = [tb["mixed/bucket"]["values"][keys.index(k)][0] for k in members]
                else:
                    rv = [rb[f"{k}/{mode}"]["values"][0][0] for k in members]
                    tv = [tb[f"{k}/{mode}"]["values"][0][0] for k in members]
                if len(members) > 1:
                    ranking.append({"mode":mode, "query":query, **ranking_diff(members,tv,rv)})
    # Public API and native official wrapper are separate from controlled tokens.
    native = reference["native"]
    ordered(native, case_ids)
    public = rurico["public"]
    wrapper = []
    if kind == "embedding":
        ordered(public["text"], case_ids)
        docs = [x["id"] for x in cases if x["prefix"] == "検索文書: "]
        ordered(public["documents"], docs)
        for p, n, token in zip(public["text"], native, token_diffs):
            require(len(p["value"]) == len(n["value"]) == 768, "wrapper dimensions")
            wrapper.append({"id":p["id"], "tokens_equal":token["wrapper_equal"],
                            "native_wrapper":vector_diff(n["value"],p["value"]),
                            "public_vs_direct":vector_diff(rb[p["id"]+"/text/bucket"]["values"][0],p["value"])})
        for doc in public["documents"]:
            token = next(r for r in rurico["tokenization"] if r["id"] == doc["id"])
            require(len(doc["chunks"]) == len(token["chunks"]) > 0, "public document chunk count")
            for i, value in enumerate(doc["chunks"]):
                key = doc["id"] + (f"/chunk-{i}" if len(doc["chunks"]) > 1 else "/text")
                wrapper.append({"id":key, "chunk_public_vs_direct":vector_diff(rb[key+"/bucket"]["values"][0],value)})
    else:
        ordered(public["singleton"], case_ids)
        for p, n, token in zip(public["singleton"], native, token_diffs):
            native_vs_controlled = reranker_diff(n["value"],rb[p["id"]+"/text/exact"]["values"][0])
            finite([p["score"]])
            require(0 <= p["score"] <= 1, "public score outside [0,1]")
            direct = rb[p["id"]+"/text/bucket"]["values"][0][1]
            wrapper.append({"id":p["id"], "tokens_equal":token["wrapper_equal"],
                            "native_score_abs":abs(p["score"]-n["value"][1]),
                            "native_vs_controlled":native_vs_controlled,
                            "public_vs_direct_score_abs":abs(p["score"]-direct)})
        scores = public["mixed_scores"]
        require(len(scores) == len(rb["mixed/bucket"]["keys"]), "public mixed count")
        finite(scores)
        require(all(0 <= s <= 1 for s in scores), "public mixed score outside [0,1]")
        wrapper.append({"id":"mixed", "public_vs_direct_score_abs":[abs(a-b[1]) for a,b in zip(scores,rb["mixed/bucket"]["values"])]})
        ranks = public["ranking"]
        require(sorted(r["index"] for r in ranks) == list(range(4)), "public rank indices")
        finite([r["score"] for r in ranks])
        require(all(0 <= r["score"] <= 1 for r in ranks), "public rank score outside [0,1]")
        actual_scores = [next(r["score"] for r in ranks if r["index"] == i) for i in range(4)]
        actual_order = [case_ids[r["index"]] for r in ranks]
        ranking.append({"mode":"public_vs_native", **ranking_diff(case_ids[:4], [n["value"][0] for n in native[:4]], actual_scores, actual_order),
                        "public_rerank_order":actual_order,
                        "score_abs":[abs(s-n["value"][1]) for s,n in zip(actual_scores,native[:4])]})
    return {"kind":kind, "tokenization":token_diffs,"same_tokens":numerical,
            "padding":padding,"wrapper_and_chunks":wrapper,"ranking":ranking,
            "same_token_numerics_within_criteria":all(row["within_criteria"] for b in numerical for row in b["rows"]),
            "interpretation":"Threshold crossings are observations. Hidden probes sample three positions, not all tensors. No product tolerance adopted."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rurico", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inputs = json.loads(Path(__file__).with_name("inputs.json").read_text())
    result = compare(json.loads(args.rurico.read_text()),json.loads(args.reference.read_text()), inputs)
    with args.output.open("x") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
