"""#367 model lane: reuse #307's official oracle and record validation."""
import json
from pathlib import Path

from compare import compare, ordered, require, vector_diff

MODELS = {
    "30m": ("24899e5de370b56d179604a007c0d727bf144504", 256),
    "70m": ("07a8b0aba47d29d2ca21f89b915c1efe2c23d1cc", 384),
    "130m": ("e3114c6ee10dbab8b4b235fbc6dcf9dd4d5ac1a6", 512),
    "310m": ("18b60fb8c2b9df296fb4212bb7d23ef94e579cd3", 768),
}
START = "fa31e0256b6d437d2fb1009fdb2b1eea0f97da45"


def cases(model):
    original = json.loads(Path(__file__).with_name("inputs.json").read_text())["embedding"]
    # The existing 310m conditions run once in this lane, with empty cases added.
    # Other variants use the existing representative cases, not a new oracle.
    selected = original if model == "310m" else [c for c in original if c["id"] in
        ("short", "japanese", "code", "local-after", "limit-after", "query-prefix", "topic-prefix", "no-prefix")]
    return {"reranker": [], "embedding": selected + [
        {"id": "empty-document", "text": "", "prefix": "検索文書: ", "repeat": 1},
        {"id": "empty-query", "text": "", "prefix": "検索クエリ: ", "repeat": 1},
        {"id": "empty-text", "text": "", "prefix": "", "repeat": 1},
    ]}


def evaluate(model, observed, reference, inputs):
    revision, dimensions = MODELS[model]
    public = observed["public"]
    require(public["model"] == {
        "repository": f"cl-nagoya/ruri-v3-{model}", "revision": revision,
        "concrete_dimensions": dimensions, "expected_dimensions": dimensions,
    }, "model identity/dimension mismatch")
    result = compare(observed, reference, inputs, dimensions=dimensions)
    for token in observed["tokenization"]:
        if token["id"] == "local-after":
            require(128 < len(token["model_ids"]) <= 512, "local padding boundary not exercised")
        if token["id"] == "limit-after":
            require(len(token["raw_ids"]) > 8192 and len(token["model_ids"]) == 8192
                    and len(token["chunks"]) > 1, "multi-chunk boundary not exercised")
    docs = [c["id"] for c in inputs["embedding"] if c["prefix"] == "検索文書: "]
    require(public["document_token_budget"] == 8192, "variant document token budget mismatch")
    ordered(public["singleton_documents"], docs)
    consistency = []
    short_ids = [t["id"] for t in observed["tokenization"]
                 if t["id"] in docs and all(len(chunk) <= 512 for chunk in t["chunks"])]
    require(len(short_ids) >= 2, "default mixed document batch not exercised")
    ordered(public["default_short_documents"], short_ids)
    for default in public["default_short_documents"]:
        single = next(s for s in public["singleton_documents"] if s["id"] == default["id"])
        require(len(default["chunks"]) == len(single["chunks"]), "default batch chunk count")
        for d, s in zip(default["chunks"], single["chunks"]):
            require(len(d) == len(s) == dimensions, "default batch dimension")
            consistency.append({"id": default["id"], "mode": "default_short_batch", **vector_diff(s, d)})
    for batch, single in zip(public["documents"], public["singleton_documents"]):
        require(len(batch["chunks"]) == len(single["chunks"]) > 0, "batch/single chunk count")
        for index, (b, s) in enumerate(zip(batch["chunks"], single["chunks"])):
            require(len(b) == len(s) == dimensions, "batch/single dimension")
            consistency.append({"id": batch["id"], "chunk": index, **vector_diff(s, b)})
    query_ids = [c["id"] for c in inputs["embedding"] if c["prefix"] == "検索クエリ: "]
    ordered(public["queries"], query_ids)
    for query in public["queries"]:
        text = next(c for c in public["text"] if c["id"] == query["id"])
        require(len(query["value"]) == dimensions, "query dimension")
        consistency.append({"id": query["id"], **vector_diff(text["value"], query["value"])})
    require(public["empty_batch"] == [], "empty batch output")
    # A wrapper-token mismatch remains visible, and never masquerades as a
    # same-token numerical failure or a new product preprocessing rule.
    public_direct = [v for row in result["wrapper_and_chunks"] for key, v in row.items()
                     if key in ("public_vs_direct", "chunk_public_vs_direct")]
    passed = (result["same_token_numerics_within_criteria"]
        and all(t["raw_equal"] for t in result["tokenization"])
        and all(p[k]["within_criteria"] for p in result["padding"] for k in ("rurico", "reference"))
        and all(v["within_criteria"] for v in public_direct + consistency))
    return {**result, "model": public["model"], "batch_single_and_query": consistency,
            "contract_within_predeclared_criteria": passed}


def observe(out, model):
    """Reuse #366 group ownership so cancellation reclaims reference workers."""
    import os
    from run import HERE
    from processes import owned_run

    code = ("import sys; from pathlib import Path; from run import numerical; "
            "numerical(Path(sys.argv[1]), variant=sys.argv[2])")
    owned_run([os.sys.executable, "-c", code, str(out), model], cwd=HERE, check=True)
