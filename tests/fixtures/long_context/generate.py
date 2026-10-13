"""Fixed synthetic comparison inputs for Issue #316 (no model or downloads).

Run from the repository root:
    python3 tests/fixtures/long_context/generate.py > /tmp/rurico-316-inputs.json

The same ordered documents are shared by ALL queries and ALL strategies.
Relevance is a semantic judgment about the complete document, not an expected
model score or ordering. Position ranks should also be reported individually.
Token lengths and evidence spans must be measured with the pinned tokenizer;
character lengths do not prove token overflow or retained context. This fixture
supplements, and never replaces or rewrites, #307's fixed amici evaluation.
"""

import json
import sys


def suite(name, question, evidence, wrong_answer, filler, unrelated):
    # Same filler, evidence and full length; only the evidence position differs.
    # Keep the generation deterministic and independent of tokenization policy.
    half = filler * 2048
    documents = []
    for position, before, after in (
        ("head", "", half + half),
        ("middle", half, half),
        ("tail", half + half, ""),
        ("short", "", ""),
    ):
        documents.append({
            "id": f"{name}-{position}",
            "text": before + evidence + after,
            "relevance": 1,
            "evidence_char_span": [len(before), len(before) + len(evidence)],
        })
    documents.extend([
        {"id": f"{name}-wrong-answer", "text": wrong_answer, "relevance": 0},
        {"id": f"{name}-unrelated", "text": unrelated, "relevance": 0},
    ])
    return {
        "id": name,
        "queries": [
            {"id": f"{name}-short-query", "text": question},
            # Retain the exact question at the end; preceding material adds
            # length but supplies no answer. Do not silently shorten this input.
            {"id": f"{name}-long-query", "text": filler * 4096 + question},
        ],
        "documents": documents,
    }


def fixtures():
    topics = [
        suite(
            "japanese",
            "あおば図書館の月曜日の開館時刻は何時ですか？",
            "あおば図書館は月曜日の午前9時に開館します。\n",
            "あおば図書館は月曜日の午後3時に開館します。\n",
            "猫 ",
            "海岸で夕日を撮影しました。\n",
        ),
        suite(
            "code",
            "Find the Rust retry_limit function that returns 7.",
            "fn retry_limit() -> u32 { 7 }\n",
            "fn retry_limit() -> u32 { 9 }\n",
            "// note\n",
            "fn square(x: i32) -> i32 { x * x }\n",
        ),
    ]
    documents = [
        {key: value for key, value in document.items() if key != "relevance"}
        for topic in topics for document in topic["documents"]
    ]
    queries = []
    for topic in topics:
        relevant = {d["id"] for d in topic["documents"] if d["relevance"] == 1}
        for query in topic["queries"]:
            queries.append({
                **query,
                "relevance": {d["id"]: int(d["id"] in relevant) for d in documents},
            })
    return {
        "issue": "https://github.com/thkt/rurico/issues/316",
        "scope": "synthetic long-context comparison; no acceptance thresholds",
        "span_unit": "Unicode code points; end exclusive; not token indices",
        "documents": documents,
        "queries": queries,
    }


if __name__ == "__main__":
    json.dump(fixtures(), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
