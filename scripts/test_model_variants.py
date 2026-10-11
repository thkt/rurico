"""Counterexamples for #367's evidence boundary, without inference or downloads."""
import copy
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"docs/research/issue-307"))

from variants import MODELS, cases, evaluate


def records(model):
    revision, dimensions = MODELS[model]
    inputs = {"embedding": [
        {"id": "doc", "prefix": "検索文書: ", "text": "猫", "repeat": 1},
        {"id": "query", "prefix": "検索クエリ: ", "text": "猫", "repeat": 1},
        {"id": "empty", "prefix": "検索文書: ", "text": "", "repeat": 1},
    ]}
    tokens, batches, native = [], [], []
    for index, case in enumerate(inputs["embedding"]):
        ids = [1, 5 + index, 2] if index < 2 else [1, 2]
        mask = [1]*len(ids)
        value = [0.]*dimensions
        value[index] = 1.
        token = dict(id=case["id"], raw_ids=ids, raw_mask=mask, model_ids=ids,
                     model_mask=mask, chunks=[ids] if case["prefix"] == "検索文書: " else [])
        tokens.append(token)
        native.append(dict(id=case["id"], value=value))
        for mode, length in (("exact",len(ids)), ("bucket",128)):
            batches.append(dict(id=case["id"]+"/text/"+mode, conditions=[mode],
                keys=[case["id"]+"/text"], ids=[ids+[0]*(length-len(ids))],
                mask=[mask+[0]*(length-len(mask))], values=[value], hidden_probes=[[value]*3]))
    bucket = batches[1::2]
    batches.append(dict(id="mixed/bucket", conditions=["bucket"],
        **{k: [b[k][0] for b in bucket] for k in ("keys","ids","mask","values","hidden_probes")}))
    docs = [dict(id=n["id"], chunks=[n["value"]]) for n in native if n["id"] != "query"]
    public = dict(text=native, documents=docs, singleton_documents=copy.deepcopy(docs),
        default_short_documents=copy.deepcopy(docs), document_token_budget=8192,
        queries=copy.deepcopy([native[1]]), empty_batch=[], model=dict(repository=f"cl-nagoya/ruri-v3-{model}",
            revision=revision, concrete_dimensions=dimensions, expected_dimensions=dimensions))
    actual = dict(schema=2, producer="rurico", kind="embedding", tokenization=tokens,
                  batches=batches, public=public)
    reference = dict(schema=2, producer="transformers", kind="embedding",
                     tokenization=copy.deepcopy(tokens), batches=copy.deepcopy(batches), native=copy.deepcopy(native))
    return actual, reference, inputs


class VariantEvidence(unittest.TestCase):
    def test_dimensions_and_mislabelled_model_are_checked_on_real_record_boundary(self):
        for model in MODELS:
            actual, reference, inputs = records(model)
            self.assertTrue(evaluate(model, actual, reference, inputs)["contract_within_predeclared_criteria"])
            for field, value in (("repository", "another/model"), ("revision", "wrong"),
                                 ("concrete_dimensions", 1), ("expected_dimensions", 1)):
                bad = copy.deepcopy(actual)
                bad["public"]["model"][field] = value
                with self.assertRaisesRegex(ValueError, "identity/dimension"):
                    evaluate(model, bad, reference, inputs)
            # Keep metadata correct: mislabeled vector shape must independently fail.
            bad = copy.deepcopy(actual)
            bad["batches"][0]["values"][0].pop()
            with self.assertRaisesRegex(ValueError, "output dimension"):
                evaluate(model, bad, reference, inputs)

    def test_public_missing_reordered_nonfinite_and_numeric_drift_do_not_pass(self):
        actual, reference, inputs = records("30m")
        for fault in ("missing", "order", "chunks", "nonfinite", "drift", "query", "empty", "default_missing", "budget"):
            bad = copy.deepcopy(actual)
            public = bad["public"]
            if fault == "missing":
                public["singleton_documents"].pop()
            elif fault == "order":
                public["singleton_documents"].reverse()
            elif fault == "chunks":
                public["singleton_documents"][0]["chunks"].append([1.]*256)
            elif fault == "nonfinite":
                public["singleton_documents"][0]["chunks"][0][0] = float("nan")
            elif fault == "drift":
                public["singleton_documents"][0]["chunks"][0][1] = .01
            elif fault == "query":
                public["queries"][0]["value"] = [0., 0., 1.] + [0.]*253
            elif fault == "default_missing":
                public["default_short_documents"].pop()
            elif fault == "budget":
                public["document_token_budget"] = None
            else:
                public["empty_batch"] = [[1.]*256]
            with self.subTest(fault=fault):
                if fault in ("drift", "query"):
                    self.assertFalse(evaluate("30m", bad, reference, inputs)["contract_within_predeclared_criteria"])
                else:
                    with self.assertRaises(ValueError):
                        evaluate("30m", bad, reference, inputs)

    def test_existing_310m_cases_are_selected_once_and_other_models_have_boundaries(self):
        for model in MODELS:
            self.assertEqual(cases(model)["reranker"], [])
            selected = cases(model)["embedding"]
            ids = [c["id"] for c in selected]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(ids), 13 if model == "310m" else 11)
            self.assertTrue({"local-after", "limit-after", "empty-document", "empty-query", "empty-text"} <= set(ids))
