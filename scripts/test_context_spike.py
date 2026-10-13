"""Independent #316 record/aggregation checks; never load a model or run main."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "context316", ROOT / "docs/research/issue-316/run.py"
)
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)


def records():
    inputs = {
        "documents": [{"id": key} for key in ("answer", "first-tie", "second-tie", "wrong")],
        "queries": [{"id": "q", "relevance": {
            "answer": 1, "first-tie": 0, "second-tie": 0, "wrong": 0,
        }}],
    }
    rows = [{"event": "context", "mode": "model", "strategy": "windows"}]
    # Max and arithmetic mean must distinguish strong local evidence from
    # average relevance. Equal means/maxima expose the original-index tie rule.
    for trial, wall in enumerate((100, 4, 8, 6)):
        documents = []
        for doc, scores in zip(inputs["documents"], ((.9, .1), (.6, .6), (.6, .6), (.2, .2))):
            windows = [{
                "ids": [1, 5, 3, 4, 2], "mask": [1] * 5,
                "sequence_ids": [None, 0, None, 1, None],
                "offsets_bytes": [[0, 0]] * 5,
                "special_tokens_mask": [1, 0, 1, 0, 1],
                "score": score, "evidence_retained": doc["id"] == "answer",
            } for score in scores]
            documents.append({
                "document": doc["id"], "raw_ids": [1, 5, 3, 4, 2],
                "raw_special_tokens_mask": [1, 0, 1, 0, 1],
                "raw_query_tokens": 1, "raw_document_tokens": 1, "windows": windows,
            })
        rows.append({
            "event": "call", "query": "q", "trial": trial, "cold": trial == 0,
            "wall_ns": wall, "metal_peak_bytes": (9999, 11, 22, 33)[trial],
            "documents": documents, "forwards": [{"rows": 1, "bucket": 128}] * 8,
        })
    return rows, inputs


class ContextComparisonTests(unittest.TestCase):
    def padding_records(self, strategy="windows"):
        rows, inputs = records()
        rows[0]["strategy"] = strategy
        # ADR-0009 at the Issue's start commit requires the smallest fixed
        # bucket at least as large as each singleton's actual token length.
        # Boundary expectations are literal, independent of assign_bucket.
        shapes = ((128, 128), (129, 512), (512, 512), (513, 2048),
                  (2048, 2048), (2049, 8192), (8192, 8192), (5, 128))
        for call in rows[1:]:
            windows = [w for doc in call["documents"] for w in doc["windows"]]
            for window, (length, _) in zip(windows, shapes):
                window.update(
                    ids=[1, 5, 3] + [4] * (length - 4) + [2],
                    mask=[1] * length,
                    sequence_ids=[None, 0, None] + [1] * (length - 4) + [None],
                    offsets_bytes=[[0, 0]] * length,
                    special_tokens_mask=[1, 0, 1] + [0] * (length - 4) + [1],
                )
            call["forwards"] = [{"rows": 1, "bucket": bucket} for _, bucket in shapes]
            if strategy != "windows":
                # Single-score methods have one window per candidate. Preserve
                # the same lengths without inventing multi-window single calls.
                call["documents"] = [
                    {**doc, "document": f"candidate-{index}", "windows": [window]}
                    for index, (doc, window) in enumerate(
                        (doc, window) for doc in call["documents"] for window in doc["windows"]
                    )
                ]
        if strategy != "windows":
            inputs["documents"] = [{"id": f"candidate-{index}"} for index in range(8)]
            inputs["queries"][0]["relevance"] = {
                doc["id"]: int(index == 0) for index, doc in enumerate(inputs["documents"])
            }
        return rows, inputs, shapes

    def test_fixed_padding_roundup_boundaries_are_accepted(self):
        for strategy in ("baseline", "query-prefix", "chunk-neighbors", "windows"):
            with self.subTest(strategy=strategy):
                rows, inputs, _ = self.padding_records(strategy)
                result = context.summarize(rows, inputs, strategy, "model")[0]
                self.assertEqual(len(result["observations"]), 4)
                self.assertTrue(all(o["forward_count"] == 8 for o in result["observations"]))

    def test_allowed_bucket_must_match_each_window_length(self):
        for strategy in ("baseline", "query-prefix", "chunk-neighbors", "windows"):
            rows, inputs, shapes = self.padding_records(strategy)
            self.assertEqual(len(context.summarize(rows, inputs, strategy, "model")), 1)
            self.assert_padding_mismatches_rejected(rows, inputs, shapes, strategy)

    def assert_padding_mismatches_rejected(self, rows, inputs, shapes, strategy):
        for index, (length, bucket) in enumerate(shapes):
            # Larger legal buckets mix in extra padding; smaller legal buckets
            # cannot hold the tokens. Membership alone detects neither error.
            wrong_buckets = [b for b in (128, 512, 2048, 8192) if b != bucket]
            for wrong in wrong_buckets:
                invalid = copy.deepcopy(rows)
                invalid[1]["forwards"][index]["bucket"] = wrong
                with self.subTest(strategy=strategy, length=length, expected=bucket, actual=wrong):
                    with self.assertRaises(ValueError):
                        context.summarize(invalid, inputs, strategy, "model")

    def test_window_aggregations_ties_quality_and_warm_cost_are_distinct(self):
        rows, inputs = records()
        original = copy.deepcopy(rows)
        result = context.summarize(rows, inputs, "windows", "model")[0]
        self.assertEqual(rows, original)  # Aggregation consumes the same forwards.
        for observation in result["observations"]:
            self.assertEqual(observation["forward_count"], 8)
            maximum = observation["aggregations"]["max"]
            mean = observation["aggregations"]["mean"]
            self.assertEqual(maximum["order"], ["answer", "first-tie", "second-tie", "wrong"])
            self.assertEqual(mean["order"], ["first-tie", "second-tie", "answer", "wrong"])
            self.assertAlmostEqual(maximum["scores"]["answer"], .9)
            self.assertAlmostEqual(mean["scores"]["answer"], .5)
            self.assertEqual(maximum["quality"]["Recall@1"], 1)
            self.assertEqual(mean["quality"]["Recall@1"], 0)
            self.assertEqual(maximum["quality"]["nDCG@5"], 1)
            self.assertAlmostEqual(mean["quality"]["nDCG@5"], 1 / math.log2(4))
            self.assertEqual(mean["quality"]["Recall@5"], 1)
        self.assertEqual(result["warm"]["wall_ns"], {"min": 4, "median": 6, "max": 8})
        self.assertEqual(result["warm"]["metal_peak_bytes"], {"min": 11, "median": 22, "max": 33})

    def test_missing_calls_candidates_boundaries_and_invalid_forward_records_fail(self):
        rows, inputs = records()
        mutations = [
            lambda r: r.pop(),
            lambda r: r[1]["documents"].reverse(),
            lambda r: r[1]["documents"].pop(),
            lambda r: r[1]["forwards"].pop(),
            lambda r: r[1]["forwards"][0].update(bucket=256),
            lambda r: r[1]["documents"][0]["windows"][0].update(score=float("nan")),
            lambda r: r[1]["documents"][0]["windows"][0].update(score=float("inf")),
            lambda r: r[1]["documents"][0]["windows"][0].update(mask=[1]),
            lambda r: r[1]["documents"][0]["windows"][0].update(sequence_ids=[None] * 5),
            lambda r: r[1]["documents"][0]["windows"][0].update(ids=[1, 5, 3, 4, 7]),
            lambda r: r[1]["documents"][0]["windows"][0].update(special_tokens_mask=[1, 0, 0, 0, 1]),
        ]
        # The valid counterpart reaches the same summarizer before each fault.
        self.assertEqual(len(context.summarize(rows, inputs, "windows", "model")), 1)
        for index, mutate in enumerate(mutations):
            invalid = copy.deepcopy(rows)
            mutate(invalid)
            with self.subTest(fault=index), self.assertRaises(ValueError):
                context.summarize(invalid, inputs, "windows", "model")

    def test_plan_records_do_not_claim_model_quality_or_cost(self):
        rows, inputs = records()
        rows = rows[:2]
        rows[0]["mode"] = "plan"
        rows[1]["forwards"] = []
        rows[1]["metal_peak_bytes"] = None
        result = context.summarize(rows, inputs, "windows", "plan")[0]
        self.assertFalse(result["model_measured"])
        self.assertNotIn("observations", result)
        self.assertNotIn("warm", result)
        for field, value in (("forwards", [{"rows": 1, "bucket": 128}]), ("metal_peak_bytes", 0)):
            invalid = copy.deepcopy(rows)
            invalid[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                context.summarize(invalid, inputs, "windows", "plan")


if __name__ == "__main__":
    unittest.main()
