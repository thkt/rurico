"""Prevent successful-looking Issue #317 records from hiding missing evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "docs/research/issue-317/run.py"
spec = importlib.util.spec_from_file_location("query317_runner", PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class QueryBatchEvidenceTests(unittest.TestCase):
    def test_collection_rejects_missing_wrong_order_and_nonfinite_outputs(self):
        rows = [{"event": "context", "domain": "pair", "case": "single", "mode": "candidate"},
                {"event": "load"}]
        for trial in range(4):
            rows.append({"event": "call", "trial": trial, "cold": trial == 0,
                         "scores": [.2], "metal_peak_bytes": 1,
                         "metrics": {"forwards": [{"batch_size": 1, "sequence_length": 128}]}})
        rows.append({"event": "comparison"})
        self.assertEqual(len(runner.validate(rows, "pair", "single", "candidate")), 4)
        missing = copy.deepcopy(rows)
        missing.pop(3)
        bad_count = copy.deepcopy(rows)
        bad_count[2]["scores"].append(.9)
        nonfinite = copy.deepcopy(rows)
        nonfinite[2]["scores"][0] = float("nan")
        reordered = copy.deepcopy(rows)
        reordered[2], reordered[3] = reordered[3], reordered[2]
        for bad in (missing, bad_count, nonfinite, reordered):
            with self.assertRaises(ValueError):
                runner.validate(bad, "pair", "single", "candidate")

    def test_query_comparison_detects_row_permutation_and_dimension_loss(self):
        a = [1., 0.] + [0.] * 766
        b = [0., 1.] + [0.] * 766
        self.assertEqual(runner.compare_queries([a, b], [a, b])["max_abs"], 0)
        for bad in ([b, a], [a], [a[:-1], b]):
            with self.assertRaises(ValueError):
                runner.compare_queries([a, b], bad)


if __name__ == "__main__":
    unittest.main()
