"""Model-free failures specific to the #308 three-table comparison."""
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tables308", ROOT / "docs/research/issue-308/compare.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def embedding(bounds):
    # Actual lengths 129/513/1025 expose both new routing boundaries.
    lengths = [129, 513, 1025]
    hist = [0] * len(bounds)
    shapes = []
    for length in lengths:
        bound = next(b for b in bounds if b >= length)
        hist[bounds.index(bound)] += 1
    for count, bound in zip(hist, bounds):
        if count:
            shapes.append({"batch_size": count, "sequence_length": bound})
    documents = [{"id": i, "chunk_ids": ["c0"], "values": [[1.] + [0.] * 767]} for i in range(3)]
    header = {"event": "input", "case": "mixed", "bounds": bounds, "options": {"token_budget": 8192},
              "chunk_counts": [1, 1, 1], "chunks": [[7] * n for n in lengths], "queries": ["q"], "model_files": []}
    metrics = {"num_chunks": 3, "bucket_hist": hist, "real_tokens": 1667,
               "padded_tokens": sum(s["batch_size"] * s["sequence_length"] for s in shapes),
               "forwards": shapes, "pause_count": len(shapes), "wall": {"secs": 1, "nanos": 0},
               "pause": {"secs": 0, "nanos": 1}}
    calls = [{"event": "call", "repeat": i, "documents": copy.deepcopy(documents), "metrics": copy.deepcopy(metrics),
              "metal_peak_bytes": 42} for i in range(4)]
    return [header, *calls, {"event": "queries", "values": [[1.] + [0.] * 767]}]


class BucketTables(unittest.TestCase):
    def test_prepare_checks_fresh_sources_once_before_completion_and_rechecks_later(self):
        with mock.patch.dict("sys.modules", {"compare": probe}):
            runner = probe.load("prepare308", ROOT / "docs/research/issue-308/run.py")
        # Real files/hashes and completion writes; only git/build/table patching are simulated.
        for mutation in (None, "add", "delete", "edit", "head", "variant"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                base = Path(directory).resolve()
                checkout = base / "checkout"
                checkout.mkdir()
                source = checkout / "input.rs"
                source.write_text("original")
                output = base / "prepared"
                head = ["start"]

                def git(args, **kwargs):
                    if args[1] == "rev-parse":
                        return head[0] + "\n"
                    return b"\0".join(p.name.encode() for p in checkout.iterdir()) + b"\0"

                def build(args, cwd, stdout, **kwargs):
                    binary = cwd / "libtest"
                    binary.write_text("binary")
                    stdout.write(json.dumps({"reason": "compiler-artifact", "executable": str(binary),
                                            "target": {"name": "rurico", "kind": ["lib"]},
                                            "profile": {"test": True}}) + "\n")
                    if cwd.name == list(probe.TABLES)[-1]:
                        if mutation == "add":
                            (checkout / "new.rs").write_text("new")
                        elif mutation == "delete":
                            source.unlink()
                        elif mutation == "edit":
                            source.write_text("changed")
                        elif mutation == "head":
                            head[0] = "other"
                        elif mutation == "variant":
                            (output / "current/input.rs").write_text("changed")

                with mock.patch.object(runner, "ROOT", checkout), \
                     mock.patch.object(runner.subprocess, "check_output", side_effect=git), \
                     mock.patch.object(runner.processes, "owned_run", side_effect=build), \
                     mock.patch.object(runner, "patch"), \
                     mock.patch.object(runner, "manifest", wraps=runner.manifest) as manifests:
                    if mutation:
                        message = "HEAD changed" if mutation == "head" else "variant source/binary" if mutation == "variant" else "checkout changed"
                        with self.assertRaisesRegex(ValueError, message):
                            runner.prepare(output)
                        self.assertFalse((output / "prepared.json").exists())
                        continue
                    runner.prepare(output)
                    self.assertEqual(sum(call.args[0] == checkout for call in manifests.call_args_list), 2)
                    plan = json.loads((output / "plan.json").read_text())
                    plan_path = output / "plan.json"
                    plan_bytes = plan_path.read_bytes()
                    reads = []
                    original_read = Path.read_bytes
                    original_text = Path.read_text

                    def changing_plan(path):
                        if path == plan_path:
                            reads.append(path)
                            return plan_bytes if len(reads) == 1 else b"{}"
                        return original_read(path)

                    def reopened_plan(path, *args, **kwargs):
                        if path == plan_path:
                            self.fail("plan was reopened for JSON parsing")
                        return original_text(path, *args, **kwargs)

                    with mock.patch.object(Path, "read_bytes", changing_plan), \
                         mock.patch.object(Path, "read_text", reopened_plan), \
                         mock.patch.object(runner.json, "loads", wraps=json.loads) as parses, \
                         mock.patch.object(runner.hashlib, "sha256", wraps=hashlib.sha256) as hashes:
                        observed, observed_hash = runner.verify_plan(output)
                        self.assertEqual(sum(c.args[0] == plan_bytes for c in parses.call_args_list), 1)
                        self.assertEqual(sum(c.args[0] == plan_bytes for c in hashes.call_args_list), 1)
                        self.assertEqual(observed, plan)
                        self.assertEqual(observed_hash, hashlib.sha256(plan_bytes).hexdigest())
                        self.assertEqual(len(reads), 1)
                        with self.assertRaisesRegex(ValueError, "plan changed"):
                            runner.verify_plan(output, observed)
                    source.write_text("changed after preparation")
                    with self.assertRaisesRegex(ValueError, "checkout changed"):
                        runner.verify_plan(output, plan)
                    source.write_text("original")
                    (output / "current/libtest").write_text("changed binary")
                    with self.assertRaisesRegex(ValueError, "variant source/binary"):
                        runner.verify_plan(output, plan)
                    (output / "current/libtest").write_text("binary")
                    (output / "plan.json").write_text("{}")
                    with self.assertRaisesRegex(ValueError, "plan changed"):
                        runner.verify_plan(output, plan)

    def test_empty_or_duplicate_selection_cannot_report_measurement_success(self):
        with mock.patch.dict("sys.modules", {"compare": probe}):
            runner = probe.load("runner308", ROOT / "docs/research/issue-308/run.py")
        for kinds, cases in ((["reranker"], ["multichunk"]),
                             (["embedding", "reranker"], ["multichunk"]),
                             (["embedding", "embedding"], ["mixed"]),
                             (["embedding"], ["mixed", "mixed"])):
            with self.subTest(kinds=kinds, cases=cases), self.assertRaises(ValueError):
                runner.measure(Path("missing"), Path("missing"), Path("unused"), 4, 4, 8192, kinds, cases, 3)
        self.assertEqual(runner.select_cases(["embedding", "reranker"], ["boundary128", "multichunk"]),
                         {"embedding": ["boundary128", "multichunk"], "reranker": ["boundary128"]})

    def test_new_table_routes_and_rejects_old_shape_or_missing_identity(self):
        for bounds in probe.TABLES.values():
            data = embedding(bounds)
            probe.validate(data, "embedding", "mixed", bounds, 4)
            data[0]["chunks"][0] = [7] * (bounds[-1] + 1)
            with self.assertRaises(StopIteration):
                probe.validate(data, "embedding", "mixed", bounds, 4)
        # Fixed expectations must still reject a wrong shape in every call,
        # including later warm repetitions.
        for index in (1, 4):
            data = embedding(probe.TABLES["1024-4096"])
            data[index]["metrics"]["forwards"][1]["sequence_length"] = 2048
            with self.assertRaisesRegex(ValueError, "forward/pause"):
                probe.validate(data, "embedding", "mixed", probe.TABLES["1024-4096"], 4)
        data = embedding(probe.TABLES["current"])
        data[2]["documents"][1]["id"] = 0
        with self.assertRaisesRegex(ValueError, "identity"):
            probe.validate(data, "embedding", "mixed", probe.TABLES["current"], 4)

    def test_partial_trial_and_changed_prefix_tokens_cannot_be_compared(self):
        data = embedding(probe.TABLES["current"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            probe.validate(data[:-2] + data[-1:], "embedding", "mixed", probe.TABLES["current"], 4)
        old = probe.validate(data, "embedding", "mixed", probe.TABLES["current"], 4)
        new_data = embedding(probe.TABLES["4096"])
        new_data[0]["chunks"][0][0] = 99
        new = probe.validate(new_data, "embedding", "mixed", probe.TABLES["4096"], 4)
        with self.assertRaisesRegex(ValueError, "different comparison input"):
            probe.difference(old, new, "embedding")

    def test_numerical_disagreement_stays_in_summary_and_warm_is_per_process(self):
        a = embedding(probe.TABLES["current"])
        b = embedding(probe.TABLES["4096"])
        b[2]["documents"][1]["values"][0][1] = .001
        old = probe.validate(a, "embedding", "mixed", probe.TABLES["current"], 4)
        new = probe.validate(b, "embedding", "mixed", probe.TABLES["4096"], 4)
        self.assertFalse(all(d["within_criteria"] for d in probe.difference(old, new, "embedding")))
        runs = [copy.deepcopy(old[1]) for _ in range(3)]
        for run, seconds in zip(runs, (2, 4, 8)):
            run[0]["metrics"]["wall"]["secs"] = 999
            for row in run[1:]:
                row["metrics"]["wall"]["secs"] = seconds
        summary = probe.timings(runs, "embedding")
        self.assertEqual(summary["warm_process_median_wall_ns"], {"n": 3, "min": 2*10**9, "median": 4*10**9, "max": 8*10**9})


if __name__ == "__main__":
    unittest.main()
