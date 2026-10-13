"""Reject incomplete, mixed or altered partitioned comparison evidence."""
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from test_bucket_tables import embedding

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("partition310", ROOT / "docs/research/issue-310/partition.py")
partition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(partition)
api = partition.load(ROOT)


class PartitionEvidence(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.phases = []
        data = embedding(api.compare.TABLES["current"])
        calls = []
        for i in range(9):
            row = copy.deepcopy(data[1])
            row["repeat"] = i
            row["metrics"]["readback_elements"] = [f["batch_size"] * 768 for f in row["metrics"]["forwards"]]
            calls.append(row)
        data = [data[0], *calls, data[-1]]
        for trial in range(3):
            for variant in partition.VARIANTS:
                name = f"embedding-mixed-{trial}-{variant}"
                directory = self.root / name
                directory.mkdir()
                context = {"kind":"embedding", "case":"mixed", "repeats":9, "budget":8192,
                           "trial":trial, "variant":variant, "name":name,
                           "plan_sha256":"fixed-unit-plan", "collector_sha256":"fixed-unit-collector"}
                api.runner.save(directory / "context.json", context)
                api.runner.save(directory / f"{name}.json", data)
                api.runner.save(directory / f"{name}.status.json", {"complete":True, "exit_code":0, "group_absent":True})
                api.runner.save(directory / f"{name}.monitor.json", [{"rss_bytes":1, "pressure":1, "swap_bytes":0, "elapsed_seconds":1}])
                (directory / f"{name}.stdout").write_text("unit fixture, not model evidence\n")
                (directory / f"{name}.stderr").write_text('mask310 {"event":"mask_graph"}\nmask310 {"event":"cleanup"}\n')
                api.runner.save(directory / "phase-complete.json", {"phase_complete":True,
                    "files":{p.name:partition.digest(p) for p in directory.iterdir()}})
                self.phases.append(directory)

    def update_hashed(self, directory, path, data):
        api.runner.save(path, data)
        complete = json.loads((directory / "phase-complete.json").read_text())
        complete["files"][path.name] = partition.digest(path)
        api.runner.save(directory / "phase-complete.json", complete)

    def test_complete_matrix_and_required_partitions(self):
        output = self.root / "combined"
        partition.combine(api, self.phases, output)
        summary = json.loads((output / "summary.json").read_text())["embedding/mixed"]
        self.assertEqual(set(summary["measurements"]), set(partition.VARIANTS))
        self.assertTrue(all(v["cold_wall_ns"]["n"] == 3 for v in summary["measurements"].values()))
        for phases, reason in ((self.phases[:-1], "all five candidates"),
                               ([*self.phases, self.phases[0]], "duplicate phase")):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                partition.combine(api, phases, self.root / "invalid")
        self.assertFalse((self.root / "invalid").exists())

    def test_measure_and_collect_use_returned_events_without_reopening_stderr(self):
        data = json.loads((self.phases[0] / f"{self.phases[0].name}.json").read_text())
        events = [{"event": "mask_graph"}, {"event": "cleanup"}]
        plan = {"binaries": {v: {"path": "fake"} for v in partition.VARIANTS}}
        args = SimpleNamespace(kind="embedding", case="mixed", repeats=9, budget=8192,
                               variant="current", trial=0, prepared=self.root / "prepared",
                               context=self.root / "unused", output=self.root / "collected",
                               kinds=["embedding"], cases=["mixed"], trials=1, trial_start=0,
                               sustained_repeats=9)
        original_read = Path.read_text

        def observe(*args, event_prefix=None):
            self.assertEqual(event_prefix, "mask310 ")
            output, name = args[5:7]
            # Raw logs belong to the runner; callers must use its parsed result.
            (output / f"{name}.stderr").write_text("mask310 invalid if reopened")
            api.runner.save(output / f"{name}.monitor.json", [])
            return data, events

        def read(path, *args, **kwargs):
            self.assertNotEqual(path.suffix, ".stderr", "caller reopened closed stderr")
            return original_read(path, *args, **kwargs)

        with mock.patch.object(api.runner, "verify_plan", return_value=(plan, "fixed-plan")), \
             mock.patch.object(api, "model_conditions", side_effect=lambda _: ({}, {}, {})), \
             mock.patch.object(api.runner.host, "snapshot", return_value={"pressure": 1}), \
             mock.patch.object(api.runner, "observe", side_effect=observe), \
             mock.patch.object(Path, "read_text", read):
            partition.collect(api, args)
            output = self.root / "measured"
            api.measure(args.prepared, args.context, output, args)
            summary = json.loads((output / "summary.json").read_text())["embedding/mixed"]
            self.assertEqual(summary["mask_cleanup_series"]["current"], [events])
        self.assertTrue((args.output / "phase-complete.json").exists())

    def test_hash_and_summary_use_the_same_observation(self):
        original_read = Path.read_bytes
        original_text = Path.read_text
        reads = {}

        def observed(path):
            content = original_read(path)
            if path.parent in self.phases:
                reads[path] = reads.get(path, 0) + 1
                if reads[path] > 1:
                    if path.name == f"{path.parent.name}.json":
                        changed = json.loads(content)
                        changed[1]["metrics"]["wall"]["secs"] += 100
                        return json.dumps(changed).encode()
                    self.fail("phase evidence was read a second time")
            return content

        def changed_text(path, *args, **kwargs):
            if path.parent in self.phases:
                self.fail("phase evidence was reopened for parsing")
            return original_text(path, *args, **kwargs)

        baseline = self.root / "baseline"
        partition.combine(api, self.phases, baseline)
        output = self.root / "single-observation"
        with mock.patch.object(Path, "read_bytes", observed), \
             mock.patch.object(Path, "read_text", changed_text):
            partition.combine(api, self.phases, output)
        self.assertEqual((output / "summary.json").read_bytes(), (baseline / "summary.json").read_bytes())
        self.assertTrue(reads)
        self.assertTrue(all(count == 1 for count in reads.values()))
        phases = json.loads((output / "context.json").read_text())["phases"]
        self.assertEqual([p["phase_complete_sha256"] for p in phases],
                         [hashlib.sha256(original_read(d / "phase-complete.json")).hexdigest() for d in self.phases])

    def test_mixed_conditions_and_missing_or_changed_evidence_fail_closed(self):
        directory = self.phases[-1]
        context_path = directory / "context.json"
        context = json.loads(context_path.read_text())
        changed = dict(context, budget=8193)
        self.update_hashed(directory, context_path, changed)
        with self.assertRaisesRegex(ValueError, "different phase conditions"):
            partition.combine(api, self.phases, self.root / "mixed")
        self.update_hashed(directory, context_path, context)
        status_path = directory / f'{context["name"]}.status.json'
        status = json.loads(status_path.read_text())
        self.update_hashed(directory, status_path, dict(status, group_absent=False))
        with self.assertRaisesRegex(ValueError, "not complete or reclaimed"):
            partition.combine(api, self.phases, self.root / "unreclaimed")
        self.update_hashed(directory, status_path, status)
        complete_path = directory / "phase-complete.json"
        complete = json.loads(complete_path.read_text())
        omitted = copy.deepcopy(complete)
        del omitted["files"][status_path.name]
        api.runner.save(complete_path, omitted)
        with self.assertRaisesRegex(ValueError, "missing hashed"):
            partition.combine(api, self.phases, self.root / "unhashed")
        api.runner.save(complete_path, complete)
        status_path.write_text(status_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            partition.combine(api, self.phases, self.root / "altered")
        self.assertFalse(any((self.root / name).exists() for name in ("mixed","unreclaimed","unhashed","altered")))


if __name__ == "__main__":
    unittest.main()
