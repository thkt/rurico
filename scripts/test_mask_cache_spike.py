"""Model-free regressions for isolated #310 patches and actual readback evidence."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import sys
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from test_bucket_tables import embedding

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("spike310", ROOT / "docs/research/issue-310/run.py")
spike = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spike)


class MaskCacheSpike(unittest.TestCase):
    def test_variants_are_independent_and_source_drift_fails_closed(self):
        before = {p: (ROOT / p).read_text() for p in ("src/modernbert/model.rs", "src/mlx_cache.rs", "src/mlx_cache/testing/runtime.rs")}
        for variant in spike.patches.VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for p in before:
                    (root / p).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / p, root / p)
                spike.patches.patch(root, variant)
                model = (root / "src/modernbert/model.rs").read_text()
                cache = (root / "src/mlx_cache.rs").read_text()
                self.assertIn('"mask_graph"', model)
                self.assertEqual('let local_mask = global_mask.add(&local_mask)?;' in model, variant == "shared")
                self.assertEqual('let m = spike310::gpu_mask' in model, variant == "gpu")
                self.assertEqual('set_cache_limit(64' in cache, variant == "buffer")
                self.assertEqual('count >= 8' in cache, variant == "compile")
                self.assertEqual('SPIKE_CALLS' in cache, variant == "compile")
                self.assertEqual('SPIKE_FAILED.set(result.is_err())' in cache, variant in ("buffer", "compile"))
                if variant in ("buffer", "compile"):
                    self.assertIn('if SPIKE_FAILED.get()', cache)
                # Reusing an already patched copy must fail rather than mixing changes.
                with self.assertRaisesRegex(ValueError, "source changed"):
                    spike.patches.patch(root, variant)
        self.assertEqual(before, {p: (ROOT / p).read_text() for p in before})

    def test_wrong_or_missing_actual_readback_is_not_successful_accounting(self):
        data = embedding(spike.compare.TABLES["current"])
        for call in data[1:-1]:
            call["metrics"]["readback_elements"] = [f["batch_size"] * 768 for f in call["metrics"]["forwards"]]
        spike.validate(data, "embedding", "mixed", 4)
        for wrong in (None, [], [768]):
            changed = copy.deepcopy(data)
            changed[1]["metrics"]["readback_elements"] = wrong
            with self.assertRaisesRegex(ValueError, "actual readback"):
                spike.validate(changed, "embedding", "mixed", 4)
        events = [{"event":"buffer_clear"}, {"event":"compile_clear"},
                  {"event":"cleanup", "failed":True}] * 72
        spike.verify_cleanup(events)
        with self.assertRaisesRegex(ValueError, "failure did not clear"):
            spike.verify_cleanup(events[1:])
        with self.assertRaisesRegex(ValueError, "missing injected"):
            spike.verify_cleanup(events[:-3])
        changed = copy.deepcopy(data)
        changed[2]["documents"][0]["values"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            spike.validate(changed, "embedding", "mixed", 4)

    def test_closed_stderr_is_shared_by_records_and_mask_events(self):
        runner = spike.runner
        normal = {"rss_bytes": 1, "pressure": 1, "swap_bytes": 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = {"embedding": root / "model"}
            header = {"model_files": [str(models["embedding"] / p) for p in
                                      ("model.safetensors", "config.json", "tokenizer.json")]}
            records = [header, {"repeat": 0}]
            events = [{"event": "mask_graph"}, {"event": "cleanup"}]
            log = "noise\n" + "\n".join("bucket308 " + json.dumps(r) for r in records)
            log += "\n" + "\n".join("mask310 " + json.dumps(r) for r in events)
            process = SimpleNamespace(pid=123, returncode=0)
            original_read = Path.read_text
            reads = []

            def launch(*args, **kwargs):
                kwargs["stderr"].write(log)
                process.poll = mock.Mock(side_effect=[None, 0, 0])
                return process

            def read(path, *args, **kwargs):
                if path.suffix == ".stderr":
                    reads.append(path)
                    self.assertEqual(reads.count(path), 1)
                return original_read(path, *args, **kwargs)

            runner.started = time.monotonic()
            with mock.patch.object(runner.subprocess, "Popen", side_effect=launch), \
                 mock.patch.object(runner.host, "snapshot", return_value=normal), \
                 mock.patch.object(runner.host, "reclaim") as reclaim, \
                 mock.patch.object(Path, "read_text", read):
                observed = runner.observe(Path("fake"), "embedding", "mixed", 9, 8192,
                                          root, "shared", models, normal, runner.started + 5,
                                          event_prefix="mask310 ")
                self.assertEqual(observed, (records, events))
                default = runner.observe(Path("fake"), "embedding", "mixed", 9, 8192,
                                         root, "default", models, normal, runner.started + 5)
                self.assertEqual(default, records)
                self.assertEqual(reclaim.call_count, 2)
            self.assertEqual(len(reads), 2)
            self.assertTrue(json.loads((root / "shared.status.json").read_text())["group_absent"])

    def test_guards_preserve_partial_logs_and_reclaim_owned_workers(self):
        runner = spike.runner
        normal = {"rss_bytes": 0, "pressure": 1, "swap_bytes": 0}
        real_popen, real_killpg = runner.subprocess.Popen, os.killpg
        real_reclaim, real_save = runner.host.reclaim, runner.save
        # Both public paths use real child groups and file writes. Sensors and
        # launch arguments are simulated; signal delivery and reclamation are not.
        for entry in ("verify", "observe"):
            for failure in (None, "rss_bytes", "pressure", "swap_bytes", "deadline",
                            signal.SIGINT, signal.SIGTERM, "reclaim"):
                with self.subTest(entry=entry, failure=failure), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    row = dict(normal)
                    if failure in row:
                        row[failure] = {"rss_bytes": 8*1024**3 + 1, "pressure": 2,
                                        "swap_bytes": 256*1024**2 + 1}[failure]
                    models = {"embedding": output / "model"}
                    header = {"model_files": [str(models["embedding"] / p) for p in
                                              ("model.safetensors", "config.json", "tokenizer.json")]}
                    log = "partial\nbucket308 " + json.dumps(header) + '\nmask310 {"event":"cleanup"}'
                    command = [sys.executable, "-c",
                        "import signal,subprocess,sys,time; "
                        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                        "print(child.pid,flush=True); print(sys.argv[2],file=sys.stderr,flush=True); "
                        "time.sleep(30 if sys.argv[1] != 'normal' else .05)",
                        "normal" if failure is None else "failure", log]
                    processes, pgids, delivered = [], [], []
                    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

                    def interrupt(signum, frame):
                        raise KeyboardInterrupt(f"probe signal {signum}")

                    def launch(*args, **kwargs):
                        process = real_popen(command, **kwargs)
                        processes.append(process)
                        return process

                    def snapshot(pgid):
                        pgids.append(pgid)
                        # Synchronize with publication after installing SIG_IGN.
                        until = time.monotonic() + 2
                        while not (output / "probe.stdout").read_text().strip():
                            self.assertLess(time.monotonic(), until)
                            time.sleep(.01)
                        if len(pgids) > 1 and failure in (signal.SIGINT, signal.SIGTERM, "reclaim"):
                            os.kill(os.getpid(), failure if failure != "reclaim" else signal.SIGINT)
                        return dict(row)

                    def killpg(pgid, signum):
                        real_killpg(pgid, signum)
                        if signum == signal.SIGTERM and failure in (signal.SIGINT, signal.SIGTERM):
                            for sig in (signal.SIGINT, signal.SIGTERM):
                                delivered.append(("reclaim", sig))
                                os.kill(os.getpid(), sig)

                    def save(path, value):
                        if failure in (signal.SIGINT, signal.SIGTERM):
                            for sig in (signal.SIGINT, signal.SIGTERM):
                                delivered.append((path.suffixes[-2], sig))
                                os.kill(os.getpid(), sig)
                        real_save(path, value)

                    def reclaim(process):
                        real_reclaim(process)
                        if failure == "reclaim":
                            raise RuntimeError("injected reclamation failure")

                    started = time.monotonic()
                    deadline = started + 5
                    clock = mock.Mock(wraps=time)
                    clock.monotonic.side_effect = lambda: deadline + 1 if failure == "deadline" and pgids else time.monotonic()
                    runner.started = started
                    try:
                        for sig in handlers:
                            signal.signal(sig, interrupt)
                        with mock.patch.object(runner.subprocess, "Popen", side_effect=launch), \
                             mock.patch.object(runner.host, "snapshot", side_effect=snapshot), \
                             mock.patch.object(runner.host, "reclaim", side_effect=reclaim), \
                             mock.patch.object(runner, "save", side_effect=save), \
                             mock.patch.object(os, "killpg", side_effect=killpg), \
                             mock.patch.object(spike, "time", clock), \
                             mock.patch.object(runner, "time", clock):
                            def run():
                                if entry == "verify":
                                    return spike.guarded_verification(command, output, output, "probe",
                                                                      normal, started, deadline)
                                return runner.observe(Path("fake"), "embedding", "mixed", 9, 8192,
                                                      output, "probe", models, normal, deadline,
                                                      event_prefix="mask310 ")
                            if failure in (signal.SIGINT, signal.SIGTERM):
                                with self.assertRaisesRegex(KeyboardInterrupt, "probe signal"):
                                    run()
                                self.assertEqual({stage for stage, sig in delivered},
                                                 {"reclaim", ".monitor", ".status"})
                            elif failure == "reclaim":
                                with self.assertRaisesRegex(RuntimeError, "injected reclamation failure"):
                                    run()
                            elif failure:
                                with self.assertRaisesRegex(ValueError, f"{'verification' if entry == 'verify' else 'measurement'} guard"):
                                    run()
                            else:
                                result = run()
                                if entry == "observe":
                                    self.assertEqual(result, ([header], [{"event": "cleanup"}]))
                        for sig in handlers:
                            self.assertIs(signal.getsignal(sig), interrupt)
                        status = json.loads((output / "probe.status.json").read_text())
                        self.assertEqual(status["complete"], failure is None)
                        self.assertEqual(status.get("group_absent", False), failure != "reclaim")
                        self.assertIsNotNone(status["exit_code"])
                        self.assertIn("partial", (output / "probe.stderr").read_text())
                        self.assertTrue(json.loads((output / "probe.monitor.json").read_text()))
                        for process in processes:
                            self.assertIsNotNone(process.poll())
                        with self.assertRaises(ProcessLookupError):
                            real_killpg(pgids[0], 0)
                    finally:
                        for sig, handler in handlers.items():
                            signal.signal(sig, handler)
                        for process in processes:
                            real_reclaim(process)

    def test_verify_entry_connects_guards_and_retains_incomplete_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, output = root / "prepared", root / "evidence"
            prepared.mkdir()
            (prepared / "plan.json").write_text(json.dumps({"binaries": {"current": {"path": "libtest"}}}))
            baseline = {"rss_bytes": 0, "pressure": 1, "swap_bytes": 123}
            # Fix the clock at the mocked guard boundary; real elapsed-time
            # subtraction can round even when the 30-minute deadline is correct.
            with mock.patch.object(spike.time, "monotonic", return_value=123.25), \
                 mock.patch.object(spike, "model_conditions", return_value=({}, {}, {})), \
                 mock.patch.object(spike.runner, "verify_plan", return_value=({"binaries": {"current": {"path": "libtest"}}}, "verified-plan")), \
                 mock.patch.object(spike.runner.host, "snapshot", return_value=baseline), \
                 mock.patch.object(spike, "guarded_verification", side_effect=ValueError("verification guard reached")) as guard:
                with self.assertRaisesRegex(ValueError, "verification guard"):
                    spike.verify(prepared, root / "context", output)
            args = guard.call_args.args
            self.assertEqual(args[3:5], ("current-cleanup", baseline))
            self.assertEqual(args[5:7], (123.25, 1923.25))
            self.assertEqual(json.loads((output / "context.json").read_text())["guards"],
                             {"seconds": 1800, "rss_bytes": 8*1024**3,
                              "swap_increase_bytes": 256*1024**2, "pressure": 1})
            self.assertEqual(json.loads((output / "context.json").read_text())["plan_sha256"], "verified-plan")
            self.assertFalse((output / "complete.json").exists())
            self.assertFalse(json.loads((output / "incomplete.json").read_text())["verification_complete"])

            plan = {"binaries": {v: {"path": "libtest"} for v in spike.patches.VARIANTS}}
            model_dirs = {kind: root / kind for kind in spike.runner.TESTS}
            models = {str(directory / filename): "fixed-model" for directory in model_dirs.values()
                      for filename in ("model.safetensors", "config.json", "tokenizer.json")}
            cleanup = [{"event": "buffer_clear"}, {"event": "compile_clear"},
                       {"event": "cleanup", "failed": True}] * 72
            cleanup += [{"event": "model_files", "kind": kind,
                         "paths": [p for p in models if Path(p).parent == directory]}
                        for kind, directory in model_dirs.items()]
            masks = [{"event": "mask_elements"}] * 48

            for fault, message in ((None, None), ("cleanup", "failure did not clear"),
                                   ("identity", "cleanup model differs"),
                                   ("mask", "incomplete full-mask"), ("plan", "plan changed"),
                                   ("model", "model changed")):
                with self.subTest(fault=fault):
                    destination = root / f"finished-{fault}"

                    def finished(args, cwd, output, name, baseline, started, deadline):
                        events = copy.deepcopy(masks if name.endswith("-mask") else cleanup)
                        if fault == "cleanup" and name == "current-cleanup":
                            events.pop(0)
                        if fault == "identity" and name == "current-cleanup":
                            events[-1]["paths"][0] = str(root / "other-model")
                        if fault == "mask" and name == "current-mask":
                            events.pop()
                        (output / f"{name}.stdout").write_text("1 passed; 0 failed; 0 ignored")
                        (output / f"{name}.stderr").write_text("noise\n" + "\n".join(
                            "mask310 " + json.dumps(event) for event in events))

                    def checked_plan(prepared, expected=None):
                        if fault == "plan" and expected is not None:
                            raise ValueError("plan changed")
                        return plan, "verified-plan"

                    with mock.patch.object(spike.time, "monotonic", return_value=123.25), \
                         mock.patch.object(spike, "model_conditions", return_value=({}, model_dirs, models)), \
                         mock.patch.object(spike.runner, "verify_plan", side_effect=checked_plan) as plans, \
                         mock.patch.object(spike.runner.host, "snapshot", return_value=baseline), \
                         mock.patch.object(spike.runner.host, "digest", return_value=(
                             "changed-model" if fault == "model" else "fixed-model")), \
                         mock.patch.object(spike, "guarded_verification", side_effect=finished) as guard:
                        if fault is None:
                            spike.verify(prepared, root / "context", destination)
                            self.assertEqual([call.args[3] for call in guard.call_args_list],
                                             ["current-cleanup", "current-mask", "shared-cleanup",
                                              "gpu-cleanup", "buffer-cleanup", "compile-cleanup"])
                            self.assertTrue(all(call.args[5:7] == (123.25, 1923.25)
                                                for call in guard.call_args_list))
                            self.assertEqual(plans.call_args_list,
                                             [mock.call(prepared), mock.call(prepared, plan)])
                            for name in ("current-cleanup", "shared-cleanup", "gpu-cleanup",
                                         "buffer-cleanup", "compile-cleanup"):
                                self.assertEqual(json.loads((destination / f"{name}.json").read_text()), cleanup)
                            self.assertEqual(json.loads((destination / "current-mask.json").read_text()), masks)
                            complete = json.loads((destination / "complete.json").read_text())
                            self.assertTrue(complete["verification_complete"])
                            self.assertIn("compile-cleanup.json", complete["files"])
                            self.assertFalse((destination / "incomplete.json").exists())
                        else:
                            with self.assertRaisesRegex(ValueError, message):
                                spike.verify(prepared, root / "context", destination)
                            self.assertFalse((destination / "complete.json").exists())
                            self.assertFalse(json.loads((destination / "incomplete.json").read_text())["verification_complete"])


if __name__ == "__main__":
    unittest.main()
