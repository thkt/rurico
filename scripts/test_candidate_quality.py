"""#366 control regressions; no model, Cargo build, network or server."""
import copy
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import tarfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]/"docs/research/issue-307"
sys.path.insert(0, str(HERE))
import candidate
from run import ROOT, command, write, sha
import search_quality
from search_quality import verify_candidate, metrics_delta
from test_compare import reranker_records
from compare import compare


class CandidateQuality(unittest.TestCase):
    def test_full_sha_isolated_from_dirty_harness_and_invalid_revision_rejected(self):
        head = command(["git","rev-parse","HEAD"])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            harness = out/"harness"
            subprocess.run(["git","clone","--quiet","--no-hardlinks",str(ROOT),str(harness)], check=True)
            committed_lock = subprocess.check_output(["git","show",head+":Cargo.lock"], cwd=harness)
            (harness/"Cargo.lock").write_bytes(committed_lock+b"\n# uncommitted dependency edit\n")
            checkout, info = candidate.prepare(head, out, root=harness)
            self.assertEqual(info["head"], head)
            self.assertFalse(info["dirty"])
            self.assertEqual(sha(checkout/"Cargo.lock"), hashlib.sha256(committed_lock).hexdigest())
            self.assertNotEqual(sha(checkout/"Cargo.lock"), sha(harness/"Cargo.lock"))
            (checkout/"Cargo.toml").write_text("changed")
            self.assertNotEqual(candidate.identity(checkout)["source_sha256"], info["source_sha256"])
            self.assertTrue(candidate.identity(checkout)["dirty"])
            blob = command(["git","rev-parse",head+":Cargo.lock"], cwd=harness)
            for index, bad in enumerate(("HEAD", head[:7], "--help", "f"*40, blob)):
                fresh = out/f"invalid-{index}"
                fresh.mkdir()
                reason = "full candidate commit SHA" if index < 3 else "candidate is not a commit"
                with self.subTest(revision=bad), self.assertRaisesRegex(ValueError, reason):
                    candidate.prepare(bad, fresh, root=harness)
                self.assertFalse((fresh/"candidate").exists())

    def test_search_uses_fresh_target_and_rejects_old_actual_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            shared = out/"old-target"
            shared.mkdir()
            (shared/"eval_harness").write_text("old embedded manifest path")
            with patch.dict(os.environ, {"CARGO_TARGET_DIR":str(shared)}):
                env = search_quality.cargo_environment(out)
            self.assertEqual(env["CARGO_TARGET_DIR"], str((out/"build-target").resolve()))
            self.assertTrue((out/"build-target").is_dir())
            fixture = out/"amici/tests/fixtures/eval"
            for label, mode in (("measure", "capture-baseline"), ("verify-existing", "verify-baseline")):
                log = out/f"{label}.private.log"
                log.write_text(f"{mode}: fixture={fixture} path=/tmp/result model=model\n")
                search_quality.verify_fixture(log, mode, fixture)
                for text in (f"{mode}: fixture={shared} path=/tmp/result model=model\n", "no fixture observation\n"):
                    log.write_text(text)
                    with self.assertRaisesRegex(ValueError, "actual amici fixture differs"):
                        search_quality.verify_fixture(log, mode, fixture)
            with self.assertRaises(FileExistsError):
                search_quality.cargo_environment(out)

            # Exercise the caller too: resolver, metadata and build must share
            # the isolated target even when the ambient cache has an old binary.
            baseline = json.loads((HERE/"results/search/amici-existing-baseline.json").read_text())
            files = {
                "Cargo.toml": ('rurico = { git = "https://github.com/thkt/rurico", rev = "'+search_quality.OLD+'" }\n')*2,
                "Cargo.lock":"original lock",
                "tests/fixtures/eval/baseline.json":json.dumps(baseline),
                "tests/fixtures/eval/documents.jsonl":"{}\n"*60,
                "tests/fixtures/eval/queries.jsonl":"".join(json.dumps({"category":c})+"\n" for c in baseline["per_category"] for _ in range(21)),
            }
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w:gz") as tar:
                for name, content in files.items():
                    data = content.encode()
                    member = tarfile.TarInfo("amici-"+search_quality.AMICI[:7]+"/"+name)
                    member.size = len(data)
                    tar.addfile(member, io.BytesIO(data))
            fixed = out/"harness/results/search"
            fixed.mkdir(parents=True)
            write(fixed/"amici-source.json", {"commit":search_quality.AMICI,"original_source":{
                name:hashlib.sha256(content.encode()).hexdigest() for name,content in files.items()}})
            measured = out/"measurement"
            measured.mkdir()
            sha_value = "a"*40
            package = {"name":"rurico","version":"0.1.0","source":f"git+https://github.com/thkt/rurico?rev={sha_value}#{sha_value}","manifest_path":str(ROOT/"Cargo.toml")}
            observation = baseline
            def cargo_or_eval(args, cwd, env, stdout, stderr):
                self.assertEqual(env["CARGO_TARGET_DIR"], str((measured/"build-target").resolve()))
                if args[:2] == ["cargo","update"]:
                    (cwd/"Cargo.lock").write_text('[['+'package'+']]\n'+"\n".join(f'{k} = {json.dumps(package[k])}' for k in ("name","version","source")))
                elif args[:2] == ["cargo","metadata"]:
                    stdout.write(json.dumps({"packages":[package],"target_directory":env["CARGO_TARGET_DIR"]}))
                elif args[:2] == ["cargo","build"]:
                    binary = Path(env["CARGO_TARGET_DIR"])/"debug/eval_harness"
                    binary.parent.mkdir()
                    binary.write_text(str(cwd/"tests/fixtures/eval"))
                else:
                    self.assertEqual(args[0], str((measured/"build-target/debug/eval_harness").resolve()))
                    self.assertFalse(any(arg.startswith("fixture_dir=") for arg in args))
                    stdout.write(f"{args[1]}: fixture={Path(args[0]).read_text()} path=/tmp/result model=model\n")
                    if args[1] == "capture-baseline":
                        # Raw consumer output must reach the comparison guard,
                        # including malformed values our own writer refuses.
                        Path(args[2].removeprefix("output=")).write_text(json.dumps(observation))
                return subprocess.CompletedProcess(args, 0)
            with patch.dict(os.environ, {"CARGO_TARGET_DIR":str(shared)}), \
                 patch.object(search_quality,"HERE",out/"harness"), \
                 patch.object(search_quality,"machine",return_value={}), \
                 patch.object(search_quality,"models",return_value=({},{})), \
                 patch.object(search_quality,"source",return_value={}), \
                 patch.object(search_quality.urllib.request,"urlopen",side_effect=lambda *a,**k:io.BytesIO(archive.getvalue())), \
                 patch.object(search_quality,"owned_run",side_effect=cargo_or_eval):
                search_quality.run(measured.resolve(), candidate=sha_value)
                self.assertTrue((measured/"complete.json").exists())
                comparison = json.loads((measured/"search-comparison.json").read_text())
                self.assertEqual(comparison["metrics"], metrics_delta(baseline, baseline))
                self.assertEqual(comparison["per_category"], {
                    name:metrics_delta({**baseline,"global":rows}, {**baseline,"global":rows})
                    for name,rows in baseline["per_category"].items()})
                name = next(iter(baseline["per_category"]))
                missing = copy.deepcopy(baseline)
                del missing["per_category"][name]
                empty = copy.deepcopy(baseline)
                empty["per_category"][name] = []
                nonfinite = copy.deepcopy(baseline)
                nonfinite["per_category"][name][0]["point_estimate"] = float("nan")
                for index, (observation, reason) in enumerate(((missing, "search categories differ"),
                                                               (empty, "empty search metrics"),
                                                               (nonfinite, "non-finite"))):
                    measured = out/f"invalid-{index}"
                    measured.mkdir()
                    with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                        search_quality.run(measured.resolve(), candidate=sha_value)
                    self.assertFalse((measured/"search-comparison.json").exists())
                    self.assertFalse((measured/"complete.json").exists())

    def test_sigterm_entry_stops_stage_descendants_before_incomplete(self):
        for stage in ("numerical", "amici", "model-free-tests"):
            with self.subTest(stage=stage):
                with tempfile.TemporaryDirectory() as directory:
                    out = Path(directory)
                    stub = out/"stub"
                    stub.mkdir()
                    for name in ("inputs.json", "requirements.txt"):
                        (stub/name).write_bytes((HERE/name).read_bytes())
                    (stub/"processes.py").write_bytes((HERE/"processes.py").read_bytes())
                    # Keep real process boundaries; model inference and Cargo work are synthetic.
                    (stub/"run.py").write_text('''import subprocess, sys
def numerical(out, **kwargs):
    child = subprocess.Popen([sys.executable, "-c", "import os,sys,time; from pathlib import Path; p=Path(sys.argv[1]); f=(p/'writes').open('ab',buffering=0); f.write(b'x'); (p/'writer.pid').write_text(str(os.getpid()));\\nwhile True: f.write(b'x'); time.sleep(.01)", str(out)])
    child.wait()
if __name__ == '__main__':
    from pathlib import Path
    numerical(Path(sys.argv[1]))
''')
                    launcher = '''import sys, json
from pathlib import Path
from unittest.mock import patch
import candidate, search_quality
snapshot = candidate.identity(candidate.ROOT)
stub = Path(sys.argv[1])
with patch.object(candidate, 'HERE', Path(sys.argv[1])), patch.object(candidate, 'prepare', return_value=(candidate.ROOT,snapshot)), patch.object(candidate.platform,'system',return_value='Darwin'), patch.object(candidate.platform,'machine',return_value='arm64'), patch.dict(candidate.os.environ, {'CODEX_SANDBOX':''}):
    if sys.argv[3] == 'model-free-tests':
        import functools
        (stub/'test_ready.py').write_text('import unittest\\nclass Ready(unittest.TestCase):\\n    def test_ready(self): pass\\n')
        nested = stub/'nested.py'
        nested.write_text("import sys, signal, os\\nfrom pathlib import Path\\nfrom processes import owned_run\\ndef interrupted(signum, frame): raise KeyboardInterrupt()\\nsignal.signal(signal.SIGTERM, interrupted)\\nout = Path(sys.argv[1])\\n(out/'nested.pid').write_text(str(os.getpid()))\\nowned_run([sys.executable, str(sys.argv[2]), str(out)])\\n")
        tests = stub/'scripts'
        tests.mkdir()
        (tests/'test_candidate_quality.py').write_text("import unittest, subprocess, sys, os\\nfrom pathlib import Path\\nclass Nested(unittest.TestCase):\\n    def test_worker(self):\\n        out = Path("+repr(sys.argv[2])+ ")\\n        entry = subprocess.Popen([sys.executable, "+repr(str(nested))+", str(out), "+repr(str(stub/'run.py'))+"], start_new_session=True)\\n        (out/'runner.pid').write_text(str(os.getpid()))\\n        try: entry.wait()\\n        finally:\\n            entry.terminate()\\n            entry.wait(timeout=10)\\n")
        candidate.ROOT = stub
        candidate.identity = lambda root: snapshot
        candidate.execute = functools.partial(candidate.execute, root=stub)
        sys.argv = ['candidate', '--candidate', snapshot['head'], '--lane', 'light', '--output', sys.argv[2]]
        candidate.main()
    if sys.argv[3] == 'amici':
        def numerical(out, **kwargs):
            candidate.write(out/'proof.json', {})
            candidate.write(out/'complete.json', {'execution':'complete','artifacts':{'proof.json':candidate.sha(out/'proof.json')}})
        def search(out, **kwargs):
            search_quality.execute_command(out, 'build', [sys.executable, str(stub/'run.py'), str(out)], cwd=stub, env=candidate.os.environ)
        candidate.numerical = numerical
        candidate.search = search
    sys.argv = ['candidate', '--candidate', snapshot['head'], '--lane', 'model', '--output', sys.argv[2]]
    candidate.main()
'''
                    result = out/"result"
                    with (out/"entry.log").open("w") as log:
                        entry = subprocess.Popen([sys.executable,"-c",launcher,str(stub),str(result),stage], cwd=HERE, stdout=log, stderr=log)
                        writer = None
                        try:
                            deadline = time.monotonic()+10
                            work = result if stage == "model-free-tests" else result/stage
                            pidfile = work/"writer.pid"
                            while not pidfile.exists() and entry.poll() is None and time.monotonic() < deadline:
                                time.sleep(.01)
                            self.assertTrue(pidfile.exists(), (out/"entry.log").read_text())
                            writer = int(pidfile.read_text())
                            entry.send_signal(signal.SIGTERM)
                            self.assertNotEqual(entry.wait(timeout=10), 0)
                            self.assertEqual(json.loads((result/"incomplete.json").read_text())["stage"], stage)
                            self.assertFalse((result/"complete.json").exists())
                            with self.assertRaises(ProcessLookupError):
                                os.kill(writer, 0)
                            if stage == "model-free-tests":
                                for name in ("runner.pid", "nested.pid"):
                                    with self.assertRaises(ProcessLookupError):
                                        os.kill(int((work/name).read_text()), 0)
                            size = (work/"writes").stat().st_size
                            time.sleep(.1)
                            self.assertEqual((work/"writes").stat().st_size, size)
                        finally:
                            if entry.poll() is None:
                                entry.terminate()
                                try:
                                    entry.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    entry.kill()
                                    entry.wait()
                            if stage == "model-free-tests" and (result/"nested.pid").exists():
                                try:
                                    os.kill(int((result/"nested.pid").read_text()), signal.SIGTERM)
                                except ProcessLookupError:
                                    pass
                            if writer is not None:
                                try:
                                    os.kill(writer, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass

    def test_search_command_keeps_metadata_channels_and_allowed_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            args = [sys.executable, "-c", "import sys; print('{\"packages\":[]}'); print('diagnostic', file=sys.stderr)"]
            info = search_quality.execute_command(out, "metadata", args, cwd=ROOT, env=os.environ)
            self.assertEqual(info["exit_code"], 0)
            self.assertEqual(json.loads((out/"metadata.private.log").read_text()), {"packages":[]})
            self.assertIn("diagnostic", (out/"metadata.stderr.private.log").read_text())
            info = search_quality.execute_command(out, "verify-existing", [sys.executable,"-c","raise SystemExit(1)"],
                                                  cwd=ROOT, env=os.environ, accepted=(0,1))
            self.assertEqual(info["exit_code"], 1)
            with self.assertRaisesRegex(ValueError, "resolve failed"):
                search_quality.execute_command(out, "resolve", [sys.executable,"-c","raise SystemExit(2)"],
                                               cwd=ROOT, env=os.environ)
            self.assertEqual(json.loads((out/"resolve.json").read_text())["exit_code"], 2)

    def test_cargo_metadata_and_lock_must_both_resolve_one_exact_candidate(self):
        head = "a"*40
        expected = "git+https://github.com/thkt/rurico?rev="+head+"#"+head
        good = {"name":"rurico", "version":"0.1.0", "source":expected}
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory)/"Cargo.lock"
            def save(packages):
                lock.write_text("\n".join('[['+'package'+']]\n'+"\n".join(f'{k} = {json.dumps(v)}' for k,v in p.items()) for p in packages))
            save([good])
            verify_candidate({"packages":[good]}, lock, head)
            for packages in ([], [good,good], [{**good,"source":expected.replace(head,"b"*40)}],
                             [{**good,"source":expected.split("?")[0]+"#"+head}],
                             [{**good,"source":None}]):
                with self.subTest(metadata=packages), self.assertRaisesRegex(ValueError,"revision differs"):
                    verify_candidate({"packages":packages}, lock, head)
            for packages in ([], [good,good], [{**good,"source":expected.replace(head,"b"*40)}]):
                save(packages)
                with self.subTest(lock=packages), self.assertRaisesRegex(ValueError,"revision differs"):
                    verify_candidate({"packages":[good]}, lock, head)

    def test_existing_diagnostics_are_not_replaced_by_complete_or_empty_metrics(self):
        actual, reference, inputs = reranker_records()
        self.assertTrue(candidate.diagnostics(compare(actual, reference, inputs)))
        for delta, expected in ((.00002, True), (.0001, False)):
            changed = copy.deepcopy(actual)
            changed["public"]["singleton"][0]["score"] += delta
            self.assertEqual(candidate.diagnostics(compare(changed, reference, inputs)), expected)
        actual["public"]["ranking"].reverse()
        self.assertFalse(candidate.diagnostics(compare(actual, reference, inputs)))
        for kind in ("embedding","reranker"):
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                recorded = json.loads((HERE/f"results/numerical-353/comparison-{kind}.json").read_text())
                write(out/f"comparison-{kind}.json", recorded)
                self.assertTrue(candidate.numerical_result(out, kind))
                bad = copy.deepcopy(recorded)
                bad["same_tokens"].pop()
                (out/f"comparison-{kind}.json").write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,"observation count"):
                    candidate.numerical_result(out, kind)
        baseline = json.loads((HERE/"results/search/amici-existing-baseline.json").read_text())
        for bad in ({**baseline,"global":[]},
                    {**baseline,"global":[{**baseline["global"][0],"point_estimate":float("nan")}]}):
            with self.assertRaises(ValueError):
                metrics_delta(bad,bad)

    def test_model_lane_binds_both_stages_and_keeps_failures_incomplete(self):
        head = command(["git","rev-parse","HEAD"])
        snapshot = candidate.identity(ROOT)
        def measured(out, **kwargs):
            for kind in ("embedding","reranker"):
                write(out/f"comparison-{kind}.json", json.loads((HERE/f"results/numerical-353/comparison-{kind}.json").read_text()))
            write(out/"complete.json", {"execution":"complete", "artifacts":{
                p.name:sha(p) for p in out.iterdir()}})
        def searched(out, **kwargs):
            write(out/"search-comparison.json", {"verify_existing_exit_code":1})
            write(out/"complete.json", {"execution":"complete", "artifacts":{
                "search-comparison.json":sha(out/"search-comparison.json")}})
        with patch.object(candidate,"prepare",return_value=(ROOT,snapshot)), \
             patch.object(candidate.platform,"system",return_value="Darwin"), \
             patch.object(candidate.platform,"machine",return_value="arm64"), \
             patch.dict(candidate.os.environ,{"CODEX_SANDBOX":""}):
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(candidate,"numerical",side_effect=measured) as numeric, \
                 patch.object(candidate,"search",side_effect=searched) as search:
                out = Path(directory)
                self.assertEqual(candidate.execute(head,out,"model"),1)
                numeric.assert_called_once_with(out/"numerical",root=ROOT,candidate=head)
                search.assert_called_once_with(out/"amici",root=ROOT,candidate=head,verify_source=True)
                result = json.loads((out/"complete.json").read_text())
                self.assertTrue(result["model_measurement"])
                self.assertFalse(result["agreement"]["amici_baseline"])
            for error in (ValueError("skip or zero records"), KeyboardInterrupt()):
                with tempfile.TemporaryDirectory() as directory, \
                     patch.object(candidate,"numerical",side_effect=error), \
                     patch.object(candidate,"search") as search:
                    out = Path(directory)
                    with self.assertRaises(type(error)):
                        candidate.execute(head,out,"model")
                    self.assertFalse((out/"complete.json").exists())
                    self.assertEqual(json.loads((out/"incomplete.json").read_text())["stage"], "numerical")
                    search.assert_not_called()
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(candidate,"numerical",side_effect=lambda out,**kw: None), \
                 patch.object(candidate,"search") as search:
                with self.assertRaises(FileNotFoundError):
                    candidate.execute(head,Path(directory),"model")
                search.assert_not_called()
            with tempfile.TemporaryDirectory() as directory:
                stage = Path(directory)
                measured(stage)
                candidate.finish_stage(stage)
                (stage/"comparison-reranker.json").write_text("{}")
                with self.assertRaisesRegex(ValueError,"stage artifact changed"):
                    candidate.finish_stage(stage)


if __name__ == "__main__":
    unittest.main()
