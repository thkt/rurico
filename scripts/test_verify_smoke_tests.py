"""Selection regressions: a successful workspace run can omit smoke tests."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "verify_smoke_tests", Path(__file__).with_name("verify-smoke-tests.py")
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def listing():
    def case(ignored=False):
        return {"ignored": ignored, "filter-match": {"status": "matches"}}

    return {"rust-suites": {
        "rurico::bin/mlx_smoke": {
            "package-name": "rurico", "binary-name": "mlx_smoke", "kind": "bin",
            "testcases": {"tests::threshold": case(), "records::tests::summary": case()},
        },
        "rurico::mlx_smoke": {
            "package-name": "rurico", "binary-name": "mlx_smoke", "kind": "test",
            "testcases": {
                "summarize_records": case(), "compare_records": case(),
                **{name: case(True) for name in (
                    "smoke_full", "smoke_verify_fixture", "smoke_measure_baseline",
                    "probe_embed_smoke_binary", "probe_reranker_smoke_binary",
                    "smoke_measure_overhead_observes_readbacks",
                )},
            },
        },
    }}


class SmokeSelectionTests(unittest.TestCase):
    def test_complete_selection_and_expected_model_ignores(self):
        verifier.verify(listing())

    def test_missing_empty_filtered_and_ignored_model_free_tests_are_rejected(self):
        for kind, key in (("bin", "rurico::bin/mlx_smoke"), ("test", "rurico::mlx_smoke")):
            for fault in ("missing", "empty", "filtered", "ignored"):
                with self.subTest(kind=kind, fault=fault):
                    value = copy.deepcopy(listing())
                    suites = value["rust-suites"]
                    cases = suites[key]["testcases"]
                    if fault == "missing":
                        del suites[key]
                    elif fault == "empty":
                        suites[key]["testcases"] = {
                            name: case for name, case in cases.items() if case["ignored"]
                        }
                    else:
                        case = next(case for case in cases.values() if not case["ignored"])
                        if fault == "filtered":
                            case["filter-match"] = {"status": "mismatch", "reason": "expression"}
                        else:
                            case["ignored"] = True
                    with self.assertRaisesRegex(ValueError, "expected one|zero model-free|filtered|unexpected ignored"):
                        verifier.verify(value)

    def test_model_tests_must_remain_ignored_before_execution(self):
        value = listing()
        value["rust-suites"]["rurico::mlx_smoke"]["testcases"]["smoke_full"]["ignored"] = False
        with self.assertRaisesRegex(ValueError, "unexpected ignored/model"):
            verifier.verify(value)

    def test_runner_propagates_errors_and_blocks_execution_after_bad_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cargo = root / "cargo"
            cargo.write_text('''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
with open(os.environ["SMOKE_CALLS"], "a") as log:
    log.write(json.dumps(args) + "\\n")

def option(name):
    return args[args.index(name) + 1]

if args[0] == "metadata":
    assert "--locked" in args
    assert option("--features") == "test-support,test-mlx,smoke"
    print('{"workspace": "current"}')
    sys.exit(int(os.environ["SMOKE_METADATA_EXIT"]))
assert args[0] == "nextest"
assert option("--profile") == "ci"
assert option("--run-ignored") == "default"
assert json.loads(Path(option("--cargo-metadata")).read_text()) == {"workspace": "current"}
if args[1] == "list" and "--list-type" in args:
    assert option("--list-type") == "binaries-only"
    assert "--locked" in args and "--workspace" in args
    assert option("--features") == "test-support,test-mlx,smoke"
    print('{"binaries": "fresh"}')
    sys.exit(int(os.environ["SMOKE_BUILD_EXIT"]))
assert "--features" not in args and "--workspace" not in args
assert json.loads(Path(option("--binaries-metadata")).read_text()) == {"binaries": "fresh"}
if args[1] == "list":
    print(Path(os.environ["SMOKE_LIST"]).read_text())
    sys.exit(int(os.environ["SMOKE_LIST_EXIT"]))
assert args[1] == "run"
assert option("--no-tests") == "fail"
assert option("--status-level") == option("--final-status-level") == "all"
Path(os.environ["SMOKE_RUN_MARKER"]).touch()
sys.exit(int(os.environ["SMOKE_RUN_EXIT"]))
''')
            cargo.chmod(0o755)
            source = root / "listing.json"
            marker = root / "ran"
            calls = root / "calls.jsonl"
            temporary = root / "temporary"
            temporary.mkdir()
            for fault, expected, ran in (
                ("normal", 0, True), ("metadata", 21, False),
                ("build", 23, False), ("list", 19, False),
                ("filtered", 1, False), ("run", 17, True),
            ):
                with self.subTest(fault=fault):
                    value = listing()
                    if fault == "filtered":
                        value["rust-suites"]["rurico::bin/mlx_smoke"]["testcases"]["tests::threshold"]["filter-match"] = {"status": "mismatch"}
                    source.write_text(json.dumps(value))
                    marker.unlink(missing_ok=True)
                    calls.unlink(missing_ok=True)
                    environment = {
                        **os.environ, "PATH": f"{root}:{os.environ['PATH']}",
                        "SMOKE_LIST": str(source), "SMOKE_RUN_MARKER": str(marker),
                        "SMOKE_CALLS": str(calls),
                        "TMPDIR": str(temporary),
                        "SMOKE_METADATA_EXIT": "21" if fault == "metadata" else "0",
                        "SMOKE_BUILD_EXIT": "23" if fault == "build" else "0",
                        "SMOKE_LIST_EXIT": "19" if fault == "list" else "0",
                        "SMOKE_RUN_EXIT": "17" if fault == "run" else "0",
                    }
                    result = subprocess.run(
                        ["bash", "scripts/test.sh"], env=environment,
                        cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                    self.assertEqual(marker.exists(), ran)
                    self.assertEqual(list(temporary.iterdir()), [])
                    invocations = [json.loads(line) for line in calls.read_text().splitlines()]
                    builds = [args for args in invocations if "--list-type" in args]
                    self.assertEqual(len(builds), 0 if fault == "metadata" else 1)
                    self.assertEqual(sum(args[0] == "metadata" for args in invocations), 1)
                    for option in ("--cargo-metadata", "--binaries-metadata"):
                        paths = [args[args.index(option) + 1] for args in invocations if option in args]
                        if paths:
                            self.assertEqual(len(set(paths)), 1)
                            self.assertFalse(Path(paths[0]).parent.exists())



if __name__ == "__main__":
    unittest.main()
