"""Model-free #309 evidence-boundary checks; no host processes or downloads."""
import copy
import json
import importlib.util
from pathlib import Path
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("length_batches", ROOT / "docs/research/issue-309/run.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
public_spec = importlib.util.spec_from_file_location("public_length_batches", ROOT / "docs/research/issue-309/summarize.py")
public = importlib.util.module_from_spec(public_spec)
public_spec.loader.exec_module(public)


def records(case, mode):
    lengths = {"same": [20]*50, "mixed": [20, 136]*25,
               **probe.BOUNDARIES,
               "limit": [8191, 8192, 8192], "empty": []}[case]
    models = [Path("/fixed") / probe.REVISION / name for name in ("model.safetensors", "config.json", "tokenizer.json")]
    header = {"event": "input", "workload": case, "revision": probe.REVISION,
              "model": "cl-nagoya/ruri-v3-reranker-310m", "dtype": "F32",
              "model_files": list(map(str, models)), "pairs": [["query", str(i)] for i in range(len(lengths))]}
    bucket = next((b for b in (128, 512, 2048, 8192) if b >= max(lengths, default=0)), 8192)
    chunks = [list(range(len(lengths)))] if lengths else []
    if mode == "candidate":
        chunks = [[i for i, n in enumerate(lengths) if next(b for b in (128, 512, 2048, 8192) if b >= n) == b]
                  for b in (128, 512, 2048, 8192)]
        chunks = [chunk for chunk in chunks if chunk]
    forwards = [{"indices": indices, "rows": len(indices),
                 "bucket": next(b for b in (128, 512, 2048, 8192) if b >= max(lengths[i] for i in indices))
                 if mode == "candidate" else bucket} for indices in chunks]
    padded = sum(f["rows"]*f["bucket"] for f in forwards)
    calls = [{"event": "call", "candidate": mode == "candidate", "repeat": i,
              "state": "cold" if i == 0 else "warm", "lengths": lengths,
              "logits": [0.0]*len(lengths), "scores": [.5]*len(lengths), "order": list(range(len(lengths))),
              "encode_ns": 1, "wall_ns": 100, "metal_peak_bytes": 1,
              "forwards": forwards, "real_tokens": sum(lengths), "padded_tokens": padded,
              "padding_ratio": padded/sum(lengths) if lengths else 1.0} for i in range(4)]
    return [header, *calls], models


class EvidenceTests(unittest.TestCase):
    def test_zero_selected_tests_missing_rows_or_wrong_model_cannot_complete(self):
        data, models = records("mixed", "candidate")
        probe.validate(data, "mixed", "candidate", models)
        for broken in ([], data[:-1]):
            with self.assertRaisesRegex(ValueError, "incomplete libtest"):
                probe.validate(broken, "mixed", "candidate", models)
        # HF snapshots may link the named files to content-addressed blobs.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / probe.REVISION
            snapshot.mkdir()
            for index, path in enumerate(models):
                blob = root / str(index)
                blob.write_text("fixture")
                (snapshot / path.name).symlink_to(blob)
            linked = [snapshot / path.name for path in models]
            data[0]["model_files"] = list(map(str, linked))
            probe.validate(data, "mixed", "candidate", linked)
        data[0]["model_files"] = list(map(str, models))
        data[1]["scores"] = data[1]["scores"][:-1]
        with self.assertRaisesRegex(ValueError, "lost/extra output"):
            probe.validate(data, "mixed", "candidate", models)
        data, models = records("mixed", "candidate")
        with self.assertRaisesRegex(ValueError, "model paths"):
            probe.validate(data, "mixed", "candidate", [Path("/wrong")])

    def test_comparison_keeps_logit_disagreement_and_input_identity_visible(self):
        self.assertFalse(probe.pair_diff([1.0, .7], [1.00004, .7])["within_criteria"])
        groups = {}
        for case in probe.CASES:
            for trial in range(3):
                for mode in ("reference", "candidate"):
                    data, models = records(case, mode)
                    groups[(case, trial, mode)] = probe.validate(data, case, mode, models)
        self.assertTrue(all(r["within_diagnostics"] for r in probe.analyze(groups).values()))
        # A changed raw logit with an unchanged sigmoid score must stay visible
        # through the #309 grouping and its agreed, stricter threshold.
        groups[("mixed", 1, "candidate")][1][2]["logits"][3] = .00005
        self.assertFalse(probe.analyze(groups)["mixed"]["within_diagnostics"])
        groups[("mixed", 1, "candidate")][0]["pairs"][0] = ["different query", "doc"]
        with self.assertRaisesRegex(ValueError, "different comparison inputs"):
            probe.analyze(groups)

    def test_public_replay_rejects_missing_outputs_and_invalid_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            hashes = {name: "fixed-hash" for name in ("model.safetensors", "config.json", "tokenizer.json")}
            model = {"repository": "cl-nagoya/ruri-v3-reranker-310m", "dtype": "F32", "files_sha256": hashes}
            (directory / "environment.json").write_text(json.dumps({"model": model}))
            for case in probe.CASES:
                for trial in range(3):
                    for mode in ("reference", "candidate"):
                        data, _ = records(case, mode)
                        del data[0]["model_files"]
                        data[0]["model_sha256"] = hashes
                        (directory / f"{case}-{trial}-{mode}.json").write_text(json.dumps(data))
            self.assertTrue(public.summarize(directory)["mixed"]["within_diagnostics"])
            path = directory / "mixed-0-candidate.json"
            original = json.loads(path.read_text())
            changes = [
                ({"logits": [], "scores": []}, "lost/extra output"),
                ({"scores": [float("nan")]*50}, "non-finite"),
                ({"candidate": False}, "wrong mode/state"),
                ({"state": "cold"}, "wrong mode/state"),
                ({"order": [0]*50}, "missing ranked identities"),
                ({"forwards": [{"bucket": 8192, "rows": 50, "indices": list(range(50))}]}, "bucket/budget"),
            ]
            for change, message in changes:
                broken = copy.deepcopy(original)
                broken[2].update(change)
                path.write_text(json.dumps(broken))
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                    public.summarize(directory)

    def test_overlapping_warm_ranges_do_not_establish_speed_signal(self):
        for candidate, expected in [([99, 99, 109], "unresolved"),
                                    ([80, 81, 82], "benefit"),
                                    ([130, 131, 132], "regression")]:
            groups = {}
            for case in probe.CASES:
                for trial in range(3):
                    for mode in ("reference", "candidate"):
                        data, models = records(case, mode)
                        value = [100, 110, 110][trial] if mode == "reference" else candidate[trial]
                        for call in data[1:]:
                            call["wall_ns"] = value
                        groups[(case, trial, mode)] = probe.validate(data, case, mode, models)
            self.assertEqual(probe.analyze(groups)["mixed"]["speed_signal"], expected)


if __name__ == "__main__":
    unittest.main()
