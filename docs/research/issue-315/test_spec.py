"""Contract examples, not inference or retrieval-quality tests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import spec

HERE = Path(__file__).resolve().parent


def document():
    return spec.read((HERE / "document.json").read_text())


def query():
    value = document()
    value["operation"].update(role="query", prefix="検索クエリ: ",
                              long_input="truncate-eos/v1", overlap=0)
    return value


class ContractExamples(unittest.TestCase):
    def test_identity_is_stable_but_preserves_meaning(self):
        value = document()
        reordered = {key: ({k: value[key][k] for k in reversed(value[key])}
                           if type(value[key]) is dict else value[key])
                     for key in reversed(value)}
        self.assertEqual(spec.fingerprint(value), spec.fingerprint(reordered))
        # An independent byte-level vector pins the encoding, not just self-equality.
        self.assertEqual(spec.canonical({"b": "検索 ", "a": 1}),
                         b'm2:s1:ai1;s1:bs7:\xe6\xa4\x9c\xe7\xb4\xa2 ')
        code = "import spec; print(spec.fingerprint(spec.read(open(sys.argv[1]).read())))"
        code = "import sys; " + code
        outputs = []
        with tempfile.TemporaryDirectory() as directory:
            for seed, cwd in [("1", HERE), ("123", HERE.parent)]:
                saved = Path(directory) / f"record-{seed}.json"
                saved.write_text(json.dumps(reordered), encoding="utf-8")
                env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(HERE))
                outputs.append(subprocess.check_output(
                    [sys.executable, "-B", "-c", code, str(saved)],
                    cwd=cwd, env=env, text=True).strip())
        self.assertEqual(outputs, [spec.fingerprint(value)] * 2)
        changed = copy.deepcopy(value)
        changed["operation"]["prefix"] += " "
        self.assertNotEqual(spec.fingerprint(value), spec.fingerprint(changed))

    def test_append_detects_same_dimension_changes(self):
        base = document()
        mutations = [
            ("common", "model", "other-model"),
            ("common", "revision", "other-revision"),
            ("common", "weights_sha256", "f" * 64),
            ("common", "config_sha256", "f" * 64),
            ("common", "tokenizer_sha256", "f" * 64),
            ("common", "tokenizer_semantics", "example.org/tokenizer/v2"),
            ("common", "pooling", "cls/v1"),
            ("common", "normalization", "none/v1"),
            ("common", "precision", "f16-weights-f32-pool/v1"),
            ("operation", "prefix", "別文書: "),
            ("operation", "outer_whitespace", "strip-after-prefix/python-3.12/v1"),
            ("operation", "outer_whitespace", "trim-body-before-prefix/v1"),
            ("operation", "long_input", "fixed-token-chunks/v1"),
            ("operation", "overlap", 1024),
            ("operation", "max_tokens", 4096),
            ("operation", "eos", 9),
        ]
        self.assertEqual(spec.append(base, base).status, "match")
        for section, field, replacement in mutations:
            with self.subTest(field=field, replacement=replacement):
                changed = copy.deepcopy(base)
                changed[section][field] = replacement
                result = spec.append(base, changed)
                self.assertEqual(result.status, "different")
                self.assertIn(f"{section}.{field}", result.differences)
                self.assertNotEqual(spec.fingerprint(base), spec.fingerprint(changed))

    def test_search_requires_an_explicit_pair_and_allows_query_only_change(self):
        doc, q = document(), query()
        pair = spec.pair(doc, q)
        self.assertEqual(spec.search(doc, q, pair).status, "match")
        self.assertEqual(spec.append(doc, q).status, "different")
        self.assertEqual(spec.append(q, q).status, "different")
        self.assertEqual(spec.search(doc, q, None).status, "unknown")
        shorter = copy.deepcopy(q)
        shorter["operation"]["max_tokens"] = 4096
        self.assertEqual(spec.search(doc, shorter, pair).status, "different")
        self.assertEqual(spec.search(doc, shorter, spec.pair(doc, shorter)).status, "match")
        self.assertEqual(spec.append(doc, doc).status, "match")  # no document rewrite
        arbitrary = copy.deepcopy(q)
        arbitrary["operation"].update(role="text", prefix="トピック: ")
        self.assertEqual(spec.search(doc, arbitrary, pair).status, "different")
        self.assertEqual(spec.search(doc, arbitrary, spec.pair(doc, arbitrary)).status, "match")
        # A pair cannot authorize a different model space by name or dimension alone.
        wrong = copy.deepcopy(q)
        wrong["common"]["revision"] = "other"
        self.assertEqual(spec.search(doc, wrong, spec.pair(doc, wrong)).status, "different")
        changed_doc = copy.deepcopy(doc)
        changed_doc["operation"]["overlap"] = 512
        self.assertEqual(spec.search(changed_doc, q, pair).status, "different")
        incomplete_query = copy.deepcopy(q)
        del incomplete_query["operation"]["prefix"]
        result = spec.search(changed_doc, incomplete_query, pair)
        self.assertEqual(result.status, "different")
        self.assertTrue(result.missing)  # keep insufficiency alongside the known document change

    def test_pair_owns_inputs_after_construction(self):
        for side in ["document", "query"]:
            with self.subTest(side=side):
                doc, q = document(), query()
                approved = spec.pair(doc, q)
                identity = spec.fingerprint(approved)
                self.assertEqual(spec.search(doc, q, approved).status, "match")
                source = doc if side == "document" else q
                source["operation"]["prefix"] += "changed"
                result = spec.search(doc, q, approved)
                self.assertEqual(result, spec.Comparison(
                    "different", (f"{side}.operation.prefix",)))
                self.assertEqual(spec.fingerprint(approved), identity)
                self.assertEqual(spec.search(doc, q, spec.pair(doc, q)).status, "match")

    def test_snapshot_owns_input_after_construction(self):
        source = document()
        snapshot = spec.Snapshot(source, "loaded-content", 4)
        identity = spec.fingerprint(snapshot.spec)
        self.assertEqual(spec.append(snapshot.spec, source).status, "match")
        source["operation"]["prefix"] += "changed"
        self.assertEqual(spec.append(snapshot.spec, source), spec.Comparison(
            "different", ("operation.prefix",)))
        self.assertEqual(spec.fingerprint(snapshot.spec), identity)
        self.assertEqual(spec.acquire(snapshot, 4).status, "match")
        self.assertEqual(spec.acquire(snapshot, 5).status, "unknown")

    def test_unknown_data_is_never_an_equal_legacy_default(self):
        value = document()
        for raw in ['{"kind":{}}', json.dumps({**value, "kind": {}})]:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(spec.Incomplete, "^kind: wrong type$"):
                    spec.fingerprint(spec.read(raw))
        for unknown in [None, {}, {**value, "schema": 2}, {**value, "future": True}]:
            with self.subTest(unknown=unknown):
                self.assertEqual(spec.append(unknown, unknown).status, "unknown")
                with self.assertRaises(spec.Incomplete):
                    spec.fingerprint(unknown)
        for section in ["common", "operation"]:
            missing = copy.deepcopy(value)
            del missing[section][next(iter(missing[section]))]
            self.assertEqual(spec.append(missing, missing).status, "unknown")
        extra = copy.deepcopy(value)
        extra["operation"]["future"] = True
        self.assertEqual(spec.append(extra, extra).status, "unknown")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            spec.read('{"schema":1,"schema":2}')
        bad = copy.deepcopy(value)
        bad["common"]["dimensions"] = True
        self.assertEqual(spec.append(bad, bad).status, "unknown")
        for raw in ['{"a":NaN}', '{"a":"\\ud800"}']:
            with self.assertRaises(ValueError):
                spec.read(raw)

    def test_acquisition_and_use_validate_and_encode_each_record_once(self):
        doc = document()
        now = spec.Snapshot(document(), "loaded-content", 4)
        current = spec.Snapshot(query(), "loaded-content", 4)
        approved = spec.pair(document(), query())  # independent of current inputs
        for name in ["_shape", "canonical", "_semantics"]:
            for route in ["create", "append", "search"]:
                with self.subTest(function=name, route=route):
                    snapshot = current if route == "search" else now
                    with patch.object(spec, name, wraps=getattr(spec, name)) as calls:
                        self.assertEqual(spec.acquire(snapshot, 4).status, "match")
                        if route == "create":
                            spec.fingerprint(snapshot.spec, expected_kind="embedding")
                            records = [snapshot.spec]
                        elif route == "append":
                            self.assertEqual(spec.append(doc, snapshot.spec).status, "match")
                            records = [doc, snapshot.spec]
                        else:
                            self.assertEqual(spec.search(doc, snapshot.spec, approved).status, "match")
                            records = [approved, approved["document"], approved["query"], doc, snapshot.spec]
                    for record in records:
                        self.assertEqual(sum(call.args[0] is record for call in calls.call_args_list), 1)
        # Validation cannot survive a later call with mutated records.
        approved["query"]["schema"] = 2
        self.assertEqual(spec.search(doc, current.spec, approved), spec.Comparison(
            "unknown", missing=("query.schema/kind: unsupported",)))
        now.spec["common"]["dimensions"] = 0
        self.assertEqual(spec.acquire(now, 4).status, "match")
        with self.assertRaisesRegex(spec.Incomplete, "invalid range"):
            spec.fingerprint(now.spec)

    def test_acquisition_does_not_authorize_incomplete_records(self):
        # A valid acquisition alone cannot authorize creation, append or search.
        mutations = [
            ("operation", "prefix", None, "operation.prefix: wrong type"),
            ("operation", "bos", -1, "unsupported canonical value"),
            ("common", "dimensions", 0, "dimensions/max_tokens/overlap: invalid range"),
        ]
        for section, field, value, reason in mutations:
            for role, source in [("document", document), ("query", query)]:
                with self.subTest(role=role, field=field):
                    record = source()
                    record[section][field] = value
                    snapshot = spec.Snapshot(record, "loaded-content", 4)
                    self.assertEqual(spec.acquire(snapshot, 4).status, "match")
                    with self.assertRaises(spec.Incomplete) as error:
                        spec.fingerprint(snapshot.spec, expected_kind="embedding")
                    self.assertEqual(str(error.exception), reason)
                    if role == "document":
                        result = spec.append(document(), snapshot.spec)
                    else:
                        result = spec.search(document(), snapshot.spec, spec.pair(document(), query()))
                    self.assertEqual(result, spec.Comparison("unknown", missing=(f"current.{reason}",)))
        # Generic IDs also support FTS/pairs; a new document index must not.
        for other in [spec.read((HERE / "fts.json").read_text()), spec.pair(document(), query())]:
            with self.subTest(kind=other["kind"]):
                snapshot = spec.Snapshot(other, "loaded-content", 4)
                self.assertEqual(spec.acquire(snapshot, 4).status, "match")
                with self.assertRaisesRegex(spec.Incomplete, "^schema/kind: unsupported$"):
                    spec.fingerprint(snapshot.spec, expected_kind="embedding")

    def test_search_keeps_validation_at_each_input_boundary(self):
        mutations = [
            ((), "schema", 2, "schema/kind: unsupported"),
            ((), "kind", {}, "kind: wrong type"),
            ((), "future", True, "future: unknown field"),
            (("common",), "dimensions", 0, "dimensions/max_tokens/overlap: invalid range"),
            (("common",), "dimensions", 2**64, "unsupported canonical value"),
            (("operation",), "bos", -1, "unsupported canonical value"),
            (("operation",), "long_input", "unknown/v2", "operation.long_input: unknown semantics"),
        ]
        for side in ["document", "query", "stored", "current"]:
            for path, key, replacement, reason in mutations:
                with self.subTest(side=side, key=key, replacement=replacement):
                    doc, q = document(), query()
                    approved = spec.pair(document(), query())
                    record = {"stored": doc, "current": q}.get(side, approved.get(side))
                    for section in path:
                        record = record[section]
                    record[key] = replacement
                    result = spec.search(doc, q, approved)
                    self.assertEqual(result.status, "unknown")
                    # Canonical transport errors apply to the whole pair.
                    prefix = "" if side in {"document", "query"} and reason == "unsupported canonical value" else (
                        f"{side}." if side in {"document", "query"} else "current.")
                    expected = (prefix + reason,)
                    if key == "long_input" and side in {"query", "current"}:
                        expected += (prefix + "operation: single-vector role requires truncate and zero overlap",)
                    self.assertEqual(result.missing, expected)

    def test_acquisition_and_generation_are_not_content_identity(self):
        value = document()
        a = spec.Snapshot(value, "loaded-content", 4)
        self.assertEqual(spec.acquire(a, 4).status, "match")
        self.assertEqual(spec.acquire(a, 5).status, "unknown")
        for evidence in ["declared-revision", "read-failed", "file-changed", "unsupported"]:
            self.assertEqual(spec.acquire(spec.Snapshot(value, evidence, 4), 4).status, "unknown")
        self.assertEqual(spec.acquire(spec.Snapshot(value, "custom-attested", 4), 4).status, "match")
        other = copy.deepcopy(value)
        other["common"]["namespace"] = "another.example/mock"
        self.assertEqual(spec.append(value, other).status, "different")

    def test_fts_index_contract_is_separate_from_search_policy(self):
        base = spec.read((HERE / "fts.json").read_text())
        self.assertEqual(spec.fts(base, base).status, "match")
        disabled = copy.deepcopy(base)
        disabled["normalization"] = dict.fromkeys(base["normalization"], False)
        self.assertEqual(spec.fts(disabled, disabled).status, "match")
        self.assertEqual(spec.fts(base, disabled).status, "different")
        mixed = {**base, "query_policy": "phrase"}
        self.assertEqual(spec.fts(mixed, mixed).status, "unknown")
        for field, value in [("nfkc", False), ("ascii_lowercase", False),
                             ("collapse_whitespace", False)]:
            changed = copy.deepcopy(base)
            changed["normalization"][field] = value
            self.assertEqual(spec.fts(base, changed).status, "different")
        for field in ["semantics", "tokenizer", "options", "preprocess"]:
            changed = copy.deepcopy(base)
            changed["consumer"][field] += "-changed"
            self.assertEqual(spec.fts(base, changed).status, "different")
        missing = copy.deepcopy(base)
        del missing["normalization"]
        self.assertEqual(spec.fts(missing, missing).status, "unknown")
        changed = copy.deepcopy(base)
        changed["normalization"]["future"] = True
        self.assertEqual(spec.fts(changed, changed).status, "unknown")
        changed = copy.deepcopy(base)
        changed["normalization_semantics"] = "unknown/v2"
        self.assertEqual(spec.fts(changed, changed).status, "unknown")


if __name__ == "__main__":
    unittest.main()
