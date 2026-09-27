"""Executable design example for #315; NOT a rurico API or model backend.

JSON is transport. canonical() defines the example's fingerprint bytes.
All semantic records are closed; evidence/runtime data must stay separate.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json


class Incomplete(ValueError):
    pass


def read(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"invalid JSON constant: {value}")

    result = json.loads(text, object_pairs_hook=unique, parse_constant=invalid)
    canonical(result)  # reject floats, negative integers and surrogate code points
    return result


def canonical(value):
    """Tagged UTF-8 bytes; counts/lengths are unsigned decimal without leading 0s."""
    if type(value) is bool:
        return b"t" if value else b"f"
    if type(value) is int and 0 <= value <= 2**64 - 1:
        return b"i" + str(value).encode("ascii") + b";"
    if type(value) is str:
        raw = value.encode("utf-8")
        return b"s" + str(len(raw)).encode("ascii") + b":" + raw
    if type(value) is dict and all(type(k) is str for k in value):
        keys = sorted(value, key=lambda k: k.encode("utf-8"))
        return b"m" + str(len(keys)).encode("ascii") + b":" + b"".join(
            canonical(k) + canonical(value[k]) for k in keys)
    if value is None:
        return b"n"  # legacy/unknown transport only; not a complete spec
    raise ValueError("unsupported canonical value")


COMMON = {
    "namespace": str, "model": str, "revision": str,
    "weights_sha256": str, "config_sha256": str, "tokenizer_sha256": str,
    "tokenizer_semantics": str, "dimensions": int, "pooling": str,
    "normalization": str, "precision": str, "inference_semantics": str,
}
OPERATION = {
    "role": str, "prefix": str, "outer_whitespace": str,
    "long_input": str, "max_tokens": int, "bos": int, "eos": int, "overlap": int,
}
EMBEDDING = {"schema": int, "kind": str, "common": COMMON, "operation": OPERATION}
FTS = {
    "schema": int, "kind": str, "normalization_semantics": str,
    "normalization": {"nfkc": bool, "ascii_lowercase": bool, "collapse_whitespace": bool},
    "consumer": {"namespace": str, "semantics": str, "tokenizer": str,
                 "options": str, "preprocess": str},
}
PAIR = {"schema": int, "kind": str, "document": EMBEDDING, "query": EMBEDDING}


def _shape(value, shape, path=""):
    if type(shape) is dict:
        if type(value) is not dict:
            return [f"{path or '$'}: missing record"]
        problems = [f"{path}{k}: unknown field" for k in sorted(value.keys() - shape.keys())]
        for key, child in shape.items():
            if key not in value:
                problems.append(f"{path}{key}: missing field")
            else:
                problems.extend(_shape(value[key], child, f"{path}{key}."))
        return problems
    if type(value) is not shape:
        return [f"{path[:-1]}: wrong type"]
    return []


def _validate(value, kind):
    """Validate once and return the same bytes for a caller that needs an ID."""
    shape = {"embedding": EMBEDDING, "fts": FTS, "pair": PAIR}[kind]
    issues = _shape(value, shape)
    if issues:
        return issues, None
    if value["schema"] != 1 or value["kind"] != kind:
        return ["schema/kind: unsupported"], None
    try:
        encoded = canonical(value)
    except ValueError as error:
        return [str(error)], None
    return _semantics(value, kind), encoded


def _semantics(value, kind):
    """The enclosing validation already checked shape and canonical transport."""
    issues = []
    if kind == "pair":
        for role in ["document", "query"]:
            child = value[role]
            if child["schema"] != 1 or child["kind"] != "embedding":
                issues.append(f"{role}.schema/kind: unsupported")
            else:
                issues.extend(f"{role}.{p}" for p in _semantics(child, "embedding"))
    elif kind == "embedding":
        common, op = value["common"], value["operation"]
        for field, item in common.items():
            if item == "":
                issues.append(f"common.{field}: unspecified")
            if field.endswith("_sha256") and (len(item) != 64 or any(c not in "0123456789abcdef" for c in item)):
                issues.append(f"common.{field}: invalid digest")
        if common["dimensions"] < 1 or op["max_tokens"] < 2 or op["overlap"] >= op["max_tokens"]:
            issues.append("dimensions/max_tokens/overlap: invalid range")
        # These are the example's understood operation semantics, not product adoption.
        choices = {
            "role": {"query", "document", "text"},
            "outer_whitespace": {"none/v1", "strip-after-prefix/python-3.12/v1", "trim-body-before-prefix/v1"},
            "long_input": {"offset-retokenize-shrink/v1", "fixed-token-chunks/v1", "truncate-eos/v1"},
        }
        for field, known in choices.items():
            if op[field] not in known:
                issues.append(f"operation.{field}: unknown semantics")
        if op["role"] in {"query", "text"} and (op["long_input"] != "truncate-eos/v1" or op["overlap"] != 0):
            issues.append("operation: single-vector role requires truncate and zero overlap")
    else:
        if value["normalization_semantics"] != "rurico/nfkc-ascii-collapse/v1":
            issues.append("normalization_semantics: unknown semantics")
        for field, item in value["consumer"].items():
            if not item:
                issues.append(f"consumer.{field}: unspecified")
    return issues


def problems(value, kind):
    return _validate(value, kind)[0]


def fingerprint(value, *, expected_kind=None):
    if type(value) is dict and "kind" in value and type(value["kind"]) is not str:
        raise Incomplete("kind: wrong type")
    kind = value.get("kind") if type(value) is dict else None
    if kind not in {"embedding", "fts", "pair"}:
        raise Incomplete("missing/unknown kind")
    if expected_kind is not None and kind != expected_kind:
        raise Incomplete("schema/kind: unsupported")
    issues, encoded = _validate(value, kind)
    if issues:
        raise Incomplete("; ".join(issues))
    raw = b"rurico/spec-example/1\0" + encoded
    return "spec-example-1:sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class Comparison:
    status: str
    differences: tuple = ()
    missing: tuple = ()


def _diff(left, right, path=""):
    result = []
    for key in sorted(left):
        name = path + key
        if type(left[key]) is dict:
            result.extend(_diff(left[key], right[key], name + "."))
        elif left[key] != right[key]:
            result.append(name)
    return result


def _compare(left, right, kind):
    return _comparison(left, right, problems(left, kind), problems(right, kind))


def _comparison(left, right, left_issues, right_issues):
    """Use validation results only within the current synchronous call."""
    missing = [f"stored.{p}" for p in left_issues]
    missing += [f"current.{p}" for p in right_issues]
    if missing:
        return Comparison("unknown", missing=tuple(missing))
    differences = tuple(_diff(left, right))
    return Comparison("different" if differences else "match", differences)


def append(stored, current):
    """Same complete operation, including text-vs-document and chunk policy."""
    result = _compare(stored, current, "embedding")
    if result.status == "match" and stored["operation"]["role"] == "query":
        return Comparison("different", ("operation.role: query is not an index producer",))
    return result


def pair(document, query):
    """Copy inputs into a proposal; the caller authorizes use and controls edits."""
    return {"schema": 1, "kind": "pair", "document": deepcopy(document), "query": deepcopy(query)}


def search(stored_document, current_query, approved_pair):
    missing = problems(approved_pair, "pair")
    if missing:
        return Comparison("unknown", missing=tuple(missing))
    # Pair validation includes both children; current inputs remain independent.
    doc = _comparison(approved_pair["document"], stored_document, (), problems(stored_document, "embedding"))
    query = _comparison(approved_pair["query"], current_query, (), problems(current_query, "embedding"))
    differences = [f"document.{p}" for p in doc.differences]
    differences += [f"query.{p}" for p in query.differences]
    # A consumer pair permits role differences, not arbitrary cross-model matching.
    differences += [f"pair.common.{p}" for p in _diff(
        approved_pair["document"]["common"], approved_pair["query"]["common"])]
    if approved_pair["document"]["operation"]["role"] not in {"document", "text"}:
        differences.append("pair.document.operation.role")
    if approved_pair["query"]["operation"]["role"] not in {"query", "text"}:
        differences.append("pair.query.operation.role")
    missing = doc.missing + query.missing
    status = "different" if differences else "unknown" if missing else "match"
    return Comparison(status, tuple(differences), missing)


def fts(index_side, query_side):
    """Both supply index-compatible preprocessing. Search policy is a separate input."""
    return _compare(index_side, query_side, "fts")


@dataclass(frozen=True)
class Snapshot:
    """Own a copy of the input spec, without freezing nested dicts or a producer."""
    spec: dict
    evidence: str
    generation: int

    def __post_init__(self):
        object.__setattr__(self, "spec", deepcopy(self.spec))


def acquire(snapshot, current_generation):
    """Gate evidence/generation only, without hashing files or locking a producer.

    A match does not validate spec: fingerprint/append/search must do that before
    any side effect, while the caller keeps the spec and producer fixed.
    """
    missing = []
    if snapshot.evidence not in {"loaded-content", "custom-attested"}:
        missing.append(f"acquisition: {snapshot.evidence}")
    if snapshot.generation != current_generation:
        missing.append("producer generation changed; reacquire under the same lease")
    return Comparison("unknown" if missing else "match", missing=tuple(missing))
