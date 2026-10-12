"""Reaggregate saved #318 observations without models or GPU execution."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from run import CASES, validate


def stats(values):
    if not values:
        raise ValueError("missing observations")
    return {"n": len(values), "min": min(values), "median": median(values), "max": max(values)}


def streams(records):
    groups = []
    current = None
    for row in records:
        event = row["event"]
        if event == "trial":
            if current is not None:
                raise ValueError("unfinished trial")
            current = {"config": row, "forward": [], "producer": [], "consumer": []}
        elif current is not None and event in ("forward", "producer", "consumer"):
            current[event].append(row)
        elif event == "stream":
            if current is None or row["completed"] != 50 or row["cancelled"]:
                raise ValueError("incomplete comparison stream")
            current["stream"] = row
            groups.append(current)
            current = None
    if current is not None:
        raise ValueError("unfinished final trial")
    return groups


def summarize(directory):
    observations = {}
    hashes = {}
    for case in CASES:
        data = (directory / f"{case}.json").read_bytes()
        rows = json.loads(data)
        validate(case, rows)
        observations[case] = rows
        hashes[f"{case}.json"] = hashlib.sha256(data).hexdigest()
    result = {"schema": 1, "inputs_sha256": hashes, "retention": [
        r for r in observations["retention"] if r["event"] == "retention"
    ]}
    for case, configs in (("forward", (1, 10, 25, 50)), ("overlap", (False, True))):
        groups = streams(observations[case])
        key = "cap" if case == "forward" else "overlap"
        summaries = []
        for config in configs:
            selected = [g for g in groups if g["config"][key] == config]
            if sorted(g["config"]["trial"] for g in selected) != [0, 1, 2]:
                raise ValueError("missing or duplicate trial")
            entry = {key: config,
                     "wall_ms": stats([g["stream"]["wall_ns"] / 1e6 for g in selected]),
                     "allocator_peak_mib": stats([max(r["peak_bytes"] for r in g["forward"]) / 2**20 for g in selected])}
            if case == "overlap" and config:
                entry["prepare_ms"] = stats([sum(p["prepared_ns"] - p["start_ns"] for p in g["producer"]) / 1e6 for g in selected])
                entry["prepare_forward_intersection_ms"] = stats([
                    sum(max(0, min(p["prepared_ns"], c["end_ns"]) - max(p["start_ns"], c["start_ns"]))
                        for p in g["producer"] for c in g["consumer"]) / 1e6
                    for g in selected])
            summaries.append(entry)
        result[case] = summaries
        direct = [r for r in observations[case] if r["event"] == "direct"]
        result[f"{case}_direct"] = {
            "wall_ms": stats([r["wall_ns"] / 1e6 for r in direct]),
            "allocator_peak_mib": stats([r["peak_bytes"] / 2**20 for r in direct]),
            "max_abs_score_difference": max(r["max_abs_score_difference"] for r in observations[case] if r["event"] == "comparison")}
    result["control"] = [r for r in observations["control"] if r["event"] in ("stream", "deadline")]
    result["fairness"] = []
    for priority in (False, True):
        for query in (True, False):
            rows = [r for r in observations["fairness"] if r["event"] == "fairness" and r["priority"] == priority and r["query"] == query]
            result["fairness"].append({"priority": priority, "query": query,
                                       "task_queue_wait_ms": stats([r["queue_wait_ns"] / 1e6 for r in rows])})
    result["ownership"] = []
    for shared in (True, False):
        rows = [r for r in observations["ownership"] if r["event"] == "ownership" and r["shared"] == shared]
        peaks = [r for r in observations["ownership"] if r["event"] == "ownership_peak" and r["shared"] == shared]
        result["ownership"].append({"shared": shared,
                                    "call_lock_wait_ms": stats([r["lock_wait_ns"] / 1e6 for r in rows]),
                                    "allocator_peak_mib": stats([r["peak_bytes"] / 2**20 for r in peaks])})
    result["mixed"] = []
    for parallel in (False, True):
        rows = [r for r in observations["mixed"] if r["event"] == "mixed_peak" and r["parallel"] == parallel]
        result["mixed"].append({"parallel": parallel,
                               "wall_ms": stats([r["wall_ns"] / 1e6 for r in rows]),
                               "allocator_peak_mib": stats([r["peak_bytes"] / 2**20 for r in rows])})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2, allow_nan=False))
