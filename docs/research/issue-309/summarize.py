"""Recompute the public #309 diagnostics; does not run inference or verify binaries."""
import argparse
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("length309", Path(__file__).with_name("run.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def compact(analysis):
    result = {}
    for case, row in analysis.items():
        result[case] = {key: value for key, value in row.items() if key != "diagnostics"}
        result[case]["diagnostics"] = []
        for d in row["diagnostics"]:
            rank = d["rank"]
            result[case]["diagnostics"].append({
                "trial": d["trial"], "repeat": d["repeat"],
                "max_logit_abs": max((v["logit_abs"] for v in d["differences"]), default=0),
                "max_score_abs": max((v["score_abs"] for v in d["differences"]), default=0),
                "exact_score_order": d["exact_score_order"],
                "exact_logit_order": rank is None or rank["reference_order"] == rank["actual_order"],
                "near_pair_count": len(rank["near_pairs"]) if rank else 0,
                "near_reversed_count": sum(p["reversed"] for p in rank["near_pairs"]) if rank else 0,
            })
    return result


def summarize(directory):
    groups = {}
    environment = json.loads((directory / "environment.json").read_text())
    for case in probe.CASES:
        for trial in range(3):
            for mode in ("reference", "candidate"):
                rows = json.loads((directory / f"{case}-{trial}-{mode}.json").read_text())
                headers = [r for r in rows if r["event"] == "input"]
                calls = [r for r in rows if r["event"] == "call"]
                probe.compare.require(len(headers) == 1, "incomplete public records")
                h = headers[0]
                probe.compare.require(h["workload"] == case and h["revision"] == probe.REVISION
                    and h["model_sha256"] == environment["model"]["files_sha256"], "different public model/workload")
                probe.compare.require(h["model"] == environment["model"]["repository"]
                    and h["dtype"] == environment["model"]["dtype"], "different public model/dtype")
                probe.validate_calls(h, calls, case, mode)
                groups[(case, trial, mode)] = h, calls
    return compact(probe.analyze(groups))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))
