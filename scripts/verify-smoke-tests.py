"""Reject missing/filtered smoke tests before nextest executes any test."""

import json
import sys


# These are the existing cached-model/Metal tests, not the model-free lane.
MODEL_TESTS = {
    "smoke_full",
    "smoke_verify_fixture",
    "smoke_measure_baseline",
    "probe_embed_smoke_binary",
    "probe_reranker_smoke_binary",
    "smoke_measure_overhead_observes_readbacks",
}


def verify(listing):
    for kind in ("bin", "test"):
        suites = [
            suite for suite in listing["rust-suites"].values()
            if suite["package-name"] == "rurico"
            and suite["binary-name"] == "mlx_smoke"
            and suite["kind"] == kind
        ]
        if len(suites) != 1:
            raise ValueError(f"mlx_smoke {kind}: expected one test suite")
        cases = suites[0]["testcases"]
        ignored = MODEL_TESTS if kind == "test" else set()
        if {name for name, case in cases.items() if case["ignored"]} != ignored:
            raise ValueError(f"mlx_smoke {kind}: unexpected ignored/model test selection")
        selected = {name: case for name, case in cases.items() if not case["ignored"]}
        if not selected:
            raise ValueError(f"mlx_smoke {kind}: zero model-free tests")
        for name, case in selected.items():
            if case["filter-match"]["status"] != "matches":
                raise ValueError(f"mlx_smoke {kind}: filtered model-free test {name}")
        print(f"mlx_smoke {kind}: {len(selected)} model-free tests; {len(ignored)} model tests ignored")


if __name__ == "__main__":
    try:
        with open(sys.argv[1], encoding="utf-8") as source:
            verify(json.load(source))
    except (ValueError, KeyError, TypeError) as error:
        sys.exit(f"smoke test selection failed: {error}")
