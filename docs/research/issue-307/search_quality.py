"""Run the existing amici composition in a fresh copy of its agreed commit."""
import difflib
import json
import shutil
import subprocess
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

from compare import finite, require
from run import ROOT, START, command, machine, models, sha, source, write

AMICI = "547f9ee2ed734a2eab316fdbd62f194849875ee4"
OLD = "7b2b2565a9e6ac80732d4a153bc64d47b137368b"


def metrics_delta(baseline, measured):
    for key in ("schema_version","kind","fixture_hash","aggregation","merge_config","normalization"):
        require(baseline[key] == measured[key], f"search comparison condition differs: {key}")
    require([m["name"] for m in baseline["global"]] == [m["name"] for m in measured["global"]], "search metric set differs")
    result = []
    for before, after in zip(baseline["global"],measured["global"]):
        require(before["k"] == after["k"], "search cutoff differs")
        for m in (before, after):
            finite([m[k] for k in ("point_estimate","ci_lower","ci_upper")])
            require(m["ci_lower"] <= m["ci_upper"], "inverted CI")
        result.append({"name":before["name"],"k":before["k"],"baseline":before,"measured":after,
                       "delta":after["point_estimate"]-before["point_estimate"]})
    return result


def run(out):
    before_source = source()
    write(out/"environment.json",{"amici_commit":AMICI,"rurico_commit":START,"machine":machine(),"source":before_source})
    paths, model_manifest = models(out)
    archive = out/"amici.tar.gz"
    url = f"https://api.github.com/repos/thkt/amici/tarball/{AMICI}"
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"rurico-issue-307"}),timeout=60) as response:
        with archive.open("xb") as dest:
            shutil.copyfileobj(response,dest)
    checkout = out/"amici"
    extracted = out/"source"
    extracted.mkdir()
    with tarfile.open(archive) as tar:
        roots = {Path(x.name).parts[0] for x in tar.getmembers()}
        require(len(roots) == 1 and next(iter(roots)).endswith(AMICI[:7]), "unexpected archive revision")
        tar.extractall(extracted,filter="data")
    (extracted/next(iter(roots))).rename(checkout)
    original = {str(p.relative_to(checkout)):sha(p) for p in checkout.rglob("*") if p.is_file()}
    manifest = checkout/"Cargo.toml"
    old_text = manifest.read_text()
    needle = f'rurico = {{ git = "https://github.com/thkt/rurico", rev = "{OLD}"'
    require(old_text.count(needle) == 2, "amici dependency premise changed")
    new_text = old_text.replace(needle,needle.replace(OLD,START))
    manifest.write_text(new_text)
    (out/"amici-dependency.patch").write_text("".join(difflib.unified_diff(old_text.splitlines(True),new_text.splitlines(True),
                                                                              fromfile="a/Cargo.toml",tofile="b/Cargo.toml")))
    shutil.copyfile(checkout/"Cargo.lock",out/"amici-original.lock")
    fixture = checkout/"tests/fixtures/eval"
    shutil.copyfile(fixture/"baseline.json",out/"amici-existing-baseline.json")
    baseline = json.loads((fixture/"baseline.json").read_text())
    write(out/"amici-source.json", {"commit":AMICI,"archive_sha256":sha(archive),"original_source":original})
    steps = []

    def execute(label,args,accepted=(0,)):
        begin = time.monotonic()
        with (out/f"{label}.private.log").open("x") as log:
            result = subprocess.run(args,cwd=checkout,stdout=log,stderr=log)
        # argv paths are not exported; retain exact command templates in README.
        info = {"step":label,"exit_code":result.returncode,"elapsed_seconds":time.monotonic()-begin}
        write(out/f"{label}.json",info)
        steps.append(info)
        require(result.returncode in accepted,f"amici {label} failed; inspect private log")
        return result.returncode

    execute("resolve",["cargo","update","-p","rurico","--precise",START])
    shutil.copyfile(checkout/"Cargo.lock",out/"amici-resolved.lock")
    (out/"amici-lock.patch").write_text("".join(difflib.unified_diff((out/"amici-original.lock").read_text().splitlines(True),
                               (out/"amici-resolved.lock").read_text().splitlines(True),fromfile="a/Cargo.lock",tofile="b/Cargo.lock")))
    metadata = json.loads(command(["cargo","metadata","--locked","--format-version=1"],cwd=checkout))
    packages = [{k:p[k] for k in ("name","version","source")} for p in metadata["packages"]]
    rurico = [p for p in packages if p["name"] == "rurico"]
    require(len(rurico) == 1 and rurico[0]["source"].endswith("#"+START),"resolved rurico revision differs")
    write(out/"resolved-packages.json",packages)
    execute("build",["cargo","build","--locked","--features","eval-harness","--bin","eval_harness"])
    binary = Path(metadata["target_directory"])/"debug/eval_harness"
    binary_hash = sha(binary)
    require(baseline["aggregation"] == "identity", "inspect baseline aggregation options before running")
    merge = baseline["merge_config"]
    normal = baseline["normalization"]
    options = ["aggregation=identity",f'rrf_k={merge["rrf_k"]}',
               f'fts_weight={merge["source_weights"]["fts"]}',f'vector_weight={merge["source_weights"]["vector"]}',
               f'normalize_nfkc={str(normal["nfkc"]).lower()}',f'normalize_lowercase={str(normal["ascii_lowercase"]).lower()}',
               f'normalize_collapse_whitespace={str(normal["collapse_whitespace"]).lower()}']
    # A new measurement file only; the existing baseline is never overwritten.
    execute("measure",[str(binary),"capture-baseline",f'output={out/"amici-measured.json"}',*options])
    verify = execute("verify-existing",[str(binary),"verify-baseline",f'baseline={fixture/"baseline.json"}',*options],accepted=(0,1))
    measured = json.loads((out/"amici-measured.json").read_text())
    for name,digest in original.items():
        if name not in ("Cargo.toml","Cargo.lock"):
            require(sha(checkout/name) == digest,"amici source/fixture/baseline changed")
    require(manifest.read_text() == new_text and sha(checkout/"Cargo.lock") == sha(out/"amici-resolved.lock"),"amici configuration changed")
    require(before_source == source() and sha(binary) == binary_hash,"execution source changed")
    for kind,path in paths.items():
        for name,info in model_manifest[kind]["files"].items():
            require(sha(path/name) == info["sha256"],"model changed")
    documents = [json.loads(s) for s in (fixture/"documents.jsonl").read_text().splitlines() if s.strip()]
    queries = [json.loads(s) for s in (fixture/"queries.jsonl").read_text().splitlines() if s.strip()]
    write(out/"search-comparison.json",{
        "scope":"amici reference composition; not recall or the entire downstream application",
        "amici_commit":AMICI,"rurico_commit":START,"previous_rurico_dependency":OLD,
        "archive_sha256":sha(archive),"original_source_sha256":original,"binary_sha256":binary_hash,
        "documents":len(documents),"queries":len(queries),"query_categories":dict(Counter(q["category"] for q in queries)),
        "pipeline_arguments":options,"metrics":metrics_delta(baseline,measured),
        "bootstrap":{"confidence":.95,"resamples":1000,"seed":42},
        "verify_existing_exit_code":verify,"steps":steps,
        "metadata_limit":"amici embeds mlx_rs_version=0.25 as a constant. Preserve raw output; actual resolved versions are in resolved-packages.json and amici-resolved.lock. No speed conclusion from one evaluation.",
        "execution":"complete"})

    write(out/"complete.json", {"execution":"complete", "artifacts":{
        p.name:sha(p) for p in out.iterdir() if p.is_file() and not p.name.endswith((".private.log", ".tar.gz"))},
        "note":"Execution complete is separate from baseline agreement. Read search-comparison.json."})
