"""Run only on the host. Results go to a new directory outside the checkout."""
import argparse
import hashlib
import json
import os
import platform
import signal
import struct
import traceback
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from compare import compare, require
from reference import REVISIONS, environment
from reference_process import capture_isolated

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
START = "d0639cc815d03f44dc7c80abebe40e9157deaca1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")


def command(args, cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True).strip()


def source(root=ROOT):
    names = command(["git","ls-files","--cached","--others","--exclude-standard"], cwd=root).splitlines()
    return {name:sha(root/name) for name in names if (root/name).is_file()
            and "__pycache__" not in Path(name).parts}


def machine():
    versions = {}
    for name,args in {"rust":["rustc","-Vv"], "cargo":["cargo","-V"],
                      "xcode":["xcodebuild","-version"],"metal":["xcrun","metal","--version"],
                      "hardware":["sysctl","-n","hw.model"],"cpu":["sysctl","-n","machdep.cpu.brand_string"],
                      "memory_bytes":["sysctl","-n","hw.memsize"]}.items():
        try:
            versions[name] = "\n".join(
                line for line in command(args).splitlines()
                if not line.strip().startswith("InstalledDir:")
            )
        except (OSError,subprocess.CalledProcessError):
            versions[name] = None
    return {"os":platform.mac_ver()[0],"architecture":platform.machine(),
            "python":platform.python_version(),"tools":versions}


def models(out, revisions=None):
    from huggingface_hub import snapshot_download, hf_hub_url, get_hf_file_metadata
    paths, manifest = {}, {}
    for kind,(repo,rev) in (REVISIONS if revisions is None else revisions).items():
        path = Path(snapshot_download(repo,revision=rev,allow_patterns=["*.json","README.md","model.safetensors","tokenizer.model"]))
        files = {}
        for name in ("model.safetensors","config.json","tokenizer.json","tokenizer.model","tokenizer_config.json","special_tokens_map.json","README.md"):
            item = path/name
            metadata = get_hf_file_metadata(hf_hub_url(repo,name,revision=rev))
            require(metadata.commit_hash == rev, "HF revision mismatch")
            # LFS SHA-256 or Git blob SHA-1; validate actual content, not cache labels.
            etag = metadata.etag.strip('"')
            if len(etag) == 64:
                digest = sha(item)
                require(etag == digest, "HF LFS content mismatch")
            else:
                data = item.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                git_hash = hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
                require(etag == git_hash, "HF blob content mismatch")
            files[name] = {"sha256":digest,"hub_etag":etag}
        for item in sorted(path.rglob("*.json")):
            name = str(item.relative_to(path))
            if name not in files:
                files[name] = {"sha256":sha(item)}
        with (path/"model.safetensors").open("rb") as f:
            header = json.loads(f.read(struct.unpack("<Q",f.read(8))[0]))
        tensors = {k:v for k,v in header.items() if k != "__metadata__"}
        require(all(v["dtype"] == "F32" for v in tensors.values()), "expected FP32 checkpoint")
        config = json.loads((path/"config.json").read_text())
        require(config["local_attention"] == 128 and config["max_position_embeddings"] == 8192,
                "input boundary conditions need re-investigation")
        config.pop("_name_or_path",None)
        manifest[kind] = {"repository":repo,"revision":rev,"files":files,"config":config,
                          "weight_tensors":len(tensors),"weight_header_sha256":hashlib.sha256(json.dumps(header,sort_keys=True).encode()).hexdigest()}
        paths[kind] = path
    write(out/"models.json",manifest)
    return paths, manifest


def numerical(out, root=ROOT, candidate=None, variant=None):
    require(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple Silicon host required")
    packages = environment()
    require(not os.environ.get("RUSTFLAGS") and not os.environ.get("CARGO_ENCODED_RUSTFLAGS"), "use the documented default build flags")
    before = source(root)
    start, input_path, revisions = START, HERE/"inputs.json", REVISIONS
    if variant is not None:
        from variants import MODELS, START as variant_start, cases
        start = variant_start
        revision, dimensions = MODELS[variant]
        revisions = {"embedding": (f"cl-nagoya/ruri-v3-{variant}", revision)}
        input_path = out/"inputs.json"
        write(input_path, cases(variant))
    subprocess.run(["git","merge-base","--is-ancestor",start,"HEAD"],cwd=root,check=True)
    write(out/"environment.json",{"start_commit":start,"head":command(["git","rev-parse","HEAD"], cwd=root),"candidate_sha":candidate,
                                "dirty":bool(command(["git","status","--porcelain"], cwd=root)),"input_sha256":sha(input_path),"machine":machine(),"python_packages":packages,
                                "source":before,"started_utc":datetime.now(timezone.utc).isoformat(),
                                "mlx":"mlx-rs/MLX versions resolved by the hashed Cargo.lock and mlx-sys source"})
    paths, model_manifest = models(out, revisions)
    inputs = json.loads(input_path.read_text())
    if variant is not None:
        require(model_manifest["embedding"]["config"]["hidden_size"] == dimensions, "fixed model dimension mismatch")
    build = ["cargo","test","--locked","--lib","--features","smoke","--no-run","--message-format=json"]
    build_result = subprocess.run(build,cwd=root,text=True,capture_output=True)
    (out/"build.private.log").write_text(build_result.stdout+build_result.stderr)
    write(out/"build-attempt.json",{"command":build,"exit_code":build_result.returncode})
    require(build_result.returncode == 0,"build failed; inspect build.private.log")
    raw = build_result.stdout
    executables = [r["executable"] for line in raw.splitlines() if (r:=json.loads(line)).get("reason") == "compiler-artifact"
                   and r.get("executable") and r["target"]["name"] == "rurico" and r["profile"]["test"]]
    require(len(executables) == 1,"could not identify library test executable")
    binary = executables[0]
    write(out/"build.json",{"command":build,"executable_sha256":sha(binary),"cargo_lock_sha256":sha(root/"Cargo.lock"),
                           "rustflags":None,"profile":"test"})
    for kind in revisions:
        env = dict(os.environ,RURICO_307_INPUTS=str(input_path),RURICO_307_OUTPUT=str(out/f"rurico-{kind}.json"),RURICO_307_MODEL_DIR=str(paths[kind]))
        test = f"research::official_comparison_{kind}"
        if variant is not None:
            env["RURICO_367_MODEL"] = variant
            test = "research::official_comparison_variant"
        # Logs remain private: existing loader logs may contain local paths.
        with (out/f"{kind}.private.log").open("x") as log:
            subprocess.run([binary,test,"--exact","--ignored","--nocapture","--test-threads=1"],cwd=root,env=env,stdout=log,stderr=log,check=True)
        observed = json.loads((out/f"rurico-{kind}.json").read_text())
        reference = capture_isolated(kind,paths[kind],observed,inputs,out)
        write(out/f"reference-{kind}.json",reference)
        if variant is None:
            result = compare(observed,reference,inputs)
        else:
            from variants import evaluate
            result = evaluate(variant,observed,reference,inputs)
        write(out/f"comparison-{kind}.json", result)
    require(before == source(root),"source changed during observation")
    require(sha(binary) == json.loads((out/"build.json").read_text())["executable_sha256"],"binary changed")
    for kind,path in paths.items():
        for name,info in model_manifest[kind]["files"].items():
            require(sha(path/name) == info["sha256"],"model content changed during observation")
    write(out/"complete.json",{"execution":"complete","finished_utc":datetime.now(timezone.utc).isoformat(),
        "artifacts":{p.name:sha(p) for p in out.iterdir() if p.is_file() and not p.name.endswith(".private.log")},
        "note":"Execution complete does not mean numerical agreement; read both comparisons. Search quality is separate."})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=["numerical","variants","amici"])
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    require(os.environ.get("CODEX_SANDBOX") != "seatbelt","run on unsandboxed host")
    require(platform.system() == "Darwin" and platform.machine() == "arm64", "Apple Silicon host required")
    out = args.output.resolve()
    require(not out.is_relative_to(ROOT),"evidence directory must be outside checkout")
    out.mkdir(parents=True,exist_ok=False)
    if args.mode == "variants":
        def interrupted(signum, frame):
            raise KeyboardInterrupt("variant validation interrupted")
        signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.mode == "numerical":
            numerical(out)
        elif args.mode == "variants":
            from variants import MODELS, START as variant_start, cases, observe
            lane_source = source()
            write(out/"plan.json", {"start_commit": variant_start, "head": command(["git","rev-parse","HEAD"]),
                "source": lane_source, "cargo_lock_sha256": lane_source["Cargo.lock"],
                "public_document_options": {"token_budget": 8192,
                    "default_trait_batch": "documents whose chunks are at most 512 tokens; mixed and empty included",
                    "unverified": "default-budget batch with multiple long documents"},
                "models": {model: {"repository": f"cl-nagoya/ruri-v3-{model}", "revision": rev,
                    "dimensions": dim, "inputs": cases(model)["embedding"]} for model,(rev,dim) in MODELS.items()},
                "criteria": {"cosine_min": .99999, "max_abs": 1e-5,
                    "counts_order_dimensions_finite": "exact; input/document/chunk order preserved",
                    "applies_to": "normalized same-token vectors, padding, public/direct, batch/single, query/text",
                    "wrapper": "independent token/mask and numerical diagnostics; no preprocessing adoption",
                    "hidden_probes": "three sampled positions, max abs diagnostic only"}})
            results = {}
            for variant in MODELS:
                directory = out/variant
                directory.mkdir()
                require(lane_source == source(), "source changed between models")
                observe(directory, variant)
                result = json.loads((directory/"comparison-embedding.json").read_text())
                results[variant] = result["contract_within_predeclared_criteria"]
            require(lane_source == source(), "source changed across model lane")
            write(out/"variants.json", {"execution": "complete", "contracts": results,
                "note": "Separate model evidence; no shared vector index or product tolerance adoption."})
            require(all(results.values()), "variant criteria crossed; inspect comparisons, do not relax tolerances")
        else:
            from search_quality import run
            run(out)
    except BaseException as e:
        (out/"failure.private.log").write_text(traceback.format_exc())
        # Error text/traceback may contain personal paths; keep it private.
        write(out/"incomplete.json",{"execution":"incomplete","stage":args.mode,"error_type":type(e).__name__,
                                   "resume":"Inspect private log, resolve environment/input cause, rerun in a new directory. Never treat partial files as success."})
        raise


if __name__ == "__main__":
    main()
