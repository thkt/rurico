"""Pinned Sentence Transformers / Transformers FP32 CPU reference for #307."""
import importlib.metadata
import platform
import re
import sys
from pathlib import Path

from compare import require

REVISIONS = {
    "embedding": ("cl-nagoya/ruri-v3-310m", "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3"),
    "reranker": ("cl-nagoya/ruri-v3-reranker-310m", "bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3"),
}

def normalized_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def requirements():
    """The installed and verified runtime versions have one source of truth."""
    from packaging.version import Version

    pins = {}
    for line in Path(__file__).with_name("requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==(\S+)", line)
        require(match is not None, f"expected an exact stable dependency pin: {line}")
        name, version = normalized_name(match[1]), match[2]
        require(not Version(version).is_prerelease, f"expected a stable dependency pin: {line}")
        require(name not in pins, f"duplicate dependency pin: {name}")
        pins[name] = version
    require({"torch", "transformers", "sentence-transformers"} <= pins.keys(),
            "reference libraries must be pinned")
    return pins


def environment():
    require(sys.prefix != sys.base_prefix, "use an isolated Python environment")
    require(platform.python_version() == "3.12.13", "use Python 3.12.13")
    pins = requirements()
    installed = {normalized_name(d.metadata["Name"]): d.version
                 for d in importlib.metadata.distributions()}
    for name, version in pins.items():
        require(installed.get(name) == version, f"install pinned {name}=={version}")
    # pip bootstraps the venv; all runtime dependencies, including transitives,
    # must be listed so a successful resolver cannot silently expand the set.
    require(installed.keys() - {"pip"} == pins.keys(), "unlisted runtime dependencies installed")
    return installed


def capture(kind, snapshot, observed, inputs):
    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder

    environment()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    kwargs = {"dtype":torch.float32, "attn_implementation":"eager"}
    if kind == "embedding":
        wrapper = SentenceTransformer(str(snapshot), device="cpu", model_kwargs=kwargs,
                                      local_files_only=True, trust_remote_code=False)
        model = wrapper[0].model
        require(wrapper.max_seq_length == 8192, "official wrapper input limit changed")
        require(len(wrapper) == 2 and wrapper[1].pooling_mode == "mean" and wrapper[1].include_prompt,
                "inspect fixed official pooling configuration")
    else:
        wrapper = CrossEncoder(str(snapshot), device="cpu", max_length=8192,
                               model_kwargs=kwargs, local_files_only=True, trust_remote_code=False)
        model = wrapper.model
    model.float().eval()
    require(not model.training and all(p.dtype == torch.float32 for p in model.parameters()), "FP32/eval required")
    require(model.config._attn_implementation == "eager", "attention implementation changed")
    # Transformers 5 removed ModernBERT's reference_compile option and compiled
    # helpers. Reject compiled modules and force eager execution for this capture.
    require(all(module._compiled_call_impl is None for module in wrapper.modules()),
            "reference compilation must remain disabled")
    tokenizer = wrapper.tokenizer
    tokenization, native = [], []
    with torch.inference_mode(), torch.compiler.set_stance("force_eager"):
        for case in inputs[kind]:
            text = case["text"] * case["repeat"]
            if kind == "embedding":
                text = case["prefix"] + text
                raw = tokenizer(text, truncation=False)
                encoded = wrapper.preprocess([text])
                # Official wrapper handles strip/truncation/pooling. Normalize
                # explicitly because the model card's encode does not request it.
                value = wrapper.encode([text], batch_size=1, normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False)[0].tolist()
            else:
                raw = tokenizer(case["query"],text,truncation=False)
                encoded = wrapper.preprocess([(case["query"], text)])
                logit = []

                class RecordingSigmoid(torch.nn.Module):
                    def forward(self, logits):
                        # Observe pre-sigmoid logits from this same predict forward.
                        logit.extend(logits.flatten().tolist())
                        return torch.sigmoid(logits)

                score = wrapper.predict([(case["query"],text)], batch_size=1,
                                        activation_fn=RecordingSigmoid(),show_progress_bar=False)
                require(len(logit) == len(score) == 1, "expected one wrapper logit and score")
                value = [float(logit[0]),float(score[0])]
            tokenization.append({"id":case["id"],"raw_ids":raw["input_ids"],"raw_mask":raw["attention_mask"],
                                 "model_ids":encoded["input_ids"][0].tolist(),
                                 "model_mask":encoded["attention_mask"][0].tolist()})
            native.append({"id":case["id"],"value":value})
        batches = []
        for source in observed["batches"]:
            ids = torch.tensor(source["ids"],dtype=torch.long)
            mask = torch.tensor(source["mask"],dtype=torch.long)
            # Intentionally bypass tokenization: always replay rurico tokens,
            # including its independently labelled chunks and bucket padding.
            output = model(input_ids=ids,attention_mask=mask,return_dict=True)
            probes = []
            if kind == "embedding":
                hidden = output.last_hidden_state
                pooled = wrapper[1]({"token_embeddings":hidden,"attention_mask":mask})["sentence_embedding"]
                values = torch.nn.functional.normalize(pooled,p=2,dim=1).tolist()
                for i, row in enumerate(source["mask"]):
                    last = max(j for j,x in enumerate(row) if x)
                    probes.append(hidden[i,[0,last//2,last],:].tolist())
            else:
                logits = output.logits.flatten()
                values = torch.stack((logits,torch.sigmoid(logits)),dim=1).tolist()
            batches.append({**{k:source[k] for k in ("id","keys","ids","mask","conditions")},
                            "values":values,"hidden_probes":probes})
    return {"schema":observed["schema"],"producer":"transformers","kind":kind,"tokenization":tokenization,
            "batches":batches,"native":native,"conditions":{
                "device":"cpu","dtype":"float32","attention":"eager","training":False,"reference_compile":False,
                "compiler_stance":"force_eager",
                "dropout":"disabled by eval; configured probabilities recorded in model manifest",
                "threads":4,"deterministic_algorithms":True,"seed":0,
                "pooling":"official mean, include prompt, explicit L2" if kind == "embedding" else "official CLS/head/classifier; sigmoid",
                "same_tokens":"rurico input IDs and masks replayed without tokenizer or wrapper truncation"}}
