"""Pinned Sentence Transformers / Transformers FP32 CPU reference for #307."""
import importlib.metadata
import platform
import sys
from pathlib import Path

from compare import require

REVISIONS = {
    "embedding": ("cl-nagoya/ruri-v3-310m", "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3"),
    "reranker": ("cl-nagoya/ruri-v3-reranker-310m", "bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3"),
}
PINS = {"torch":"2.8.0", "transformers":"4.56.2", "sentence-transformers":"5.1.1"}


def environment():
    require(sys.prefix != sys.base_prefix, "use an isolated Python environment")
    pins = dict(line.split("==", 1) for line in Path(__file__).with_name("requirements.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    require(all(pins[name] == version for name,version in PINS.items()), "reference library pins changed")
    for name, version in pins.items():
        require(importlib.metadata.version(name) == version, f"install pinned {name}=={version}")
    require(platform.python_version() == "3.12.13", "use Python 3.12.13")
    return {d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}


def capture(kind, snapshot, observed, inputs):
    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder

    environment()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    kwargs = {"torch_dtype":torch.float32, "attn_implementation":"eager"}
    if kind == "embedding":
        wrapper = SentenceTransformer(str(snapshot), device="cpu", model_kwargs=kwargs,
                                      config_kwargs={"reference_compile":False},
                                      local_files_only=True, trust_remote_code=False)
        model = wrapper[0].auto_model
        require(wrapper.max_seq_length == 8192, "official wrapper input limit changed")
        require(len(wrapper) == 2 and wrapper[1].pooling_mode_mean_tokens and
                not wrapper[1].pooling_mode_cls_token and wrapper[1].include_prompt,
                "inspect fixed official pooling configuration")
    else:
        wrapper = CrossEncoder(str(snapshot), device="cpu", max_length=8192,
                               model_kwargs=kwargs, config_kwargs={"reference_compile":False},
                                      local_files_only=True, trust_remote_code=False)
        model = wrapper.model
    model.float().eval()
    require(not model.training and all(p.dtype == torch.float32 for p in model.parameters()), "FP32/eval required")
    require(model.config._attn_implementation == "eager", "attention implementation changed")
    require(model.config.reference_compile is False, "reference compilation must remain disabled")
    tokenizer = wrapper.tokenizer
    tokenization, native = [], []
    with torch.inference_mode():
        for case in inputs[kind]:
            text = case["text"] * case["repeat"]
            if kind == "embedding":
                text = case["prefix"] + text
                raw = tokenizer(text, truncation=False)
                encoded = wrapper.tokenize([text])
                # Official wrapper handles strip/truncation/pooling. Normalize
                # explicitly because the model card's encode does not request it.
                value = wrapper.encode([text], batch_size=1, normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False)[0].tolist()
            else:
                raw = tokenizer(case["query"],text,truncation=False)
                encoded = tokenizer([(case["query"], text)], padding=True,
                                    truncation=True, return_tensors="pt")
                logit = []

                class RecordingSigmoid(torch.nn.Module):
                    def forward(self, logits):
                        # CrossEncoder replaces a registered activation child module.
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
                "dropout":"disabled by eval; configured probabilities recorded in model manifest",
                "threads":4,"deterministic_algorithms":True,"seed":0,
                "pooling":"official mean, include prompt, explicit L2" if kind == "embedding" else "official CLS/head/classifier; sigmoid",
                "same_tokens":"rurico input IDs and masks replayed without tokenizer or wrapper truncation"}}
