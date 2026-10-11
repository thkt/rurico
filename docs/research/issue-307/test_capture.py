"""Capture setup checks without model downloads or inference."""
import hashlib
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import run


class ModelManifest(unittest.TestCase):
    def test_each_file_hashed_once_and_verified_metadata_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, output = root/"snapshot", root/"output"
            snapshot.mkdir()
            output.mkdir()
            header = json.dumps({"weight":{"dtype":"F32"}}).encode()
            files = {"model.safetensors":struct.pack("<Q",len(header))+header,
                     "config.json":b'{"local_attention":128,"max_position_embeddings":8192}',
                     "tokenizer.json":b'{}', "tokenizer_config.json":b'{}',
                     "special_tokens_map.json":b'{}', "tokenizer.model":b'tokens',
                     "README.md":b'model card', "1_Pooling/config.json":b'{}'}
            for name,data in files.items():
                (snapshot/name).parent.mkdir(parents=True,exist_ok=True)
                (snapshot/name).write_bytes(data)
            digests = {name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
            git_files = {"config.json", "tokenizer.json", "tokenizer_config.json",
                         "special_tokens_map.json", "README.md"}
            etags = {name:hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
                     if name in git_files else digests[name] for name,data in files.items()}
            hub = types.ModuleType("huggingface_hub")
            hub.snapshot_download = lambda *a,**kw: str(snapshot)
            hub.hf_hub_url = lambda repo,name,revision: name
            hub.get_hf_file_metadata = lambda name: types.SimpleNamespace(commit_hash="pinned",etag=etags[name])
            opened = []
            original_open = Path.open
            def track_open(path, *args, **kwargs):
                if path.is_relative_to(snapshot):
                    opened.append(str(path.relative_to(snapshot)))
                return original_open(path, *args, **kwargs)
            with patch.dict(sys.modules,{"huggingface_hub":hub}), \
                 patch.object(run,"REVISIONS",{"embedding":("model","pinned")}), \
                 patch.object(Path,"open",track_open):
                _,manifest = run.models(output)
            # Content hashing reads once; later header/config inspection stays independent.
            self.assertCountEqual(opened, [*files, "model.safetensors", "config.json"])
            self.assertEqual(manifest["embedding"]["files"], {
                name:{"sha256":digest, **({"hub_etag":etags[name]} if name != "1_Pooling/config.json" else {})}
                for name,digest in digests.items()})
            self.assertEqual(json.loads((output/"models.json").read_text()),manifest)
            for name, reason in (("README.md", "HF blob content mismatch"),
                                 ("model.safetensors", "HF LFS content mismatch")):
                with self.subTest(name=name):
                    correct = etags[name]
                    etags[name] = "0"*len(correct)
                    rejected = root/(name+"-rejected")
                    rejected.mkdir()
                    with patch.dict(sys.modules,{"huggingface_hub":hub}), \
                         patch.object(run,"REVISIONS",{"embedding":("model","pinned")}), \
                         self.assertRaisesRegex(ValueError,reason):
                        run.models(rejected)
                    self.assertFalse((rejected/"models.json").exists())
                    etags[name] = correct


if __name__ == "__main__":
    unittest.main()
