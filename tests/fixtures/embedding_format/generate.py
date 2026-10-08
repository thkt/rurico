"""Reproduce public synthetic fixtures without model, network or private input."""
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent


def generate():
    # One document, one chunk, two finite f32 values.
    legacy = struct.pack('<IIIff', 1, 1, 2, 1.0, 0.0)
    metadata = {
        'producer': 'synthetic',
        'model': 'none: synthetic vectors',
        'model_revision': 'not applicable',
        'tokenizer': 'none: no tokenization',
        'inputs': ['public synthetic unit vector'],
        'generation_code': 'generate.py sha256:' + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'settings': 'one document, one chunk, two dimensions; no inference',
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()
    return {
        'legacy.bin': legacy,
        'v1.bin': struct.pack('<III', 0xffffffff, 1, len(encoded)) + encoded + legacy,
        'hostile-dim.bin': struct.pack('<III', 1, 1, 0xffffffff),
        'nonfinite.bin': struct.pack('<IIIf', 1, 1, 1, float('inf')),
        'trailing.bin': legacy + b'\x00',
    }


if __name__ == '__main__':
    for name, data in generate().items():
        (ROOT / name).write_bytes(data)
