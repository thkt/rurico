#!/usr/bin/env python3
"""Snapshot actual product preprocessing functions; compile without MLX.
Use already-built locked dependencies, never change Cargo features/lock.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

BASE = 'fb62091fcf34b01bce080ea7d4ec588d4aa6a988'
REPO = Path(__file__).resolve().parents[3]
SOURCES = ['src/embed.rs', 'src/embed/processing.rs', 'src/embed/metrics.rs',
           'src/model_io.rs', 'src/embed/mlx.rs', 'Cargo.lock']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def section(source, start, end):
    return source[source.index(start):source.index(end, source.index(start))]


def prepare(out, version, data):
    dest = out / version
    dest.mkdir()
    for name, content in data.items():
        path = dest / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    model = data['src/model_io.rs'].decode()
    # Only exclude model loading and other unrelated MLX/HF dependencies.
    io = '\n'.join([
        re.search(r'pub const MAX_SEQ_LEN: usize = .*?;', model)[0],
        re.search(r'pub\(crate\) const BUCKET_BOUNDS: [^\n]+', model)[0],
        re.search(r'pub\(crate\) const TOKEN_BUDGET: .*?;', model)[0],
        section(model, 'pub(crate) fn assign_bucket', '/// Truncate'),
        section(model, 'pub(crate) fn compute_sub_batch_size', '/// Pad variable'),
        section(model, 'pub(crate) fn pad_sequences', '#[cfg(test)]'),
    ])
    (dest / 'model_io.rs').write_text(io)
    embed = data['src/embed.rs'].decode()
    constants = '\n'.join(re.search(pattern, embed)[0] for pattern in [
        r'pub const DOCUMENT_PREFIX: .*?;',
        r'pub\(crate\) const CHUNK_OVERLAP_TOKENS: .*?;',
    ])
    tokens = section(embed, 'pub struct TokenizedInput', '#[cfg(all(test, feature = "smoke"))]')
    # Stubs are only forward/readback types: the benchmark never executes them.
    # Tokenization, planning, shrinking, indexing, distribution and padding below
    # are copied verbatim from the identified product source, not reimplemented.
    wrapper = '''
#![allow(dead_code)]
use super::model_io::MAX_SEQ_LEN;
#[derive(Debug)]
pub enum EmbedError { Inference { message: String }, BufferShapeMismatch { expected: usize, actual: usize }, NonFiniteOutput }
impl EmbedError {
    fn tokenizer(e: impl std::fmt::Display) -> Self { Self::inference_message(e.to_string()) }
    fn inference_message(message: String) -> Self { Self::Inference { message } }
}
#[derive(Debug)]
pub struct ChunkedEmbedding;
impl ChunkedEmbedding { fn try_new(_: Vec<Vec<f32>>) -> Result<Self, EmbedError> { Ok(Self) } }
fn first_non_finite(x: &[f32]) -> Option<usize> { x.iter().position(|x| !x.is_finite()) }
pub struct EmbedOptions { pub token_budget: Option<usize>, pub forward_pause: Option<std::time::Duration> }
#[path="source/src/embed/metrics.rs"] mod metrics;
#[path="source/src/embed/processing.rs"] mod processing;
use processing::{build_indexed_chunks, distribute_into_buckets, plan_document_chunks, shrink_chunk_to_fit};
pub fn plan(t: &tokenizers::Tokenizer, text: &str, budget: usize, prefix: &[u32]) -> (Vec<Vec<u32>>, Vec<usize>) {
    plan_document_chunks(t, &[text], prefix, budget).unwrap()
}
pub fn shrink(t: &tokenizers::Tokenizer, text: &str, offsets: &[(usize, usize)], mut end: usize) -> (Vec<u32>, usize) {
    let ids = shrink_chunk_to_fit(t, text, offsets, 0, &mut end).unwrap(); (ids, end)
}
pub fn route(tokens: Vec<Vec<u32>>, sorted: bool) -> Vec<usize> {
    let counts = [tokens.len()];
    let mut buckets = distribute_into_buckets(build_indexed_chunks(tokens, &counts).unwrap());
    if sorted { for b in &mut buckets { b.sort_by_key(|c| c.global_idx); } }
    buckets.iter().flatten().map(|c| c.global_idx).collect()
}
pub struct PadInput(Vec<processing::IndexedChunk>);
pub fn pad_input(tokens: Vec<Vec<u32>>) -> PadInput {
    let counts = [tokens.len()];
    PadInput(build_indexed_chunks(tokens, &counts).unwrap())
}
pub fn pad(input: &PadInput) -> (Vec<u32>, Vec<u32>, usize, usize) {
    let chunks = &input.0;
    PAD_BODY
}
'''
    if version == 'baseline':
        pad = '''let copied: Vec<Vec<u32>> = chunks.iter().map(|c| c.tokens.clone()).collect();
    super::model_io::pad_sequences(&copied, None, Some(512))'''
    else:
        pad = 'super::model_io::pad_sequences(&chunks, None, Some(512))'
    # Match the actual forward adapter; include source hash in the manifest.
    adapter = section(data['src/embed/mlx.rs'].decode(), 'fn forward_sub_batch(', 'metrics.real_tokens +=')
    if version == 'baseline':
        assert re.search(r'let sub_tokens: Vec<Vec<u32>> = sub_batch\.iter\(\)\.map\(\|c\| c\.tokens\.clone\(\)\)\.collect\(\);', adapter)
        assert 'pad_sequences(&sub_tokens, None, Some(BUCKET_BOUNDS[bucket_idx]))' in ' '.join(adapter.split())
    else:
        assert 'pad_sequences(sub_batch, None, Some(BUCKET_BOUNDS[bucket_idx]))' in ' '.join(adapter.split())
    (dest / 'embed.rs').write_text(wrapper.replace('PAD_BODY', pad) + constants + '\n' + tokens)
    (out / f'{version}.rs').write_text('pub mod model_io; pub mod embed;\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deps', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rlib', action='append', default=[], help='name=path to disambiguate an existing dependency rlib')
    parser.add_argument('--prepare-only', action='store_true', help='snapshot and compile without running timed trials')
    parser.add_argument('--tokenizer', type=Path, help='fixed ruri-v3-310m tokenizer.json; omit for synthetic WordLevel')
    args = parser.parse_args()
    overrides = dict(item.split('=', 1) for item in args.rlib)
    if set(overrides) - {'tokenizers', 'serde', 'serde_json', 'tracing', 'libc'}:
        parser.error('unknown rlib override')
    out = args.out.resolve()
    if out == REPO or REPO in out.parents:
        parser.error('output must be outside checkout')
    out.mkdir(parents=True, exist_ok=False)
    data = {
        'baseline': {s: subprocess.check_output(['git', 'show', f'{BASE}:{s}'], cwd=REPO) for s in SOURCES},
        'current': {s: (REPO / s).read_bytes() for s in SOURCES},
    }
    assert data['baseline']['Cargo.lock'] == data['current']['Cargo.lock']
    manifest = {'baseline': BASE, 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                'sources': {v: {s: sha(b) for s, b in files.items()} for v, files in data.items()},
                'rustc': subprocess.check_output(['rustc', '--version'], text=True).strip(),
                'probe_sha256': sha(Path(__file__).with_name('probe.rs').read_bytes()),
                'run_sha256': sha(Path(__file__).read_bytes()), 'dependencies': {}}
    assets = ['src/embed/processing/tests.rs', 'src/model_io/tests.rs', 'CONTRIBUTING.md', 'docs/benchmarks/issue-311/README.md']
    manifest['verification_assets'] = {p: sha((REPO / p).read_bytes()) for p in assets}
    if args.tokenizer:
        manifest['tokenizer_sha256'] = sha(args.tokenizer.read_bytes())
    for v in data:
        prepare(out, v, data[v])
    # processing uses crate::model_io; identical padding implementation except
    # the borrowed generic parameter, used by both variants in this harness.
    (out / 'probe.rs').write_bytes(Path(__file__).with_name('probe.rs').read_bytes())
    flags = ['rustc', '--edition=2024', '-O', '-L', f'dependency={args.deps.resolve()}']
    for name in ['tokenizers', 'serde', 'serde_json', 'tracing', 'libc']:
        matches = [Path(overrides[name])] if name in overrides else list(args.deps.glob(f'lib{name}-*.rlib'))
        if any(p.resolve().parent != args.deps.resolve() or not p.name.startswith(f'lib{name}-') or p.suffix != '.rlib' for p in matches):
            raise RuntimeError('rlib overrides must identify existing files in --deps')
        if len(matches) != 1:
            raise RuntimeError(f'expected one {name} rlib, got {len(matches)}; specify an unambiguous dependency directory')
        flags += ['--extern', f'{name}={matches[0].resolve()}']
        manifest['dependencies'][matches[0].name] = sha(matches[0].read_bytes())
    manifest['flags'] = flags[:3]
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    subprocess.run(flags + [str(out / 'probe.rs'), '-o', str(out / 'probe')], check=True)
    manifest['binary_sha256'] = sha((out / 'probe').read_bytes())
    if args.prepare_only:
        manifest['complete'] = False
        manifest['prepared_only'] = True
        manifest['command'] = [str(out / 'probe')] + ([str(args.tokenizer.resolve())] if args.tokenizer else [])
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(out)
        return
    cmd = [str(out / 'probe')]
    if args.tokenizer:
        cmd.append(str(args.tokenizer.resolve()))
    with (out / 'raw.jsonl').open('w') as raw:
        subprocess.run(cmd, stdout=raw, env={**os.environ, 'TOKENIZERS_PARALLELISM': 'false'}, check=True)
    for source, content in data['current'].items():
        assert (REPO / source).read_bytes() == content, f'source changed: {source}'
    if args.tokenizer:
        assert sha(args.tokenizer.read_bytes()) == manifest['tokenizer_sha256']
    for path, digest in manifest['verification_assets'].items():
        assert sha((REPO / path).read_bytes()) == digest, f'verification asset changed: {path}'
    rows = [json.loads(line) for line in (out / 'raw.jsonl').read_text().splitlines()]
    summary = []
    for case, variant in dict.fromkeys((r['case'], r['variant']) for r in rows):
        samples = [r for r in rows if r['case'] == case and r['variant'] == variant]
        assert len(samples) == 7
        result = {'case': case, 'variant': variant}
        for key in ['wall_ns_per_call', 'cpu_ns_per_call', 'rust_allocations_reallocations', 'rust_peak_extra_requested_bytes']:
            values = [r[key] for r in samples]
            result[key] = {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
        summary.append(result)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest['complete'] = True
    manifest['raw_sha256'] = sha((out / 'raw.jsonl').read_bytes())
    manifest['summary_sha256'] = sha((out / 'summary.json').read_bytes())
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
