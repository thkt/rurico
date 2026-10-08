#!/usr/bin/env python3
"""Compile identical source probes with already-built locked Rust dependencies.
No Cargo configuration, dependency graph, server, model or GPU is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--deps', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True, help='new directory outside checkout')
parser.add_argument('--baseline-dir', type=Path, help='saved source tree for a prior evaluated artifact')
parser.add_argument('--baseline-id', help='identity of that artifact (not a Git commit)')
parser.add_argument('--probe', type=Path, default=Path(__file__).with_name('probe.rs'), help='probe source; use repair-4/probe.rs for the latest normalization/query comparison')
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
args.deps = args.deps.resolve()
args.out = args.out.resolve()
if bool(args.baseline_dir) != bool(args.baseline_id):
    parser.error('--baseline-dir and --baseline-id must be supplied together')
if args.out == repo or repo in args.out.parents:
    parser.error('output must be outside checkout')
args.out.mkdir(parents=True, exist_ok=False)
base = '24725a72be44300afc24186b82b14bcb3f5f9d3d'
sha = lambda data: hashlib.sha256(data).hexdigest()
def baseline_bytes(source):
    if args.baseline_dir:
        return (args.baseline_dir / source).read_bytes()
    return subprocess.check_output(['git', 'show', f'{base}:{source}'], cwd=repo)
assert sha(baseline_bytes('Cargo.lock')) == sha((repo / 'Cargo.lock').read_bytes()), 'baseline/current Cargo.lock differ'
manifest = {'baseline': base, 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), 'cargo_lock_sha256': sha((repo / 'Cargo.lock').read_bytes()), 'rustc': subprocess.check_output(['rustc', '--version'], text=True).strip(), 'sources': {}, 'dependencies': {}}
if args.baseline_dir:
    manifest['baseline'] = args.baseline_id
    manifest['baseline_kind'] = 'evaluated-artifact'
for version in ['baseline', 'current']:
    dest = args.out / version
    dest.mkdir()
    (args.out / f'{version}.rs').write_text('pub mod query_normalize; pub mod search;\n')
    for name in ['query_normalize', 'search']:
        for suffix in [f'{name}.rs', f'{name}/tests.rs']:
            source = f'src/storage/{suffix}'
            data = baseline_bytes(source) if version == 'baseline' else (repo / source).read_bytes()
            target = dest / suffix
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(data)
            manifest['sources'][f'{version}/{source}'] = sha(data)
shutil.copyfile(args.probe, args.out / 'probe.rs')
manifest['probe_sha256'] = sha((args.out / 'probe.rs').read_bytes())
flags = ['rustc', '--edition=2024', '-O', '-C', 'lto', '-L', f'dependency={args.deps}']
for name in ['rusqlite', 'serde', 'serde_json', 'unicode_normalization', 'thiserror']:
    matches = list(args.deps.glob(f'lib{name}-*.rlib'))
    if len(matches) != 1:
        raise RuntimeError(f'expected one {name} rlib, got {len(matches)}')
    flags += ['--extern', f'{name}={matches[0]}']
    manifest['dependencies'][matches[0].name] = sha(matches[0].read_bytes())
manifest['flags'] = ['--edition=2024', '-O', '-C lto', 'same prebuilt dependency rlibs']
subprocess.run(flags + [str(args.out / 'probe.rs'), '-o', str(args.out / 'probe')], check=True)
manifest['binary_sha256'] = sha((args.out / 'probe').read_bytes())
with (args.out / 'raw.jsonl').open('w') as output:
    subprocess.run([str(args.out / 'probe')], stdout=output, check=True)
# Reuse all relevant existing module tests without MLX initialization.
(args.out / 'tests.rs').write_text('mod current;\n')
test_dependencies = list(args.deps.glob('libtempfile-*.rlib'))
if len(test_dependencies) != 1:
    raise RuntimeError(f'expected one tempfile rlib, got {len(test_dependencies)}')
tempfile = test_dependencies[0]
manifest['test_dependencies'] = {tempfile.name: sha(tempfile.read_bytes())}
subprocess.run(flags + ['--extern', f'tempfile={tempfile}', '--test', str(args.out / 'tests.rs'), '-o', str(args.out / 'tests')], check=True)
with (args.out / 'tests.txt').open('w') as output:
    subprocess.run([str(args.out / 'tests')], stdout=output, check=True)
for source, digest in manifest['sources'].items():
    if source.startswith('current/'):
        assert sha((repo / source.removeprefix('current/')).read_bytes()) == digest
assert sha((repo / 'Cargo.lock').read_bytes()) == manifest['cargo_lock_sha256']
(args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
# Generate the display from raw JSONL; do not hand-edit the measured values.
rows = [json.loads(line) for line in (args.out / 'raw.jsonl').read_text().splitlines()]
table = ['| 対象・入力 | 変更前 ns/call 中央値 [最小,最大] | 変更後 ns/call 中央値 [最小,最大] | Rust allocation/reallocation回数 前→後 |', '| --- | ---: | ---: | ---: |']
for kind in ['normalization', 'query']:
    for case in dict.fromkeys(row['case'] for row in rows if row['kind'] == kind):
        values, allocations = [], []
        for version in [0, 1]:
            samples = [row for row in rows if row['kind'] == kind and row['case'] == case and row['version'] == version]
            times = [row['ns_per_call'] for row in samples]
            assert len(times) == 7
            counts = {row['rust_allocations'] for row in samples}
            assert len(counts) == 1
            allocations.append(counts.pop())
            values.append(f'{statistics.median(times):.2f} [{min(times):.2f}, {max(times):.2f}]')
        table.append(f'| {kind}/{case} | {values[0]} | {values[1]} | {allocations[0]}→{allocations[1]} |')
(args.out / 'table.md').write_text('\n'.join(table) + '\n')
print(args.out)
