"""Run each unchanged official-reference observation in a fresh process."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from compare import require


def write(path, value):
    with path.open('x') as file:
        json.dump(value, file, ensure_ascii=False, allow_nan=False)


def capture_isolated(kind, snapshot, observed, inputs, out):
    directory = out / f'reference-{kind}-units'
    directory.mkdir()
    result = {'schema': observed['schema'], 'producer': 'transformers', 'kind': kind,
              'tokenization': [], 'native': [], 'batches': []}
    units = [('native', case['id'], case, None) for case in inputs[kind]]
    units.extend(('batch', batch['id'], None, batch) for batch in observed['batches'])
    require(bool(units), 'no reference observations')
    with (out / 'reference-events.jsonl').open('a') as events:
        for index, (phase, identifier, case, batch) in enumerate(units):
            prefix = directory / str(index)
            request = prefix.with_suffix('.request.private.json')
            output = prefix.with_suffix('.json')
            shape = [len(batch['ids']), len(batch['ids'][0])] if batch else None
            descriptor = {'kind': kind, 'phase': phase, 'id': identifier, 'shape': shape,
                          'index': index, 'dtype': 'float32'}
            def event(status, **details):
                events.write(json.dumps({**descriptor, 'status': status,
                    'utc': datetime.now(timezone.utc).isoformat(), **details}) + '\n')
                events.flush()
            write(request, {'kind': kind, 'snapshot': str(snapshot),
                            'observed': {'schema': observed['schema'], 'batches': [batch] if batch else []},
                            'inputs': {kind: [case] if case else []},
                            'events': str(out / 'reference-events.jsonl'), 'descriptor': descriptor})
            event('started')
            started = time.monotonic()
            try:
                with prefix.with_suffix('.private.log').open('x') as log:
                    process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                              str(request), str(output)], stdout=log, stderr=log, check=True)
                payload = json.loads(output.read_text())
                part = payload['result']
                require(part['schema'] == result['schema'] and part['kind'] == kind
                        and part['producer'] == 'transformers', 'reference unit identity differs')
                expected_native = [identifier] if phase == 'native' else []
                expected_batches = [identifier] if phase == 'batch' else []
                require([x['id'] for x in part['native']] == expected_native
                        and [x['id'] for x in part['tokenization']] == expected_native
                        and [x['id'] for x in part['batches']] == expected_batches,
                        'reference unit observations missing or reordered')
                if 'conditions' in result:
                    require(result['conditions'] == part['conditions'], 'reference unit conditions differ')
                else:
                    result['conditions'] = part['conditions']
                for key in ('tokenization', 'native', 'batches'):
                    result[key].extend(part[key])
                actual_shape = shape if batch else [1, len(part['tokenization'][0]['model_ids'])]
                event('complete', shape=actual_shape, pid=payload['pid'], exit_code=process.returncode,
                      elapsed_seconds=time.monotonic()-started)
            except BaseException as error:
                event('incomplete', error_type=type(error).__name__, elapsed_seconds=time.monotonic()-started)
                raise
    return result


def main():
    from reference import capture
    request = json.loads(Path(sys.argv[1]).read_text())
    def observe(identifier, shape):
        require(identifier == request['descriptor']['id'], 'native input identity differs')
        with Path(request['events']).open('a') as events:
            events.write(json.dumps({**request['descriptor'], 'shape': shape,
                'status': 'inference', 'pid': os.getpid(),
                'utc': datetime.now(timezone.utc).isoformat()}) + '\n')
    result = capture(request['kind'], Path(request['snapshot']), request['observed'], request['inputs'], observe)
    write(Path(sys.argv[2]), {'pid': os.getpid(), 'result': result})


if __name__ == '__main__':
    main()
