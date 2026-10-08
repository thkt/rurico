"""Real SQLite MATCH experiments over public synthetic cases; no inference."""
import argparse
import hashlib
import json
import platform
import sqlite3
import time
from pathlib import Path

from plan import Expansion, Literal, Phrase, Plan, expand

HERE = Path(__file__).resolve().parent


def hits(conn, wire):
    return [row[0] for row in conn.execute(
        'SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY rowid', (wire,))]


def quality(actual, relevant):
    actual, relevant = set(actual), set(relevant)
    return {'true_positive': len(actual & relevant),
            'false_positive': len(actual - relevant),
            'false_negative': len(relevant - actual),
            'precision': len(actual & relevant) / len(actual) if actual else None,
            'recall': len(actual & relevant) / len(relevant) if relevant else None}


def measure():
    cases = json.loads((HERE / 'cases.json').read_text())
    results = []
    for tokenizer in ('unicode61', 'trigram'):
        for case in cases:
            conn = sqlite3.connect(':memory:')
            conn.execute(f"CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='{tokenizer}')")
            conn.executemany('INSERT INTO docs VALUES (?)', [(s,) for s in case['bodies']])
            conn.execute('CREATE VIRTUAL TABLE vocab USING fts5vocab(docs, row)')
            for mode in ('legacy', 'deterministic', 'missing-vocab', 'explicit-phrase'):
                if mode == 'explicit-phrase':
                    if 'phrase' not in case:
                        continue
                    plan = Plan((Phrase(case['phrase']),))
                else:
                    plan = Plan(tuple(expand(conn, term, deterministic=mode == 'deterministic',
                                             missing=mode == 'missing-vocab')
                                      for term in case['legacy_terms']))
                wire = plan.match_sql()
                actual = hits(conn, wire)
                results.append(dict(case=case['name'], tokenizer=tokenizer, mode=mode,
                                    wire=wire, hits=actual, quality=quality(actual, case['relevant']),
                                    expansions=[dict(source=n.source, selected=list(n.terms),
                                                     truncated=n.truncated if mode != 'legacy' else None,
                                                     fallback=n.fallback)
                                                for n in plan.nodes if isinstance(n, Expansion)]))
            conn.close()
    # Intent-specific expectations are independent of serializer/expansion logic.
    for r in results:
        if r['case'] == 'phrase':
            expected = [1] if r['mode'] == 'explicit-phrase' else (
                [1, 2] if r['tokenizer'] == 'unicode61' else [])
            assert r['hits'] == expected, r
        if r['case'] == 'cap-ties' and r['mode'] in ('legacy', 'deterministic'):
            assert len(r['hits']) == 50 and 61 not in r['hits'], r
            assert r['quality']['false_negative'] == 11, r
            if r['mode'] == 'deterministic':
                assert r['expansions'][0]['truncated'], r
        if r['case'] == 'operator':
            assert r['wire'] == '"foo" AND "OR" AND "bar"', r
            assert r['hits'] == ([1] if r['tokenizer'] == 'unicode61' else []), r
    return results


def ngram_comparison():
    # Only considered because short-query loss has been demonstrated above.
    bodies = ['日本語', '日本海', '休日', '日', '本日', '日と本'] * 256
    queries = ['日', '日本', '休日']
    measurements = []
    for n in (3, 2, 1):
        conn = sqlite3.connect(':memory:')
        tokenizer = 'trigram' if n == 3 else 'unicode61'
        conn.execute(f"CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='{tokenizer}')")
        conn.execute('CREATE VIRTUAL TABLE vocab USING fts5vocab(docs, row)')
        before = conn.execute('PRAGMA page_count').fetchone()[0]
        texts = bodies if n == 3 else [
            ' '.join(s[i:i+n] for i in range(len(s)-n+1)) for s in bodies]
        start = time.perf_counter()
        conn.executemany('INSERT INTO docs VALUES (?)', [(s,) for s in texts])
        conn.commit()
        insert_seconds = time.perf_counter() - start
        # A content UPDATE traverses SQLite's delete/add FTS path, even with
        # the same value. This is a controlled write-cost sample, not latency SLA.
        start = time.perf_counter()
        conn.executemany('UPDATE docs SET body=? WHERE rowid=?',
                         [(s, i+1) for i, s in enumerate(texts)])
        conn.commit()
        update_seconds = time.perf_counter() - start
        conn.execute("INSERT INTO docs(docs) VALUES ('optimize')")
        conn.commit()
        after = conn.execute('PRAGMA page_count').fetchone()[0]
        evaluations = []
        for q in queries:
            wire = Plan((Literal(q),)).match_sql() if n == 3 else Plan(tuple(
                Literal(q[i:i+n]) for i in range(len(q)-n+1))).match_sql() if len(q) >= n else None
            actual = hits(conn, wire) if wire else []
            relevant = [i+1 for i, s in enumerate(bodies) if q in s]
            evaluations.append(dict(query=q, wire=wire, quality=quality(actual, relevant)))
        measurements.append(dict(ngram=n, implementation='builtin-trigram' if n == 3 else 'pretokenized-unicode61-example',
                                 documents=len(bodies), page_size=conn.execute('PRAGMA page_size').fetchone()[0],
                                 pages_before=before, pages_after=after, insert_seconds=insert_seconds,
                                 update_seconds=update_seconds,
                                 token_occurrences=conn.execute('SELECT COALESCE(SUM(cnt), 0) FROM vocab').fetchone()[0],
                                 evaluations=evaluations))
        conn.close()
    return measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = HERE.parents[2]
    paths = ['Cargo.lock', 'src/storage/search.rs', 'src/storage/query_normalize.rs',
             'docs/research/issue-314/cases.json', 'docs/research/issue-314/plan.py',
             'docs/research/issue-314/run.py']
    result = dict(schema=1, python=platform.python_version(), sqlite=sqlite3.sqlite_version,
                  platform=platform.platform(), normalization='all-off on both sides',
                  source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
                  cases=measure(), ngrams=ngram_comparison())
    # Never overwrite historical evidence.
    with args.output.open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
