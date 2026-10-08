import sqlite3
import unittest

from plan import Expansion, Literal, Phrase, Plan, expand


class PlanTests(unittest.TestCase):
    def test_legacy_adapter_keeps_literal_contents_and_rejects_phrase(self):
        conn = sqlite3.connect(':memory:')
        self.addCleanup(conn.close)
        conn.execute('CREATE VIRTUAL TABLE docs USING fts5(body)')
        conn.execute('INSERT INTO docs VALUES (?)', ('foo OR bar rate-limit',))
        plan = Plan((Literal('foo'), Literal('OR'), Literal('bar'),
                     Expansion('ra', ('rate-limit', 'rare'), False, None)))
        wire = plan.legacy_wire()
        self.assertEqual(wire, '"foo" AND "OR" AND "bar" AND ("rate-limit" OR "rare")')
        self.assertEqual(conn.execute('SELECT rowid FROM docs WHERE docs MATCH ?',
                                      (wire,)).fetchall(), [(1,)])
        phrase = Plan((Phrase('foo "bar"'),))
        self.assertEqual(phrase.match_sql(), '"foo ""bar"""')
        with self.assertRaisesRegex(ValueError, 'phrase requires'):
            phrase.legacy_wire()

    def test_lookahead_reports_cap_and_stable_tie_order(self):
        conn = sqlite3.connect(':memory:')
        self.addCleanup(conn.close)
        # Ordinary table deliberately exposes a nonlexical tie order. An
        # fts5vocab scan happens to be lexical on the measured SQLite version.
        conn.execute('CREATE TABLE vocab(term TEXT, cnt INTEGER)')
        terms = [f'日{chr(0x4e00+i)}語' for i in range(26)]
        conn.executemany('INSERT INTO vocab VALUES (?, 1)', [(t,) for t in reversed(terms[:25])])
        before = expand(conn, '日', deterministic=True)
        self.assertEqual(before.terms, tuple(terms[:25]))
        self.assertFalse(before.truncated)
        conn.execute('INSERT INTO vocab VALUES (?, 1)', (terms[25],))
        after = expand(conn, '日', deterministic=True)
        self.assertEqual(after.terms, tuple(terms[:25]))
        self.assertTrue(after.truncated)
        self.assertEqual(expand(conn, '無', deterministic=True).fallback, 'no-candidates')
        self.assertEqual(expand(conn, '日', missing=True).fallback, 'missing-vocab')
        conn.execute('DROP TABLE vocab')
        conn.execute('CREATE TABLE vocab(term TEXT)')
        with self.assertRaisesRegex(sqlite3.OperationalError, 'no such column: cnt'):
            expand(conn, '日', deterministic=True)


if __name__ == '__main__':
    unittest.main()
