"""Unadopted query-plan example. No user query language or legacy parser."""
from dataclasses import dataclass


def quote(text):
    return '"' + text.replace('"', '""') + '"'


@dataclass(frozen=True)
class Literal:
    text: str


@dataclass(frozen=True)
class Phrase:
    text: str


@dataclass(frozen=True)
class Expansion:
    source: str
    terms: tuple[str, ...]
    truncated: bool
    fallback: str | None  # missing-vocab / no-candidates; other DB errors propagate


@dataclass(frozen=True)
class Plan:
    nodes: tuple[Literal | Phrase | Expansion, ...]

    def match_sql(self):
        if not self.nodes:
            raise ValueError('empty plan')
        parts = []
        for node in self.nodes:
            if isinstance(node, (Literal, Phrase)):
                # Type carries intent; both use SQLite's quoted string grammar.
                parts.append(quote(node.text))
            elif node.terms:
                parts.append('(' + ' OR '.join(map(quote, node.terms)) + ')')
            else:
                parts.append(quote(node.source))
        return ' AND '.join(parts)

    def legacy_wire(self):
        # The old wire cannot carry phrase intent or expansion diagnostics.
        # Do not silently downgrade a new node for an unverified consumer.
        if any(isinstance(node, Phrase) for node in self.nodes):
            raise ValueError('phrase requires a coordinated consumer API')
        return self.match_sql()


def expand(conn, text, *, deterministic=False, limit=25, missing=False):
    """Research SQL: current cnt-only vs proposed binary tie key + lookahead.

    Uses a fixed, locally owned vocab identifier. Not a production SQL-name API.
    """
    if len(text) >= 3 or text.upper() in ('AND', 'OR', 'NOT'):
        return Literal(text)
    if missing:
        return Expansion(text, (), False, 'missing-vocab')
    pattern = text.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_') + '%'
    order = 'cnt DESC, term COLLATE BINARY ASC' if deterministic else 'cnt DESC'
    count = limit + 1 if deterministic else limit
    rows = conn.execute(
        f"SELECT term FROM vocab WHERE term LIKE ? ESCAPE '\\' ORDER BY {order} LIMIT ?",
        (pattern, count),
    ).fetchall()
    return Expansion(text, tuple(row[0] for row in rows[:limit]),
                     deterministic and len(rows) > limit,
                     'no-candidates' if not rows else None)
