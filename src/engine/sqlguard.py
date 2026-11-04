"""Guard for the public SQL box: one read-only statement, no file or network access, no state changes.

This is defence in depth. The connection also runs with external access disabled and a memory limit
(see `lock_down`), so even a statement that slips past the text checks cannot read files or write to disk.
"""
from __future__ import annotations

import re

ALLOWED_START = ("select", "with", "explain", "describe", "show", "summarize", "from")
FORBIDDEN = ("insert", "update", "delete", "drop", "create", "alter", "attach", "detach", "copy", "export", "import", "install",
             "load", "pragma", "call", "set", "reset", "use", "vacuum", "checkpoint", "truncate", "replace", "merge", "grant",
             "read_csv", "read_parquet", "read_json", "read_text", "read_blob", "glob", "parquet_scan", "csv_scan", "sniff_csv",
             "httpfs", "getenv", "current_setting")
_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)
_STRING = re.compile(r"'(?:[^']|'')*'")


def validate_readonly(sql: str, max_chars: int = 5000) -> str:
    """Return the cleaned single statement or raise ValueError with a message for the user."""
    if not sql or not sql.strip():
        raise ValueError("write a query first")
    if len(sql) > max_chars:
        raise ValueError(f"query is longer than {max_chars} characters")
    clean = _COMMENT.sub(" ", sql).strip().rstrip(";").strip()
    if ";" in _STRING.sub("''", clean):
        raise ValueError("only one statement at a time")
    bare = _STRING.sub("''", clean).lower()
    words = re.findall(r"[a-z_]+", bare)
    if not words or words[0] not in ALLOWED_START:
        raise ValueError("only read-only queries are allowed: start with SELECT, WITH, EXPLAIN, DESCRIBE or SHOW")
    bad = sorted(set(words) & set(FORBIDDEN))
    if bad:
        raise ValueError(f"not allowed in this box: {', '.join(bad)}")
    return clean


def lock_down(con, memory_limit: str = "1GB", threads: int = 2) -> None:
    """Disable file/network access and cap resources. Call once after the tables are loaded."""
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute("SET enable_external_access=false")
    con.execute("SET lock_configuration=true")
