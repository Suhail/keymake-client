"""Per-agent SQLite store for connections and trust tiers."""

import os
import sqlite3
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS connections (
    agent TEXT NOT NULL PRIMARY KEY,
    status TEXT NOT NULL CHECK(status IN ('pending','accepted','rejected')),
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS trust (
    agent TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('inbound','outbound')),
    status TEXT NOT NULL CHECK(status IN ('pending','accepted','rejected')),
    created_at REAL NOT NULL,
    PRIMARY KEY (agent, direction)
);
"""

class AgentStore:
    def __init__(self, agent_name: str, data_dir: str = "data"):
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, f"{agent_name}.db")
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)

    # -- connection helpers --

    def add_connection(self, agent: str, status: str = "pending"):
        self._conn.execute(
            "INSERT OR REPLACE INTO connections (agent, status, created_at) VALUES (?,?,?)",
            (agent, status, time.time()),
        )
        self._conn.commit()

    def update_connection(self, agent: str, status: str):
        self._conn.execute(
            "UPDATE connections SET status=? WHERE agent=?",
            (status, agent),
        )
        self._conn.commit()

    def is_connected(self, agent: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM connections WHERE agent=? AND status='accepted'",
            (agent,),
        ).fetchone()
        return row is not None

    def get_connections(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT agent FROM connections WHERE status='accepted'"
        ).fetchall()
        return [r[0] for r in rows]

    def remove_connection(self, agent: str):
        self._conn.execute("DELETE FROM connections WHERE agent=?", (agent,))
        self._conn.commit()

    # -- trust helpers --

    def add_trust(self, agent: str, direction: str, status: str = "pending"):
        self._conn.execute(
            "INSERT OR REPLACE INTO trust (agent, direction, status, created_at) VALUES (?,?,?,?)",
            (agent, direction, status, time.time()),
        )
        self._conn.commit()

    def update_trust(self, agent: str, direction: str, status: str):
        self._conn.execute(
            "UPDATE trust SET status=? WHERE agent=? AND direction=?",
            (status, agent, direction),
        )
        self._conn.commit()

    def remove_trust(self, agent: str):
        self._conn.execute("DELETE FROM trust WHERE agent=?", (agent,))
        self._conn.commit()

    def is_trusted_by(self, agent: str) -> bool:
        """Does `agent` trust me? (inbound accepted trust = they granted me access.)"""
        row = self._conn.execute(
            "SELECT 1 FROM trust WHERE agent=? AND direction='inbound' AND status='accepted'",
            (agent,),
        ).fetchone()
        return row is not None

    def trusts(self, agent: str) -> bool:
        """Have I granted trust to `agent`? (outbound accepted.)"""
        row = self._conn.execute(
            "SELECT 1 FROM trust WHERE agent=? AND direction='outbound' AND status='accepted'",
            (agent,),
        ).fetchone()
        return row is not None

    def get_trusted_by(self) -> list[str]:
        """Agents that trust me (I can invoke their capabilities)."""
        rows = self._conn.execute(
            "SELECT agent FROM trust WHERE direction='inbound' AND status='accepted'"
        ).fetchall()
        return [r[0] for r in rows]

    def get_trusting(self) -> list[str]:
        """Agents I have granted trust to (they can invoke my capabilities)."""
        rows = self._conn.execute(
            "SELECT agent FROM trust WHERE direction='outbound' AND status='accepted'"
        ).fetchall()
        return [r[0] for r in rows]

    def close(self):
        self._conn.close()
