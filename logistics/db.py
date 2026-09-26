"""SQLite: планы по дням и журнал (исходные заявки, результат ИИ, правки логиста, утверждение).

Журнал нужен, чтобы сравнивать качество: что предложила система и сколько логист переделал.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from .config import DATA
from .models import Plan

DB_PATH = DATA / "logistics.db"


def _conn() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.executescript("""
        CREATE TABLE IF NOT EXISTS plans (day TEXT PRIMARY KEY, json TEXT NOT NULL, updated TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, day TEXT NOT NULL,
            event TEXT NOT NULL, payload TEXT NOT NULL);
    """)
    return c


def save_plan(plan: Plan) -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO plans VALUES (?, ?, ?)",
                  (plan.date, plan.model_dump_json(), datetime.now().isoformat(timespec="seconds")))


def load_plan(day: str) -> Plan | None:
    with _conn() as c:
        row = c.execute("SELECT json FROM plans WHERE day = ?", (day,)).fetchone()
    return Plan.model_validate_json(row[0]) if row else None


def log(day: str, event: str, payload) -> None:
    """event: import | normalized | solved | edit | agent | approved | sent"""
    with _conn() as c:
        c.execute("INSERT INTO journal (ts, day, event, payload) VALUES (?, ?, ?, ?)",
                  (datetime.now().isoformat(timespec="seconds"), day, event,
                   payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str)))


def journal(day: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT ts, event, payload FROM journal WHERE day = ? ORDER BY id", (day,)).fetchall()
    return [{"ts": ts, "event": ev, "payload": p} for ts, ev, p in rows]
