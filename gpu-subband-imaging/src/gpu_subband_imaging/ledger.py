"""SQLite state for restartable `(band, batch)` jobs."""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

QUEUED, RUNNING, DONE, FAILED, SKIPPED = "queued", "running", "done", "failed", "skipped"


@dataclass
class Job:
    band: int
    batch: int
    n_files: int
    status: str = QUEUED
    slot: str = ""
    attempts: int = 0
    error: str = ""

    @property
    def key(self) -> str:
        return f"{self.band}:{self.batch}"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    band INTEGER, batch INTEGER, n_files INTEGER,
    status TEXT, slot TEXT, attempts INTEGER, error TEXT,
    updated REAL,
    PRIMARY KEY (band, batch)
);
"""


class Ledger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # One orchestrator owns the connection; its dispatch threads share it.
        self._lock = threading.Lock()
        self._c = sqlite3.connect(str(path), isolation_level=None,
                                  check_same_thread=False, timeout=60)
        # DELETE mode avoids WAL coordination files on shared storage.
        self._c.execute("PRAGMA journal_mode=DELETE")
        self._c.execute("PRAGMA busy_timeout=60000")
        self._c.execute(_SCHEMA)

    def reset_stale_running(self) -> int:
        """Return jobs left running by a dead orchestrator to the retry queue."""
        with self._lock:
            cur = self._c.execute(
                "UPDATE jobs SET status=?, error='stale running (reset)' WHERE status=?",
                (FAILED, RUNNING))
            return cur.rowcount

    def seed(self, jobs: List[Job]) -> None:
        """Insert new jobs without replacing prior run state."""
        with self._lock:
            now = time.time()
            rows = [(j.band, j.batch, j.n_files, QUEUED, "", 0, "", now)
                    for j in jobs]
            self._c.execute("BEGIN")
            try:
                self._c.executemany(
                    "INSERT OR IGNORE INTO jobs VALUES (?,?,?,?,?,?,?,?)", rows)
                self._c.execute("COMMIT")
            except Exception:
                self._c.execute("ROLLBACK")
                raise

    def _set(self, band: int, batch: int, **cols) -> None:
        cols["updated"] = time.time()
        keys = ", ".join(f"{k}=?" for k in cols)
        with self._lock:
            self._c.execute(f"UPDATE jobs SET {keys} WHERE band=? AND batch=?",
                            (*cols.values(), band, batch))

    def mark_running(self, j: Job, slot: str) -> None:
        self._set(j.band, j.batch, status=RUNNING, slot=slot, attempts=j.attempts + 1)

    def mark_done(self, j: Job) -> None:
        self._set(j.band, j.batch, status=DONE, error="")

    def mark_failed(self, j: Job, error: str) -> None:
        self._set(j.band, j.batch, status=FAILED, error=error[-500:])

    def mark_skipped(self, j: Job, reason: str) -> None:
        self._set(j.band, j.batch, status=SKIPPED, error=reason[-500:])

    def _rows(self, where: str, args=()) -> List[Job]:
        with self._lock:
            cur = self._c.execute(
                f"SELECT band,batch,n_files,status,slot,attempts,error FROM jobs {where}",
                args)
            return [Job(b, bt, n, st, sl, at, er) for b, bt, n, st, sl, at, er in cur]

    def pending(self, include_running: bool = True) -> List[Job]:
        states = [QUEUED, FAILED] + ([RUNNING] if include_running else [])
        q = ",".join("?" * len(states))
        return self._rows(f"WHERE status IN ({q}) ORDER BY band, batch", states)

    def all(self) -> List[Job]:
        return self._rows("ORDER BY band, batch")

    def summary(self) -> dict:
        with self._lock:
            cur = self._c.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
            return dict(cur.fetchall())

    def close(self) -> None:
        with self._lock:
            self._c.close()
