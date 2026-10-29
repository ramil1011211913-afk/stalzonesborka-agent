"""Shared SQLite application state and fenced auction-refresh leases.

No market data or legacy JSON is modified by this module. Connections are
short-lived; BEGIN IMMEDIATE serializes ownership and state changes.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid

APPLICATION_OWNER = 'application'
DEFAULT_LEASE_SECONDS = 60.0


class LeaseLost(RuntimeError):
    pass


class RefreshAlreadyRunning(RuntimeError):
    pass


class StateStore:
    def __init__(self, path, clock=time.time):
        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            # Concurrent first-open journal-mode upgrades can return BUSY without
            # honoring busy_timeout. Retry only SQLite lock errors, bounded.
            deadline = time.monotonic() + 30
            while True:
                try:
                    if conn.execute('PRAGMA journal_mode').fetchone()[0].lower() != 'wal':
                        conn.execute('PRAGMA journal_mode=WAL')
                    conn.executescript('''
                CREATE TABLE IF NOT EXISTS owned_state (
                    owner_key TEXT PRIMARY KEY, payload TEXT NOT NULL,
                    revision INTEGER NOT NULL, updated_at REAL NOT NULL,
                    source_sha256 TEXT);
                CREATE TABLE IF NOT EXISTS refresh_jobs (
                    name TEXT PRIMARY KEY, token TEXT, generation INTEGER NOT NULL,
                    running INTEGER NOT NULL, lease_until REAL NOT NULL,
                    done INTEGER NOT NULL, total INTEGER NOT NULL, error TEXT,
                    owner_pid INTEGER, updated_at REAL NOT NULL);
                    ''')
                    break
                except sqlite3.OperationalError as error:
                    if (getattr(error, 'sqlite_errorcode', -1) & 255 not in
                            (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or time.monotonic() >= deadline):
                        raise
                    time.sleep(.01)

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA busy_timeout=30000')
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self):
        with self.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def save_owned(self, artifacts, owner=APPLICATION_OWNER):
        if not isinstance(artifacts, list) or any(not isinstance(a, dict) for a in artifacts):
            raise ValueError('Owned artifacts must be a list of records')
        payload = json.dumps(artifacts, ensure_ascii=False, allow_nan=False)
        with self.transaction() as conn:
            conn.execute('''INSERT INTO owned_state VALUES (?, ?, 1, ?, NULL)
                ON CONFLICT(owner_key) DO UPDATE SET payload=excluded.payload,
                revision=owned_state.revision+1, updated_at=excluded.updated_at''',
                (owner, payload, self.clock()))

    def import_legacy(self, path, owner=APPLICATION_OWNER):
        """Import only into an absent owner; never write/delete the source JSON."""
        with self.connection() as conn:
            if conn.execute('SELECT 1 FROM owned_state WHERE owner_key=?', (owner,)).fetchone():
                return False
        path = Path(path)
        if not path.exists():
            return False
        raw = path.read_bytes()
        value = json.loads(raw.decode('utf-8-sig'))
        if not isinstance(value, list) or any(not isinstance(a, dict) for a in value):
            raise ValueError('Legacy owned artifacts must be a list of records')
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False)
        with self.transaction() as conn:
            result = conn.execute('INSERT OR IGNORE INTO owned_state VALUES (?, ?, 1, ?, ?)',
                                  (owner, payload, self.clock(), hashlib.sha256(raw).hexdigest()))
            return result.rowcount == 1

    def owned(self, owner=APPLICATION_OWNER):
        with self.connection() as conn:
            row = conn.execute('SELECT * FROM owned_state WHERE owner_key=?', (owner,)).fetchone()
        return dict(scope=owner, artifacts=json.loads(row['payload']) if row else [],
                    revision=row['revision'] if row else 0)

    def claim(self, name, seconds=DEFAULT_LEASE_SECONDS):
        if not 0 < seconds < float('inf'):
            raise ValueError('Lease duration must be finite and positive')
        with self.transaction() as conn:
            now = self.clock()
            row = conn.execute('SELECT * FROM refresh_jobs WHERE name=?', (name,)).fetchone()
            if row and row['running'] and row['lease_until'] > now:
                return None
            generation = row['generation'] + 1 if row else 1
            token = uuid.uuid4().hex
            conn.execute('''INSERT INTO refresh_jobs VALUES (?, ?, ?, 1, ?, 0, 0, NULL, ?, ?)
                ON CONFLICT(name) DO UPDATE SET token=excluded.token, generation=excluded.generation,
                running=1, lease_until=excluded.lease_until, done=0, total=0, error=NULL,
                owner_pid=excluded.owner_pid, updated_at=excluded.updated_at''',
                (name, token, generation, now + seconds, os.getpid(), now))
        return Lease(self, name, token, generation, seconds)

    def status(self, name):
        with self.connection() as conn:
            row = conn.execute('SELECT * FROM refresh_jobs WHERE name=?', (name,)).fetchone()
        if row is None:
            return dict(running=False, done=0, total=0, error=None, state='idle', job_id=0)
        expired = bool(row['running'] and row['lease_until'] <= self.clock())
        running = bool(row['running'] and not expired)
        return dict(running=running, done=row['done'], total=row['total'],
                    error='Refresh owner lease expired' if expired else row['error'],
                    state='expired' if expired else 'running' if running else 'failed' if row['error'] else 'completed',
                    job_id=row['generation'], lease_expires_at=row['lease_until'])


@dataclass
class Lease:
    store: StateStore
    name: str
    token: str
    generation: int
    seconds: float

    def _valid(self, conn):
        return conn.execute('''SELECT 1 FROM refresh_jobs WHERE name=? AND token=? AND generation=?
            AND running=1 AND lease_until>?''',
            (self.name, self.token, self.generation, self.store.clock())).fetchone() is not None

    @contextmanager
    def guard(self):
        """Keep the state writer lock until the guarded market transaction commits.

        No new generation can be claimed during that commit. A stale owner
        cannot enter this guard after another worker has acquired ownership.
        """
        with self.store.transaction() as conn:
            if not self._valid(conn):
                raise LeaseLost('Auction refresh ownership lost')
            yield

    def heartbeat(self):
        with self.store.transaction() as conn:
            if not self._valid(conn):
                return False
            now = self.store.clock()
            conn.execute('UPDATE refresh_jobs SET lease_until=?, updated_at=? WHERE name=?',
                         (now + self.seconds, now, self.name))
            return True

    def progress(self, done, total):
        if not 0 <= done <= total:
            raise ValueError('Invalid refresh progress')
        with self.store.transaction() as conn:
            if not self._valid(conn):
                raise LeaseLost('Auction refresh ownership lost')
            conn.execute('UPDATE refresh_jobs SET done=?, total=?, updated_at=? WHERE name=?',
                         (done, total, self.store.clock(), self.name))

    def finish(self, error=None):
        with self.store.transaction() as conn:
            if not self._valid(conn):
                return False
            conn.execute('UPDATE refresh_jobs SET running=0, lease_until=0, error=?, updated_at=? WHERE name=?',
                         (error, self.store.clock(), self.name))
            return True

    @contextmanager
    def keep_alive(self):
        stop, lost = threading.Event(), threading.Event()
        def renew():
            while not stop.wait(self.seconds / 3):
                try:
                    if not self.heartbeat():
                        lost.set()
                        return
                except Exception:
                    lost.set()
                    return
        thread = threading.Thread(target=renew, name='auction-lease-heartbeat', daemon=True)
        thread.start()
        try:
            yield lost
        finally:
            stop.set()
            thread.join(timeout=31)
            if thread.is_alive():
                raise RuntimeError('Lease heartbeat did not stop')


def application_store(data_dir):
    return StateStore(Path(data_dir) / 'app_state.sqlite3')


def auction_job(region):
    return 'auction:' + region
