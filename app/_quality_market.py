"""Optimistic, commit-aware read consistency for explicit quality requests only.

One autocommit observer connection spans preparation, solve and validation.
data_version is compared only on that connection; no long read transaction pins
an old market. Any intervening commit invalidates the attempt, even inside one
refresh generation. The optional existing lease row also rejects partial refresh.
No refresh is initiated, no state file is created and no writer is blocked.
"""
from contextlib import closing
from pathlib import Path
import sqlite3
import sys


MIN_RETRY_SECONDS = 1.0


class MarketUnavailable(RuntimeError):
    pass


class MarketWatch:
    def __init__(self, connection, region):
        self.connection = connection
        self.region = region
        self.changed_seen = False

    def __enter__(self):
        self.manager = self.connection()
        self.conn = self.manager.__enter__()
        try:
            if self.conn.in_transaction:
                raise MarketUnavailable('observer_transaction')
            paths = self.conn.execute('PRAGMA database_list').fetchall()
            path = next((r[2] for r in paths if r[1] == 'main'), '')
            self.state_path = Path(path).parent / 'app_state.sqlite3' if path else None
            self.initial = self.read()
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *args):
        try:
            return self.manager.__exit__(*args)
        finally:
            if isinstance(self.manager, sqlite3.Connection):
                self.conn.close()

    def read(self):
        try:
            version = self.conn.execute('PRAGMA data_version').fetchone()[0]
            refresh = dict(generation=0, running=False, tracked=False)
            if self.state_path is not None and self.state_path.exists():
                with closing(sqlite3.connect(self.state_path.resolve().as_uri()+'?mode=ro', uri=True, timeout=.05)) as c:
                    row = c.execute('SELECT generation,running FROM refresh_jobs WHERE name=?', ('auction:'+self.region,)).fetchone()
                    if row:
                        # An expired running lease is not a completed publication.
                        refresh = dict(generation=row[0], running=bool(row[1]), tracked=True)
            return dict(data_version=version, **refresh)
        except sqlite3.Error as error:
            raise MarketUnavailable('market_state_unavailable') from error

    def changed(self):
        self.latest = self.read()
        self.changed_seen |= self.latest != self.initial
        return self.changed_seen
