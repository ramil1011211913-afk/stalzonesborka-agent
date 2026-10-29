"""Run the test suite against the read-only frozen snapshot, without live DB access."""
from contextlib import closing
import argparse
import json
import os
from pathlib import Path
import sqlite3

import pytest

from app import builder, test_builder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookahead", choices=["0", "1"], default="0")
    args = parser.parse_args()
    os.environ["STALZONE_DANGER_LOOKAHEAD"] = args.lookahead
    config = json.loads(Path("data/validation/exact-test-config.json").read_text(encoding="utf-8-sig"))
    snapshot = Path(config["snapshot"]).resolve()
    wal = Path(str(snapshot) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError("Snapshot contains uncheckpointed WAL data")

    def connection():
        conn = sqlite3.connect(snapshot.as_uri() + "?mode=ro&immutable=1", uri=True)
        conn.row_factory = sqlite3.Row
        return closing(conn)

    def get_artefact(item_id):
        with connection() as conn:
            row = conn.execute("SELECT stats, bonus_stats FROM artefacts WHERE id=?", (item_id,)).fetchone()
        assert row is not None
        return json.loads(row["stats"]), json.loads(row["bonus_stats"])

    builder.get_connection = connection
    test_builder._get_artefact = get_artefact
    raise SystemExit(pytest.main(["app/test_builder.py", "tests", "-q"]))


if __name__ == "__main__":
    main()
