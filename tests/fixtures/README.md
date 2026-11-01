`feasibility-market.sqlite.gz` is the 2026-10-17 14:58:42 UTC market
snapshot used by the feasibility regression tests. Its source was the audit
snapshot with SHA-256 `71ca56312db5cfd75a9eed95d5ee94abbae60943208995d50583ee074759b72e`.
The auction history was removed and the database vacuumed before compression;
artifact, container, armor and active-lot tables are preserved. The snapshot
contains historical market data, not a current price feed.
