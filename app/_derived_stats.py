"""Linear derived statistics shared by every search path.

A derived statistic is not a separate game property: it is the sum of named
game stats.  Materialising it in option/build stats keeps Beam, Numba, MILP,
thresholds and the exported payload on one definition.
"""
from __future__ import annotations

DERIVED_STATS = {
    "total_speed": ("speed_modifier", "sprint_speed_modifier"),
}
NAMES = {
    "total_speed": "Общая скорость",
}


def value(stats: dict, key: str) -> float:
    """Return a stat, computing a known linear derivative when absent."""
    if key in stats:
        return stats[key]
    parts = DERIVED_STATS.get(key)
    if parts is None:
        return 0.0
    return sum(stats.get(part, 0.0) for part in parts)


def augment(stats: dict) -> dict:
    """Recompute materialised derivatives from their current components."""
    for key, parts in DERIVED_STATS.items():
        if any(part in stats for part in parts):
            stats[key] = sum(stats.get(part, 0.0) for part in parts)
    return stats


def add_names(stats: dict, names: dict) -> dict:
    for key in DERIVED_STATS:
        if key in stats:
            names.setdefault(key, NAMES[key])
    return names


def _bonus_value(entry) -> float:
    return entry.get("value", 0) if isinstance(entry, dict) else entry


def augment_bonus_stats(bonus_stats: dict) -> dict:
    """Recompute derivative entries in container/armor bonus stats."""
    if not bonus_stats:
        return bonus_stats
    for key, parts in DERIVED_STATS.items():
        if not any(part in bonus_stats for part in parts):
            continue
        total = sum(_bonus_value(bonus_stats.get(part, 0)) for part in parts)
        if any(isinstance(bonus_stats.get(part), dict) for part in parts):
            data = dict(bonus_stats[key]) if isinstance(bonus_stats.get(key), dict) else {}
            data["value"] = total
            data.setdefault("name_ru", NAMES[key])
            bonus_stats[key] = data
        else:
            bonus_stats[key] = total
    return bonus_stats
