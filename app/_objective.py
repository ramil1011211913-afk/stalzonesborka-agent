"""Immutable request-wide objectives shared by all search engines.

Lexicographic equality uses fixed quantization bins of width tolerance, rather
than a non-transitive pairwise abs(a-b) <= tolerance comparison.
"""
from dataclasses import dataclass
import math
import numpy as np

DANGER_KEYS = ("radiation_accumulation", "biological_accumulation",
               "psycho_accumulation", "thermal_accumulation", "frost_accumulation")
CONTRACT_VERSION = "balanced-v1"


def direction(key):
    return -1.0 if key.rsplit(".", 1)[-1] in DANGER_KEYS else 1.0


@dataclass(frozen=True)
class Objective:
    keys: tuple
    weights: tuple
    scales: tuple
    mode: str = "balanced"
    order: tuple = ()
    tolerance: float = 1e-6

    def __post_init__(self):
        for field in ("keys", "weights", "scales", "order"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if (len(self.keys) != len(self.weights) or len(self.keys) != len(self.scales)
                or any(not math.isfinite(s) or s <= 0 for s in self.scales)
                or any(not math.isfinite(w) or w <= 0 for w in self.weights)):
            raise ValueError("Invalid objective dimensions, weights or scales")
        if self.mode not in ("balanced", "lexicographic", "min_budget") or not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("Invalid objective mode or tolerance")
        if any(not isinstance(key, str) for key in self.keys) or len(set(self.keys)) != len(self.keys):
            raise ValueError("Invalid objective stat keys")
        if self.mode == "lexicographic" and (not self.order or len(set(self.order)) != len(self.order) or set(self.order) != set(self.keys)):
            raise ValueError("Invalid lexicographic order")
        if self.mode != "lexicographic" and self.order:
            raise ValueError("Only lexicographic objectives have an order")

    @classmethod
    def create(cls, priorities, scales=None, mode="balanced", order=None, tolerance=1e-6):
        if mode not in ("balanced", "lexicographic", "min_budget"):
            raise ValueError("Unknown optimization mode")
        if not isinstance(priorities, dict):
            raise ValueError("Priorities must be a mapping")
        if scales is not None and not isinstance(scales, dict):
            raise ValueError("Scales must be a mapping")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Objective tolerance must be finite and positive")
        keys, weights, fixed = [], [], []
        for key, weight in priorities.items():
            if not isinstance(key, str) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
                raise ValueError("Weights must be finite numbers")
            if weight <= 0:
                continue
            scale = (scales or {}).get(key, 1.0)
            if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
                raise ValueError("Objective scales must be finite and positive")
            if not math.isfinite(weight / scale):
                raise ValueError("Objective coefficient overflows")
            keys.append(key)
            weights.append(float(weight))
            fixed.append(float(scale))
        if mode == "lexicographic":
            if (not isinstance(order, (list, tuple)) or not order
                    or any(not isinstance(key, str) for key in order)
                    or len(set(order)) != len(order) or set(order) != set(keys)):
                raise ValueError("Lexicographic order must list each positive-weight stat exactly once")
        return cls(tuple(keys), tuple(weights), tuple(fixed), mode,
                   tuple(order or ()) if mode == "lexicographic" else (), float(tolerance))

    @property
    def coefficients(self):
        return tuple(w * direction(k) / scale for k, w, scale in zip(self.keys, self.weights, self.scales))

    def evaluate(self, stats, price=0):
        if self.mode == "min_budget":
            return -float(price)
        return sum(stats.get(k, 0.0) * c for k, c in zip(self.keys, self.coefficients))

    def matrix_score(self, stats, stat_keys, xp=np, prices=None):
        if self.mode == "min_budget":
            return -prices.copy()
        columns = xp.asarray([stat_keys.index(k) for k in self.keys], dtype=xp.int32)
        coefficients = xp.asarray(self.coefficients, dtype=xp.float64)
        return xp.sum(stats[:, columns] * coefficients, axis=1)

    def matrix_ranks(self, stats, stat_keys, xp=np, adjustment=None):
        columns = xp.asarray([stat_keys.index(k) for k in self.order], dtype=xp.int32)
        signs = xp.asarray([direction(k) for k in self.order])
        values = stats[:, columns] * signs
        if adjustment is not None:
            values[:, 0] += adjustment
        return xp.floor(values / self.tolerance + 0.5)

    def rank(self, stats, price=0):
        if self.mode != "lexicographic":
            return (self.evaluate(stats, price),)
        return tuple(math.floor(direction(k) * stats.get(k, 0.0) / self.tolerance + 0.5)
                     for k in self.order)

    def sort_key(self, build):
        return tuple(-v for v in self.rank(build.combined_stats, build.total_price)) + (build.total_price,)

    def metadata(self):
        from ._survivability import REGISTRY, CONTRACT
        version = CONTRACT if any(k in REGISTRY for k in self.keys) else CONTRACT_VERSION
        return {"contract": version, "mode": self.mode, "weights": dict(zip(self.keys, self.weights)),
                "scales": dict(zip(self.keys, self.scales)), "order": list(self.order),
                "tolerance": self.tolerance}


def derive_scales(stat_rows, keys, slots, bonuses=None):
    """A beam-independent uncoupled magnitude bound; zero columns use scale 1."""
    bonuses = bonuses or {}
    result = {}
    for key in keys:
        values = [abs(float(row.get(key, 0.0))) for row in stat_rows]
        bonus = abs(float(bonuses.get(key, 0.0)))
        if not all(math.isfinite(value) for value in values) or not math.isfinite(bonus):
            raise ValueError("Nonfinite stats cannot define an objective scale")
        bound = slots * max(values, default=0.0) + bonus
        if not math.isfinite(bound):
            raise ValueError("Objective scale overflows")
        result[key] = max(1.0, bound)
    return result
