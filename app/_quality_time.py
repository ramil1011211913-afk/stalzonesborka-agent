"""Explicit deep-search time and finite-node policy; mathematical model unchanged."""
import math

DEEP_SECONDS = 300.
DEEP_NODE_LIMIT = 50000  # Finite deep-only cap; standard model remains 20,000.
VALIDATION_RESERVE = 1.


def request_seconds(seconds, deep=False):
    maximum = DEEP_SECONDS if deep else 60.
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= maximum:
        raise ValueError('quality_timeout must be in (0, %g]' % maximum)
    return DEEP_SECONDS if deep else seconds


def stage_seconds(model, stage, count, available):
    if model.get('time_policy') != 'deep_remaining':
        return min(model['model_seconds'], available)
    # Reserve the old per-stage allowance for each later priority/cost stage.
    # With little time left, divide it rather than silently starving later goals.
    available = max(0., available - .1)  # native result/checkpoint teardown
    later = count-stage-1
    reserve = min(model['stage_reserve_seconds']*later, available*later/(later+1))
    return min(model['model_seconds'], available-reserve)
