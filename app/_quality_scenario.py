"""Point-scenario calculation views for unknown quality percentages.

Only explicit scenario requests use these helpers. A view is a request-local
replacement for an unknown physical percentage; it is never written back to the
market descriptor and never weakens original-descriptor validation.
"""
from dataclasses import replace
import hashlib
import json
import math

UNKNOWN_PERCENT_MODES = ('minimum', 'midpoint', 'maximum')
_MODE_NAMES = {
    'minimum': 'minimum_quality_point_scenario',
    'midpoint': 'expected_quality_point_scenario',
    'maximum': 'maximum_quality_point_scenario',
}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def resolve_mode(conservative=False, unknown_percent=None):
    """Resolve the effective unknown-percent mode.

    ``unknown_percent`` wins when explicitly supplied. ``conservative=True`` is
    the backward-compatible alias for the strict minimum mode; its absence keeps
    the historical maximum-tier behavior.
    """
    if unknown_percent is not None:
        if unknown_percent not in UNKNOWN_PERCENT_MODES:
            raise ValueError("unknown_percent must be one of: minimum, midpoint, maximum")
        return unknown_percent
    return 'minimum' if conservative else 'maximum'


def scenario_active(mode):
    return mode in ('minimum', 'midpoint')


def percent_known(raw, stored=None, quality=None, art_percent=None):
    """Whether a physical percentage came from a real quality roll.

    eAPI omits ``stats_random`` for many lots. The collector then stores the
    tier maximum as a display default, which must not be mistaken for a known
    percentage. The raw additional object preserves the distinction.
    """
    if isinstance(raw, dict) and 'stats_random' in raw:
        return _number(raw.get('stats_random'))
    if _number(stored) and stored != 0:
        return True
    # Legacy rows without raw additional: a non-maximum percentage is specific
    # enough to keep. A maximum may still be the synthetic default.
    if _number(art_percent) and quality is not None:
        from . import builder as b
        qr = b.QUALITY_RANGES.get(quality)
        if qr is not None:
            return not math.isclose(art_percent, qr['min'] + qr['range'], abs_tol=1e-9)
    return False


def source(option):
    return getattr(option, '_quality_source', option)


def percent_range(quality):
    """Nominal tier range for an artifact quality (None for unsupported tiers)."""
    from . import builder as b
    qr = b.QUALITY_RANGES.get(quality)
    if qr is None:
        return None
    return dict(min=qr['min'], midpoint=qr['min'] + qr['range'] / 2,
                maximum=qr['min'] + qr['range'])


def percent_views(options, priorities, mode='minimum'):
    """Replace only unknown physical percentages with a selected tier point.

    Known auction lots and owned instances keep their exact percentage in every
    mode. The original descriptor is retained on ``_quality_source`` for market
    validation and export.
    """
    if mode not in UNKNOWN_PERCENT_MODES:
        raise ValueError("mode must be one of: minimum, midpoint, maximum")
    if mode == 'maximum':
        return list(options)
    from . import builder as b
    result = []
    for original in options:
        physical = source(original)
        if getattr(physical, 'art_percent_known', False):
            result.append(original)
            continue
        qr = b.QUALITY_RANGES.get(physical.quality)
        if qr is None:
            # Capability checking must reject unsupported tiers, not coerce them.
            result.append(original)
            continue
        percent = qr['min'] if mode == 'minimum' else qr['min'] + qr['range'] / 2
        stats, raw, names, recommended = b.compute_artifact_stats(
            physical.base_stats, physical.bonus_stats, physical.unlocked_bonus_ids,
            percent, physical.ptn, qr['min'], qr['range'], physical.stats_random)
        view = replace(physical, art_percent=percent, stats=stats, stats_raw=raw,
                       stats_names=names, recommended_percent=recommended,
                       score=b._score_option(stats, priorities))
        view._quality_source = physical
        view._quality_percent_mode = mode
        result.append(view)
    return result


def minimum_views(options, priorities):
    """Backward-compatible strict worst-case view."""
    return percent_views(options, priorities, 'minimum')


def current_views(options, connection, region, normalize):
    """Check original descriptors with unchanged strict market validation."""
    from ._feasibility_fallback import current_candidates
    originals = [source(o) for o in options]
    valid = {id(o) for o in current_candidates(originals, connection, region, normalize)}
    return [o for o, original in zip(options, originals) if id(original) in valid]


def physical_digest(options):
    from ._physical_items import physical_key
    return hashlib.sha256(json.dumps([
        [physical_key(p), p.item_id, p.buyout_price, p.quality, p.ptn,
         p.art_percent, p.stats_random, p.unlocked_bonus_ids]
        for p in map(source, options)
    ], sort_keys=True, allow_nan=False).encode()).hexdigest()


def metadata(mode='minimum'):
    if mode not in UNKNOWN_PERCENT_MODES:
        raise ValueError("mode must be one of: minimum, midpoint, maximum")
    common = dict(mode=_MODE_NAMES[mode], unknown_percent=mode,
                  conservative=mode == 'minimum', robust=False,
                  guaranteed_carryability_implied=False,
                  known_percent_preserved=True)
    if mode == 'minimum':
        return dict(common, percentage_scope='quality_minimum_for_unknown_candidates',
                    message='Известный процент лота сохраняется; минимальный процент качества применяется только к неизвестным процентам. Это не гарантия характеристик и носимости при всех возможных процентах.')
    if mode == 'midpoint':
        return dict(common, percentage_scope='quality_midpoint_for_unknown_candidates',
                    message='Известный процент лота сохраняется; для неизвестных процентов берётся середина диапазона качества как ожидаемая оценка. Фактический ролл может быть ниже; это не гарантия.')
    return dict(common, percentage_scope='quality_maximum_for_unknown_candidates',
                message='Известный процент лота сохраняется; для неизвестных процентов берётся максимум диапазона качества. Это оптимистичная оценка, а не ожидаемая и не гарантированная.')
