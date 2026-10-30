"""Minimum-quality calculation views, separate from original market descriptors.

Only explicit conservative Max Quality requests use these helpers. This is a
point scenario, not a robust bound and not a change to stats_random conversion.
"""
from dataclasses import replace
import hashlib
import json


def source(option):
    return getattr(option, '_quality_source', option)


def minimum_views(options, priorities):
    """Use the tier minimum only where the physical percentage is unknown.

    Auction lots and owned instances with a stored percentage keep that exact
    percentage; replacing it would lower known items below their real stats.
    """
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
        stats, raw, names, recommended = b.compute_artifact_stats(
            physical.base_stats, physical.bonus_stats, physical.unlocked_bonus_ids,
            qr['min'], physical.ptn, qr['min'], qr['range'], physical.stats_random)
        view = replace(physical, art_percent=qr['min'], stats=stats, stats_raw=raw,
                       stats_names=names, recommended_percent=recommended,
                       score=b._score_option(stats, priorities))
        view._quality_source = physical
        result.append(view)
    return result


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


def metadata():
    return dict(mode='minimum_quality_point_scenario', conservative=True,
                robust=False, guaranteed_carryability_implied=False,
                percentage_scope='quality_minimum_for_unknown_candidates',
                known_percent_preserved=True,
                message='Известный процент лота сохраняется; минимальный процент качества применяется только к неизвестным процентам. Это не гарантия характеристик и носимости при всех возможных процентах.')
