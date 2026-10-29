"""Full physical input and exact, request-local reduction for Max Quality only.

Ordinary search never calls this module. Unknown records are accounted for, not
completed with simulated bonus hypotheses or SQL defaults presented as facts.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import time


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def catalogue_supported(base, bonus, ids, normalize):
    """Only opened bonus coefficients are mandatory for this physical profile."""
    if not isinstance(base, list) or not base or not isinstance(bonus, list):
        return False
    if any(not isinstance(s, dict) or not isinstance(s.get('prop_id'), str)
           or not s['prop_id'] for s in base + bonus):
        return False
    selected = set(map(normalize, ids))
    if selected - {normalize(s['prop_id'].upper()) for s in bonus}:
        return False
    relevant = base + [s for s in bonus if normalize(s['prop_id'].upper()) in selected]
    return all(_number(s.get('min')) and _number(s.get('max')) for s in relevant)


def load(builder, region, max_price, stock_only, audit):
    """Stream the whole region; never window/rank/fairness-filter physical lots."""
    from ._max_quality_model import descriptor_supported
    exclusions = Counter()
    audit.update(source='full_physical_auction', region=region, original_physical_offers=0,
                 visited_physical_offers=0, enumeration_complete=True,
                 exclusions={}, unknown_exclusions={}, accepted_physical_offers=0,
                 actual_percent_confirmed=False)
    mandatory = Counter(); unknown = Counter(); result = []; catalogue = {}
    now = datetime.now(timezone.utc)
    with builder.get_connection() as conn:
        audit['original_physical_offers'] = conn.execute(
            'SELECT COUNT(*) FROM auction_lots WHERE region=?', (region,)).fetchone()[0]
        rows = conn.execute('''SELECT l.*, a.id AS catalogue_id,
                a.name_ru, a.name_en, a.subcategory, a.stats AS catalogue_stats,
                a.bonus_stats AS catalogue_bonuses, a.icon_url
            FROM auction_lots l LEFT JOIN artefacts a ON a.id=l.item_id
            WHERE l.region=? ORDER BY l.id''', (region,))
        for r in rows:
            if time.monotonic() >= audit.get('deadline', math.inf):
                audit.update(enumeration_complete=False, error='preparation_time_limit'); break
            audit['visited_physical_offers'] += 1
            reason = None; is_unknown = False
            try:
                end = datetime.fromisoformat(r['end_time'].replace('Z', '+00:00'))
                if end.tzinfo is None:
                    reason = 'unknown_expiry'; is_unknown = True
                elif end <= now:
                    reason = 'expired'
                elif not _number(r['buyout_price']) or r['buyout_price'] <= 0:
                    reason = 'no_positive_buyout'
                elif max_price is not None and r['buyout_price'] > max_price:
                    reason = 'individual_price_exceeds_budget'
                elif type(r['buyout_price']) is not int:
                    reason = 'invalid_price_descriptor'; is_unknown = True
                elif 'ptn' not in r.keys() or type(r['ptn']) is not int or not 0 <= r['ptn'] <= 15:
                    reason = 'unknown_or_invalid_ptn'; is_unknown = True
                elif stock_only and r['ptn'] != 0:
                    reason = 'stock_only'
                elif r['catalogue_id'] is None:
                    reason = 'unknown_catalogue_type'; is_unknown = True
                else:
                    raw = json.loads(r['additional_raw']) if 'additional_raw' in r.keys() and r['additional_raw'] else {}
                    if not isinstance(raw, dict) or type(raw.get('qlt')) is not int or 'quality' not in r.keys() or type(r['quality']) is not int or raw['qlt'] != r['quality'] or not 0 <= r['quality'] <= 5:
                        reason = 'unknown_or_inconsistent_quality'; is_unknown = True
                    elif 'ptn' in raw and (type(raw['ptn']) is not int or raw['ptn'] != r['ptn']):
                        reason = 'inconsistent_ptn'; is_unknown = True
                    elif 'amount' not in r.keys() or r['amount'] != 1:
                        reason = 'ambiguous_physical_quantity'; is_unknown = True
                    else:
                        if r['item_id'] not in catalogue:
                            base = json.loads(r['catalogue_stats'] or '[]')
                            bonus = json.loads(r['catalogue_bonuses'] or '[]')
                            catalogue[r['item_id']] = base, bonus
                        base, bonus = catalogue[r['item_id']]
                        ids = json.loads(r['bonus_properties'] or 'null')
                        norm = builder._normalize_bonus_id
                        if r['item_id'] in builder.EXCLUDED_ARTIFACTS or not base:
                            reason = 'unsupported_base_or_catalogue_type'; is_unknown = True
                        elif not isinstance(base, list) or not isinstance(bonus, list) or any(
                                not isinstance(s, dict) or not isinstance(s.get('prop_id'), str) or not s['prop_id']
                                for s in base + bonus):
                            reason = 'invalid_catalogue_coefficients'; is_unknown = True
                        elif not isinstance(ids, list) or any(not isinstance(v, str) for v in ids):
                            reason = 'missing_bonus_descriptor'; is_unknown = True
                        elif len(set(map(norm, ids))) != len(ids) or len(ids) != r['ptn'] // 5:
                            reason = 'incomplete_or_duplicate_bonuses'; is_unknown = True
                        elif set(map(norm, ids)) - {norm(s['prop_id'].upper()) for s in bonus}:
                            reason = 'unsupported_bonus_ids'; is_unknown = True
                        elif not catalogue_supported(base, bonus, ids, norm):
                            reason = 'invalid_catalogue_coefficients'; is_unknown = True
                        elif not _number(r['art_percent']) or not 0 < r['art_percent'] <= 190 or (r['stats_random'] is not None and not _number(r['stats_random'])):
                            reason = 'invalid_original_percent_descriptor'; is_unknown = True
                        else:
                            qr = builder.QUALITY_RANGES[r['quality']]
                            values, raw_values, names, recommended = builder.compute_artifact_stats(
                                base, bonus, ids, r['art_percent'], r['ptn'], qr['min'], qr['range'], r['stats_random'])
                            if not all(_number(v) for v in values.values()):
                                reason = 'nonfinite_characteristics'; is_unknown = True
                            else:
                                option = builder.ArtifactOption(item_id=r['item_id'], name_ru=r['name_ru'], name_en=r['name_en'],
                                    subcategory=r['subcategory'], quality=r['quality'], quality_name=builder.QUALITY_NAMES[r['quality']],
                                    art_percent=r['art_percent'], buyout_price=r['buyout_price'], ptn=r['ptn'],
                                    stats=values, stats_raw=raw_values, stats_names=names, unlocked_bonus_ids=ids,
                                    icon_url=r['icon_url'], base_stats=base, bonus_stats=bonus, stats_random=r['stats_random'],
                                    recommended_percent=recommended, auction_lot_id=r['id'], physical_id=f"auction:{region}:{r['id']}")
                                if descriptor_supported(option):
                                    result.append(option)
                                else:
                                    reason = 'unsupported_descriptor'; is_unknown = True
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError):
                reason = 'malformed_or_incomplete_record'; is_unknown = True
            if reason:
                exclusions[reason] += 1
                (unknown if is_unknown else mandatory)[reason] += 1
            if len(result) > audit.get('max_input_candidates', 10000):
                audit.update(enumeration_complete=False, error='input_candidate_limit'); break
    audit.update(exclusions=dict(exclusions), mandatory_exclusions=dict(mandatory),
                 unknown_exclusions=dict(unknown), accepted_physical_offers=len(result),
                 coverage='all_supported_physical_offers' if audit['enumeration_complete'] else 'incomplete_enumeration')
    return result


def fingerprint(options):
    from ._physical_items import physical_key
    from ._quality_scenario import source
    return hashlib.sha256(json.dumps([
        [physical_key(o), source(o).item_id, source(o).quality, source(o).ptn,
         source(o).buyout_price, source(o).art_percent, source(o).stats_random,
         source(o).unlocked_bonus_ids, o.stats] for o in options
    ], sort_keys=True, allow_nan=False).encode()).hexdigest()


def exact_reduce(candidates, slots, duplicates):
    """Keep k cheapest independent physical copies per EXACT calculation profile.

    Equivalence contract: the same type, level, selected bonus set, and *every*
    binary64 stat coefficient equal bit-for-bit (float.hex; never isclose or
    rounding). All model constraints/objectives and native recomputation use
    these same complete vectors at this container. Same type preserves native
    summation order; downstream derived properties depend only on the totals.
    No robust/guaranteed carry model is covered by this point-model contract.
    Owned/mixed sources and alternative physical hypotheses are left intact.
    """
    from ._physical_items import physical_key
    from ._quality_scenario import source
    from .builder import _normalize_bonus_id
    meta = dict(before_exact_reduction=len(candidates), provably_removed=0,
                solver_candidates=len(candidates), contract='exact-binary64-full-vector-v1',
                skipped_reason=None, full_supported_pool_sha256=fingerprint(candidates))
    if not duplicates or any(o.is_owned for o in candidates):
        meta['skipped_reason'] = 'type_uniqueness_or_owned_semantics'
        return candidates, meta
    physical_counts = Counter(physical_key(o) for o in candidates)
    groups = {}; retain = set()
    for i, o in enumerate(candidates):
        if physical_counts[physical_key(o)] != 1 or source(o).auction_lot_id is None:
            retain.add(i); continue
        if any(not _number(v) for v in o.stats.values()):
            retain.add(i); continue
        key = (o.item_id, o.ptn, tuple(sorted(map(_normalize_bonus_id, o.unlocked_bonus_ids))),
               tuple(sorted((k, float(v).hex()) for k, v in o.stats.items())))
        groups.setdefault(key, []).append(i)
    for group in groups.values():
        retain.update(sorted(group, key=lambda i: (candidates[i].buyout_price, i))[:slots])
    result = [o for i, o in enumerate(candidates) if i in retain]
    meta.update(provably_removed=len(candidates)-len(result), solver_candidates=len(result))
    return result, meta
