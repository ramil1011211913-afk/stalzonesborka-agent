"""Bounded process-local witnesses, never a solver certificate or saved user state."""
from collections import OrderedDict
import hashlib
import json
import threading

from ._physical_items import physical_key, owned_identity


def pool_key(region, container, candidates, budget, duplicates, guaranteed):
    """Fingerprint the entire ordered prepared pool and effective game profile.

    Thresholds are deliberately absent: each witness must be validated against
    the CURRENT request. Any changed candidate, price, physical identity or
    armor profile invalidates this conservative match.
    """
    if len(candidates) > 5000:
        return None
    try:
        payload = [region, container.item_id, container.slots,
                   container.inner_protection, container.effectiveness,
                   container.bonus_stats, getattr(container, '_armor', None),
                   budget, duplicates, guaranteed,
                   [[c.item_id, physical_key(c), c.buyout_price,
                     sorted((k, float(v)) for k, v in c.stats.items())]
                    for c in candidates]]
        raw = json.dumps(payload, sort_keys=True, allow_nan=False,
                         separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError, OverflowError):
        return None
    return hashlib.sha256(raw).digest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def profile_key(region, container, budget, duplicates, guaranteed, carry):
    """Only compatible game profiles may exchange physical witnesses."""
    try:
        value = [region, container.item_id, container.slots,
                 container.inner_protection, container.effectiveness,
                 container.bonus_stats, getattr(container, '_armor', None),
                 budget, duplicates, guaranteed, carry]
        return hashlib.sha256(_canonical(value).encode()).digest()
    except (TypeError, ValueError, OverflowError):
        return None


def _reference(candidate):
    """Identity and variant digest; never old values for game calculation."""
    identity = physical_key(candidate)
    if (identity[0] == 'anonymous' or str(identity[1]).startswith('anonymous-row:')
            or len(str(identity[1]).encode('utf-8')) > 512):
        return None
    fields = [candidate.item_id, candidate.buyout_price, candidate.stats,
              [getattr(candidate, key, None) for key in
               ('quality', 'ptn', 'art_percent', 'stats_random', 'is_owned',
                'base_stats', 'bonus_stats')],
              sorted(getattr(candidate, 'unlocked_bonus_ids', []))]
    try:
        return identity, hashlib.sha256(_canonical(fields).encode()).digest()
    except (TypeError, ValueError, OverflowError):
        return None


def _owned_proof(selected, owned):
    """Anonymous owned identities are positional: accept unchanged prefixes only.

    Explicit instance IDs may move, but must have exactly one matching record.
    Bound retained descriptions independently from the number of auction rows.
    """
    end = 0
    explicit = []
    for candidate in selected:
        if not getattr(candidate, 'is_owned', False):
            continue
        identity = getattr(candidate, 'physical_id', '') or ''
        if identity.startswith('owned:position:'):
            index = int(identity.rsplit(':', 1)[1])
            if not 0 <= index < len(owned) or index >= 64:
                return None
            if owned_identity(owned[index], index)[1] != identity:
                return None
            end = max(end, index + 1)
        elif identity.startswith('owned:input:'):
            matches = [record for i, record in enumerate(owned)
                       if owned_identity(record, i)[1] == identity]
            if len(matches) != 1:
                return None
            explicit.append((identity, _canonical(matches[0])))
        else:
            return None
    prefix = tuple(_canonical(record) for record in owned[:end])
    proof = prefix, tuple(explicit)
    return proof if len(_canonical(proof).encode()) <= 16384 else None


def _matches_owned(proof, owned):
    prefix, explicit = proof
    if len(owned) < len(prefix) or tuple(_canonical(r) for r in owned[:len(prefix)]) != prefix:
        return False
    for identity, description in explicit:
        matches = [r for i, r in enumerate(owned) if owned_identity(r, i)[1] == identity]
        if len(matches) != 1 or _canonical(matches[0]) != description:
            return False
    return True


class Incumbents:
    def __init__(self, max_pools=64, max_per_pool=4):
        self._pools = OrderedDict()
        self._physical = OrderedDict()
        self._lock = threading.Lock()
        self.max_pools = max_pools
        self.max_per_pool = max_per_pool

    def get(self, key):
        with self._lock:
            if key not in self._pools:
                return ()
            self._pools.move_to_end(key)
            return tuple(self._pools[key])

    def remember(self, key, indices):
        if key is None:
            return
        witness = tuple(int(i) for i in indices)
        with self._lock:
            rows = self._pools.setdefault(key, [])
            if witness in rows:
                rows.remove(witness)
            rows.append(witness)
            del rows[:-self.max_per_pool]
            self._pools.move_to_end(key)
            while len(self._pools) > self.max_pools:
                self._pools.popitem(last=False)

    def remember_physical(self, profile, candidates, indices, owned):
        if profile is None or not 1 <= len(indices) <= 7:
            return
        selected = [candidates[int(i)] for i in indices]
        references = tuple(_reference(c) for c in selected)
        if any(r is None for r in references) or len({r[0] for r in references}) != len(references):
            return
        try:
            proof = _owned_proof(selected, owned)
        except (ValueError, TypeError, OverflowError):
            return
        if proof is None:
            return
        witness = references, proof
        with self._lock:
            rows = self._physical.setdefault(profile, [])
            if witness in rows:
                rows.remove(witness)
            rows.append(witness)
            del rows[:-self.max_per_pool]
            self._physical.move_to_end(profile)
            while len(self._physical) > self.max_pools:
                self._physical.popitem(last=False)

    def remap(self, profile, candidates, owned):
        """At most max_per_pool candidates, using only CURRENT pool indices."""
        if profile is None or len(candidates) > 5000:
            return ()
        with self._lock:
            if profile not in self._physical:
                return ()
            witnesses = tuple(self._physical[profile])
            self._physical.move_to_end(profile)
        try:
            witnesses = [w for w in witnesses if _matches_owned(w[1], owned)]
        except (TypeError, ValueError, OverflowError):
            return ()
        wanted = {reference[0] for references, _ in witnesses for reference in references}
        positions = {}
        for index, candidate in enumerate(candidates):
            if physical_key(candidate) not in wanted:
                continue
            reference = _reference(candidate)
            if reference is not None:
                positions.setdefault(reference, []).append(index)
        result = []
        for references, _ in witnesses:
            # Ambiguous representations are rejected, never guessed.
            if all(len(positions.get(ref, ())) == 1 for ref in references):
                indices = tuple(positions[ref][0] for ref in references)
                if len(set(indices)) == len(indices) and indices not in result:
                    result.append(indices)
        return tuple(result)


INCUMBENTS = Incumbents()
