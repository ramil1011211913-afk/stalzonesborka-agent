"""Request-local physical availability, independent of artifact type repeats."""
import numpy as np

AVAILABILITY_CONTRACT = "physical-v1"


def owned_identity(record, index):
    identity = record.get("instance_id")
    if identity is None:
        identity = record.get("owned_instance_id")
    if identity is None:
        return str(index), f"owned:position:{index}"
    return str(identity), f"owned:input:{identity}"


def physical_key(option):
    # The source row ID takes precedence over a derived/display identity.
    lot_id = getattr(option, "auction_lot_id", None)
    if lot_id is not None:
        return ("auction", str(lot_id))
    identity = getattr(option, "physical_id", None)
    if identity is not None:
        return ("physical", str(identity))
    owned_id = getattr(option, "owned_instance_id", None)
    if owned_id is not None:
        return ("owned", str(owned_id))
    # Anonymous programmatic candidates are distinct objects, not interchangeable
    # copies of an auction record. Production loaders always supply explicit IDs.
    return ("anonymous", id(option))


def conflict_ids(candidates, allow_duplicates):
    """Encode pair conflicts as one group ID, after checking source consistency.

    One physical object has one item_id. Therefore type uniqueness subsumes
    physical uniqueness when type repeats are forbidden.
    """
    types = {}
    groups = {}
    ids = np.empty(len(candidates), dtype=np.int32)
    for i, option in enumerate(candidates):
        physical = physical_key(option)
        if physical in types and types[physical] != option.item_id:
            raise ValueError("A physical artifact cannot have different item_id values")
        types[physical] = option.item_id
        key = physical if allow_duplicates else ("type", option.item_id)
        if key not in groups:
            groups[key] = len(groups)
        ids[i] = groups[key]
    return ids


def is_available(combo, allow_duplicates=True):
    physical = [physical_key(option) for option in combo]
    return (len(set(physical)) == len(physical)
            and (allow_duplicates or len({o.item_id for o in combo}) == len(combo)))


def availability_groups(candidates, allow_duplicates=True):
    """Groups for exact-model inequalities sum(x[group]) <= 1."""
    conflict_ids(candidates, True)  # Validate that each object has one item type.
    physical, types = {}, {}
    for i, option in enumerate(candidates):
        physical.setdefault(physical_key(option), []).append(i)
        types.setdefault(option.item_id, []).append(i)
    groups = list(physical.values())
    if not allow_duplicates:
        groups += list(types.values())
    return tuple(dict.fromkeys(tuple(group) for group in groups if len(group) > 1))
