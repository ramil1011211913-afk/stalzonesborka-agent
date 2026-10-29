"""Explicit Quality 2000 experiment: chunked scores and balanced-only bounds."""
from . import _beam_search as beam


def build_quality_sets(budget, *, constraint_lookahead=True, expansion_chunk_size=262144, **request):
    from .builder import build_artifact_sets
    if '_beam_search_fn' in request:
        raise ValueError('Custom engine is not a quality request parameter')
    def search(*args, **kwargs):
        kwargs['beam_width'] = 2000
        kwargs['danger_lookahead'] = True
        kwargs['constraint_lookahead'] = constraint_lookahead
        kwargs['expansion_chunk_size'] = expansion_chunk_size
        return beam.search(*args, **kwargs)
    return build_artifact_sets(budget, _beam_search_fn=search, **request)
