import numpy as np

from app import _beam_search


def test_partial_danger_penalty_can_be_relaxed():
    # Use an accumulation above its actual danger threshold.
    danger = float(_beam_search._DANGER_THRESH[0])

    stats = np.zeros((1, 5), dtype=np.float64)
    stats[0, 0] = danger + 1.0

    wstats = np.zeros_like(stats)
    prices = np.zeros(1, dtype=np.float64)

    common = dict(
        stats=stats,
        wstats=wstats,
        prices=prices,
        min_budget=False,
        hp_idx=-1,
        bul_idx=-1,
        ehp_w=0.0,
        max_ehp=0.0,
        prot_mult=1.0,
        n_b=0,
        b_idx=np.zeros(0, dtype=np.int32),
        b_val=np.zeros(0, dtype=np.float64),
        tear_idx=-1,
        stop_idx=-1,
        stop_w=0.0,
        has_tear=False,
        m_idx=np.zeros(0, dtype=np.int32),
        m_val=np.zeros(0, dtype=np.float64),
        n_m=0,
        mx_idx=np.zeros(0, dtype=np.int32),
        mx_val=np.zeros(0, dtype=np.float64),
        n_mx=0,
    )

    strict = _beam_search._score_vec(
        **common,
        ignore_danger_penalty=False,
    )

    relaxed = _beam_search._score_vec(
        **common,
        ignore_danger_penalty=True,
    )

    assert strict.shape == (1,)
    assert relaxed.shape == (1,)

    # Exceeding the threshold by exactly 1.0 adds a 1000-point penalty.
    np.testing.assert_allclose(
        relaxed - strict,
        [1000.0],
        atol=1e-8,
    )
