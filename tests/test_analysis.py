from itertools import combinations

import numpy as np

from lost_in_injection.analysis import (
    bootstrap_mean,
    fixed_order_curve,
    greedy_order,
    mean_pairwise_correlation,
    oracle_curve,
    random_order_curve,
)

# 4 languages x 6 cases: which language breaks which case.
MATRIX = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
    ],
    dtype=bool,
)


def test_random_order_matches_enumerating_every_subset():
    expected = [
        np.mean([MATRIX[list(s)].any(axis=0) for s in combinations(range(4), k)], axis=0)
        for k in range(1, 5)
    ]
    assert np.allclose(random_order_curve(MATRIX).T, expected)


def test_fixed_order_and_greedy():
    assert fixed_order_curve(MATRIX, [3, 0, 1, 2]).sum(axis=0).tolist() == [1, 2, 3, 5]
    assert greedy_order(MATRIX)[:2] == [1, 2]  # 2 new cases each; ties keep the variant order


def test_oracle_is_the_best_subset_of_each_size():
    matrix = np.random.default_rng(0).random((5, 150)) < 0.1  # More cases than a 64-bit mask
    best = [
        max(matrix[list(s)].any(axis=0).sum() for s in combinations(range(5), k))
        for k in range(1, 6)
    ]
    assert oracle_curve(matrix).sum(axis=0).tolist() == best


def test_bootstrap_mean_value_and_interval():
    per_case = np.random.default_rng(0).random((3, 200, 2)) < [0.2, 0.6]
    value, low, high = bootstrap_mean(per_case, np.random.default_rng(1))
    assert np.allclose(value, per_case.mean(axis=(0, 1)))
    assert np.all(low < value) and np.all(value < high)


def test_mean_pairwise_correlation_ignores_sets_that_never_vary():
    same = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0]], dtype=bool)
    assert mean_pairwise_correlation(same) == 1.0
    opposite = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=bool)
    assert mean_pairwise_correlation(opposite) == -1.0
    assert np.isnan(mean_pairwise_correlation(same[2:]))


def test_comprehension_adjusted_drops_languages_the_model_cannot_read():
    from lost_in_injection.analysis import comprehension_adjusted
    from lost_in_injection.metrics import EVALUATED

    cases = 20
    hit = np.ones((6, cases), bool)
    miss = np.zeros((6, cases), bool)
    # Undefended: obeys English and French, never Tamil (it cannot read it).
    base = {v: miss if v == "tamil" else hit for v in EVALUATED}
    # Defended: holds in English, fails in French, "holds" in Tamil only because it cannot read it.
    defended = {v: hit if v == "french" else miss for v in EVALUATED}
    hits = {("llama8b-base", "0"): base, ("llama8b-english-only", "0"): defended}
    rows = comprehension_adjusted(hits, {}, np.random.default_rng(0))
    by = {r["language"]: r for r in rows}
    assert by["tamil"]["reads_language"] is False and by["french"]["reads_language"] is True
    assert by["french"]["relative_reduction"] == 0.0
    assert by["union_read"]["defended_rate"] == 1.0 and by["excess_read"]["defended_rate"] == 1.0
    assert by["union_read"]["reads_language"] == len(EVALUATED) - 1
