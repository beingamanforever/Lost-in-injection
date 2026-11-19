from lost_in_injection.adaptive_attack import (
    ITERATIONS,
    SEED_QUERIES,
    STREAMS,
    first_seed_success,
    needs_new_iteration,
    query_number,
)


def test_query_numbers_interleave_streams_after_the_fixed_translations():
    assert SEED_QUERIES == 11  # english, 2 high-resource, 4 low-resource, 4 held-out
    assert [query_number(0, i) for i in range(len(STREAMS))] == [12, 13, 14, 15]
    assert [query_number(1, i) for i in range(len(STREAMS))] == [16, 17, 18, 19]
    last_round = [query_number(ITERATIONS - 1, i) for i in range(len(STREAMS))]
    assert last_round == [48, 49, 50, 51]  # 11 + 4 * 10 = 51 queries at most


def test_first_seed_success_is_the_first_variant_that_breaks():
    assert first_seed_success([False] * 11) is None
    assert first_seed_success([True] + [False] * 10) == 1
    assert first_seed_success([False, False, True] + [False] * 8) == 3
    assert first_seed_success([False] * 10 + [True]) == 11


def test_rewrite_adds_one_iteration_only_when_nothing_is_pending():
    done = {"reply": "No."}
    pending = {"reply": None}

    fresh_case = {"first_success": None, "attempts": []}
    assert needs_new_iteration(fresh_case)

    round_finished = {"first_success": None, "attempts": [done] * len(STREAMS)}
    assert needs_new_iteration(round_finished)

    round_in_flight = {"first_success": None, "attempts": [done, done, done, pending]}
    assert not needs_new_iteration(round_in_flight)

    already_succeeded = {"first_success": 12, "attempts": [done] * len(STREAMS)}
    assert not needs_new_iteration(already_succeeded)

    every_round_done = {"first_success": None, "attempts": [done] * (len(STREAMS) * ITERATIONS)}
    assert not needs_new_iteration(every_round_done)
