from lost_in_injection.metrics import MARKDOWN_TOLERANT
from lost_in_injection.tables import body, extra_languages, seeds_ablations


def test_seed_cells_show_mean_and_spread():
    key = ("english-only", MARKDOWN_TOLERANT, "combined", "english", "asr")
    by_seed = {
        "0": {key: 0.075},
        "1": {key: 0.087},
        "2": {key: 0.099},
        "mean": {key: 0.087},
    }
    html = seeds_ablations(by_seed)
    assert "8.7 ± 1.2" in html  # Mean of the three seeds, sample SD
    assert "–" in html  # Held-out mean etc. have no data: missing, not zero


def test_single_seed_recipe_shows_a_plain_value_and_one_seed():
    key = ("no-input-role", MARKDOWN_TOLERANT, "combined", "english", "asr")
    html = seeds_ablations({"0": {key: 0.05}})
    assert '<td class="">1</td>' in html  # One seed, no spread to show
    assert "5.0 ±" not in html


def test_body_marks_missing_values_dash_and_out_of_the_best_choice():
    values = {"base": [1.0, float("nan")], "english-only": [2.0, 3.0]}
    html = body(["base", "english-only"], values, [min, min], set())
    assert '<td class="best ">1.0</td>' in html
    assert '<td class="">–</td>' in html
    assert '<td class="best ">3.0</td>' in html  # english-only's 3.0 is the only real value there


def test_extra_languages_table_is_skipped_without_data():
    assert extra_languages({}) == ""
