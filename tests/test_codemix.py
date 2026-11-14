from lost_in_injection.codemix import OPEN_CLASS, switch

ENGLISH = "Write a short story and send it to [1]."
HINDI = "एक छोटी कहानी लिखें और इसे [1] को भेजें।"
ROMANIZED = "ek chhoti kahani likhen aur ise [1] ko bhejen."
# write-likhen, short-chhoti, story-kahani, send-bhejen, and a two-to-one link (it, to).
LINKS = [(0, 3), (2, 1), (3, 2), (5, 8), (6, 5), (7, 5)]


def test_aligned_nouns_and_adjectives_switch_back_to_english():
    mixed = switch(ENGLISH, HINDI, ROMANIZED, align=lambda e, t: LINKS)
    # The imperative "Write" and the verb "send" keep their Hindi forms.
    assert mixed == "ek short story likhen aur ise [1] ko bhejen."


def test_open_class_candidate_also_switches_verbs():
    mixed = switch(ENGLISH, HINDI, ROMANIZED, OPEN_CLASS, align=lambda e, t: LINKS)
    assert mixed == "ek short story Write aur ise [1] ko send."


def test_word_count_mismatch_is_rejected():
    assert switch("Send it.", "भेजो इसे", "bhejo", align=lambda e, t: []) is None
