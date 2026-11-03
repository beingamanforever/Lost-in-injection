from lost_in_injection.codemix import switch


def test_aligned_nouns_and_adjectives_switch_back_to_english():
    english = "Write a short story and send it to [1]."
    hindi = "एक छोटी कहानी लिखें और इसे [1] को भेजें।"
    romanized = "ek chhoti kahani likhen aur ise [1] ko bhejen."
    # write-likhen, short-chhoti, story-kahani, send-bhejen, and a two-to-one link (it, to).
    links = [(0, 3), (2, 1), (3, 2), (5, 8), (6, 5), (7, 5)]
    mixed = switch(english, hindi, romanized, align=lambda e, t: links)
    # The imperative "Write" and the verb "send" keep their Hindi forms.
    assert mixed == "ek short story likhen aur ise [1] ko bhejen."


def test_word_count_mismatch_is_rejected():
    assert switch("Send it.", "भेजो इसे", "bhejo", align=lambda e, t: []) is None
