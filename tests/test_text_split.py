"""Unit tests for sentence/sub-unit splitting."""

from app.core.text_split import split_for_synthesis


def test_empty_and_blank():
    assert split_for_synthesis("") == []
    assert split_for_synthesis("   \n  ") == []


def test_sentence_final_punctuation_always_splits():
    text = "पहला वाक्य। दूसरा वाक्य! तीसरा? और चौथा."
    assert split_for_synthesis(text) == [
        "पहला वाक्य।",
        "दूसरा वाक्य!",
        "तीसरा?",
        "और चौथा.",
    ]


def test_short_clause_boundary_does_not_split():
    # Commas inside short phrases are kept together.
    assert split_for_synthesis("हाँ, ठीक है।") == ["हाँ, ठीक है।"]


def test_long_clause_boundary_splits():
    text = "अगर पत्तियां हल्की पीली दिखाई दे रही हैं, तो नाइट्रोजन की कमी हो सकती है।"
    parts = split_for_synthesis(text)
    assert len(parts) == 2
    assert parts[0].endswith(",")
    assert parts[1].endswith("।")
    # No text is lost.
    assert "".join(parts).replace(" ", "") == text.replace(" ", "")


def test_hard_cap_splits_long_unpunctuated_text():
    text = "श" * 500
    parts = split_for_synthesis(text, max_chars=100, min_clause_chars=100)
    assert len(parts) == 5
    assert all(len(p) <= 100 for p in parts)
    assert "".join(parts) == text


def test_hard_cap_breaks_at_last_space():
    words = ["शब्द" * 40] * 12  # each "word" 80 chars
    text = " ".join(words)
    parts = split_for_synthesis(text, max_chars=200, min_clause_chars=200)
    assert len(parts) > 1
    for part in parts[:-1]:
        assert not part.endswith(" ")  # no dangling space
    # No character is lost (whitespace at split boundaries is dropped).
    assert "".join(parts).replace(" ", "") == text.replace(" ", "")
