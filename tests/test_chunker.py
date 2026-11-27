from app.agent.chunker import chunk_text
from app.contracts import MAX_SPEAKABLE_CHARS


def test_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text():
    assert chunk_text("Hello.") == ["Hello."]


def test_arabic_text():
    text = "أهلاً بيك. إزاي أقدر أساعدك؟"
    assert chunk_text(text) == [
        "أهلاً بيك.",
        "إزاي أقدر أساعدك؟",
    ]


def test_max_length():
    text = " ".join(["hello"] * 100)
    chunks = chunk_text(text)

    assert chunks
    assert all(len(chunk) <= MAX_SPEAKABLE_CHARS for chunk in chunks)


def test_long_word():
    text = "a" * 450
    chunks = chunk_text(text, max_chars=200)

    assert [len(chunk) for chunk in chunks] == [200, 200, 50]