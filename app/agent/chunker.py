import re
from app.contracts import MAX_SPEAKABLE_CHARS

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟。])(?:\s+|(?=\S))")


def _split_long_word(word: str, max_chars: int) -> list[str]:
    return [word[i : i + max_chars] for i in range(0, len(word), max_chars)]


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:

    chunks: list[str] = []
    current = ""

    for word in sentence.split():
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            pieces = _split_long_word(word, max_chars)
            chunks.extend(pieces[:-1])
            current = pieces[-1]
            continue

        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = word

    if current:
        chunks.append(current)

    return chunks


def chunk_text(text: str,max_chars: int = MAX_SPEAKABLE_CHARS) -> list[str]:
    """Split generated_repsonse into TTS-friendly chunks, one sentence per chunk.
    Sentences longer than ``max_chars`` are split on word boundaries;
    single words longer than ``max_chars`` are hard-split.
    """
    if not text or not text.strip():
        return []

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[str] = []

    for sentence in SENTENCE_BOUNDARY.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) <= max_chars:
            chunks.append(sentence)
        else:
            chunks.extend(_split_long_sentence(sentence, max_chars))

    return chunks