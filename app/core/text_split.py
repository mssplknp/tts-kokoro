"""Split free text into synthesis-friendly sub-units.

The agent already sends one sentence per request, but sentences can be long.
Kokoro synthesizes each unit in a single model call, so smaller units mean:

* a lower time-to-first-byte (audio streams per unit),
* cancellation can take effect between units,
* output stays well under Kokoro's 510-phoneme limit (avoids truncation).

Boundaries are chosen at natural pause points so prosody is preserved: always
split on sentence-final punctuation, and split on clause punctuation only when
the running segment is long enough that we are not chopping up every phrase.
"""

_SENTENCE_FINAL = {"।", ".", "!", "?", "…"}
_CLAUSE = {",", ";", ":", "—", "–"}


def split_for_synthesis(
    text: str,
    max_chars: int = 200,
    min_clause_chars: int = 30,
) -> list[str]:
    """Split ``text`` into segments suitable for one Kokoro synthesis call.

    ``max_chars`` caps segment length; ``min_clause_chars`` prevents splitting
    on every comma when a clause boundary appears very early. Punctuation is
    kept attached to the segment it terminates. Empty segments are dropped.
    """
    if not text:
        return []

    segments: list[str] = []
    buf: list[str] = []
    length = 0

    for char in text:
        buf.append(char)
        length += 1

        if char in _SENTENCE_FINAL:
            segments.append("".join(buf).strip())
            buf, length = [], 0
        elif char in _CLAUSE and length >= min_clause_chars:
            segments.append("".join(buf).strip())
            buf, length = [], 0
        elif length >= max_chars:
            joined = "".join(buf)
            split_at = max(
                joined.rfind(" "), joined.rfind("\t"), joined.rfind("\n")
            )
            if split_at > 0:
                segments.append(joined[:split_at].strip())
                remainder = joined[split_at + 1:]
                buf, length = list(remainder), len(remainder)
            else:
                # Single token longer than the cap: hard break.
                segments.append(joined.strip())
                buf, length = [], 0

    if buf:
        segments.append("".join(buf).strip())

    return [s for s in segments if s]
