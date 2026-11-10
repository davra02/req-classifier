"""Deterministic extraction of candidate requirement sentences from Markdown or plain text.

No model takes part: a system that is later evaluated on these sentences must not pick them.
Only markup is removed (code, HTML, emphasis, link syntax, list and table markers, leading
requirement IDs and short labels); the words are kept verbatim and whitespace is collapsed.
"""

import html
import re
from collections.abc import Iterator
from dataclasses import dataclass

from reqcore.normalize import dedup_key, normalize_text

MIN_WORDS = 6
MAX_WORDS = 60

MODAL = re.compile(r"\b(shall|must)\b", re.IGNORECASE)
_FENCE = re.compile(r"^(```|~~~)")
_HEADING = re.compile(r"^(#{1,6}\s|={3,}$|-{3,}$|\*{3,}$|_{3,}$)")
_LIST_ITEM = re.compile(r"^(?:[-*+]|\d{1,3}[.)])\s+(?:\[[ xX]\]\s+)?")
_TABLE_SEPARATOR = re.compile(r"^\|?[\s:|-]+\|?$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_EMPHASIS = re.compile(r"\*+|(?<!\w)_+|_+(?!\w)")
# "FR-01:", "[NFR-3.2]", "REQ_12 -", "3.2.1 " at the start of a unit.
_LEADING_ID = re.compile(
    r"^(?:\[?[A-Z]{1,8}[-_ ]?\d+(?:[.-]\d+)*\]?|\d+(?:\.\d+)+\.?)\s*[:.)\-–—]?\s+"
)
# "Description: The system shall ..." -> drop a label of at most four words.
_LEADING_LABEL = re.compile(r"^([A-Z][\w/()-]*(?: [\w/()-]+){0,3}):\s+")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"“(\[]?[A-Z0-9])")
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "approx.", "no.", "fig.", "dr.", "mr.", "ms.")

_BOILERPLATE = re.compile(
    r"RFC\s*2119|RFC\s*8174|key ?words|\"(MUST|SHALL)\"|lorem ipsum", re.IGNORECASE
)
_PLACEHOLDER = re.compile(r"[\[\]{}|]|\bTBD\b|\bTODO\b|\bXXX\b")


@dataclass(frozen=True)
class Candidate:
    path: str
    line: int
    text: str

    @property
    def location(self) -> str:
        return f"{self.path}#L{self.line}"


def _blocks(content: str) -> Iterator[tuple[int, str]]:
    """Yield (first line, text) for each paragraph, list item and table cell outside code."""
    content = _HTML_COMMENT.sub(lambda m: "\n" * m.group().count("\n"), content)
    buffer: list[str] = []
    start = 0
    in_fence = False
    for lineno, raw in enumerate(content.splitlines(), start=1):
        line = raw.strip()
        if _FENCE.match(line):
            in_fence = not in_fence
            line = ""
        if in_fence:
            continue
        if not line or _HEADING.match(line) or line.startswith("|"):
            if buffer:
                yield start, " ".join(buffer)
                buffer = []
            if line.startswith("|") and not _TABLE_SEPARATOR.match(line):
                for cell in line.strip("|").split("|"):
                    yield lineno, cell
            continue
        line = line.lstrip("> ").strip()
        item = _LIST_ITEM.match(line)
        if item or not buffer:
            if buffer:
                yield start, " ".join(buffer)
            buffer, start = [line[item.end() :] if item else line], lineno
        else:
            buffer.append(line)
    if buffer:
        yield start, " ".join(buffer)


def strip_markup(text: str) -> str:
    text = _IMAGE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _HTML_TAG.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _EMPHASIS.sub("", text)
    return normalize_text(html.unescape(text))


def strip_prefix(text: str) -> str:
    for _ in range(2):  # an ID can precede a label: "FR-1: Login: The system shall ..."
        text = _LEADING_ID.sub("", text)
        label = _LEADING_LABEL.match(text)
        if label and not MODAL.search(label.group(1)):
            text = text[label.end() :]
    return text


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for piece in _SENTENCE_END.split(text):
        if sentences and sentences[-1].lower().endswith(_ABBREVIATIONS):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return [s.strip() for s in sentences if s.strip()]


def is_candidate(sentence: str) -> bool:
    words = sentence.split()
    letters = [ch for ch in sentence if ch.isalpha()]
    return (
        MIN_WORDS <= len(words) <= MAX_WORDS
        and MODAL.search(sentence) is not None
        and (sentence[0].isupper() or sentence[0].isdigit())
        and MODAL.match(sentence) is None  # "Must link to ..." is a fragment without subject
        and not sentence.endswith(("?", ":"))
        and _BOILERPLATE.search(sentence) is None
        and _PLACEHOLDER.search(sentence) is None
        and bool(letters)
        and sum(ch.isascii() for ch in letters) / len(letters) >= 0.95
    )


def extract_candidates(path: str, content: str) -> list[Candidate]:
    """Candidate requirement sentences of one file, in document order, without exact repeats."""
    seen: set[str] = set()
    out: list[Candidate] = []
    for lineno, block in _blocks(content):
        for piece in split_sentences(strip_markup(block)):
            sentence = strip_prefix(piece)
            key = dedup_key(sentence)
            if key in seen or not is_candidate(sentence):
                continue
            seen.add(key)
            out.append(Candidate(path, lineno, sentence))
    return out
