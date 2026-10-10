"""Text normalization (rule P-01) and stable hashing of requirement text."""

import hashlib
import re
import unicodedata

# Zero-width and bidi control characters often used to hide or obfuscate text.
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿]")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFKC-normalize, drop control and zero-width characters, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = "".join(ch if unicodedata.category(ch)[0] != "C" else " " for ch in text)
    return _WHITESPACE.sub(" ", text).strip()


def text_hash(text: str) -> str:
    """SHA-256 of the normalized text; the identity of a requirement across datasets."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def dedup_key(text: str) -> str:
    """Aggressive key for duplicate detection: lowercase alphanumerics and single spaces."""
    lowered = normalize_text(text).lower()
    return _WHITESPACE.sub(" ", re.sub(r"[^a-z0-9 ]", " ", lowered)).strip()
