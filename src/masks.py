"""Token masks used by the constrained decoder."""

import re

import numpy as np

BOOLEANS = ("true", "false")
_SIMPLE_ESCAPES = '"\\/bfnrt'
_HEX_DIGITS = "0123456789abcdefABCDEF"


def pick_allowed_token(logits: list[float], allowed: list[int]) -> int:
    """Return the allowed token id with the highest logit.

    Raises:
        ValueError: If no token is allowed.
    """
    if not allowed:
        raise ValueError("No token allowed at this step")
    scores = np.array(logits)
    masked = np.full(len(logits), -np.inf)
    masked[allowed] = scores[allowed]
    return int(np.argmax(masked))


def valid_name_token_ids(
    generated: str,
    names: list[str],
    vocab: dict[int, str],
    quote_id: int,
) -> list[int]:
    """Return the ids of tokens that keep `generated` a prefix of a name."""
    allowed: list[int] = []
    for token_id, text in vocab.items():
        if not text:
            continue
        candidate = generated + text
        if any(name.startswith(candidate) for name in names):
            allowed.append(token_id)
    if generated in names:
        allowed.append(quote_id)
    return allowed


def is_number_prefix(text: str) -> bool:
    """Return True if `text` can still grow into a valid number."""
    return re.fullmatch(r"-?(\d+(\.\d*)?)?", text) is not None


def is_complete_number(text: str) -> bool:
    """Return True if `text` is a finished, valid number."""
    return re.fullmatch(r"-?\d+(\.\d+)?", text) is not None


def valid_number_token_ids(
    generated: str,
    vocab: dict[int, str],
    end_id: int,
) -> list[int]:
    """Return the ids of tokens that keep `generated` a valid number."""
    allowed: list[int] = []
    for token_id, text in vocab.items():
        if not text:
            continue
        if is_number_prefix(generated + text):
            allowed.append(token_id)
    if is_complete_number(generated):
        allowed.append(end_id)
    return allowed


def scan_json_string(text: str) -> tuple[bool, bool]:
    """Check `text` as the inside of a JSON string (without quotes).

    Returns:
        (valid, complete): `valid` is True if `text` can still grow into
        valid JSON string content, `complete` is True if it can end here.
    """
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"' or ord(char) < 0x20:
            return False, False
        if char != "\\":
            index += 1
            continue
        following = text[index + 1:index + 2]
        if following == "":
            return True, False
        if following in _SIMPLE_ESCAPES:
            index += 2
            continue
        if following != "u":
            return False, False
        hex_part = text[index + 2:index + 6]
        if any(h not in _HEX_DIGITS for h in hex_part):
            return False, False
        if len(hex_part) < 4:
            return True, False
        index += 6
    return True, True


def static_string_token_ids(vocab: dict[int, str]) -> list[int]:
    """Ids of the tokens valid as string content after a complete prefix.

    Computed once: when no escape sequence is pending, the validity of a
    token does not depend on the text written before it.
    """
    return [i for i, t in vocab.items() if t and scan_json_string(t)[0]]


def closing_string_tokens(
    vocab: dict[int, str],
) -> dict[int, tuple[str, str]]:
    """Tokens able to close a JSON string, computed once.

    A token qualifies if the text before its first double quote is
    complete string content on its own.

    Returns:
        token id -> (content before the quote, text after the quote).
    """
    closers: dict[int, tuple[str, str]] = {}
    for token_id, text in vocab.items():
        if '"' not in text:
            continue
        before, after = text.split('"', 1)
        if scan_json_string(before) == (True, True):
            closers[token_id] = (before, after)
    return closers


def valid_string_token_ids(
    generated: str,
    vocab: dict[int, str],
    static_ids: list[int],
    closers: dict[int, tuple[str, str]],
    follow: str,
) -> list[int]:
    """Return the ids of tokens allowed while writing a JSON string.

    Args:
        generated: Content of the string produced so far.
        vocab: Mapping token id -> token text.
        static_ids: Result of static_string_token_ids(vocab).
        closers: Result of closing_string_tokens(vocab).
        follow: JSON text that must come right after the closing quote.

    Returns:
        The ids of the allowed tokens.
    """
    if scan_json_string(generated)[1]:
        ends = [
            i for i, (_, after) in closers.items()
            if follow.startswith(after)
        ]
        return static_ids + ends
    return [
        i for i, t in vocab.items()
        if t and scan_json_string(generated + t)[0]
    ]


def valid_boolean_token_ids(
    generated: str,
    vocab: dict[int, str],
) -> list[int]:
    """Return the ids of tokens that keep `generated` a prefix of a boolean."""
    allowed: list[int] = []
    for token_id, text in vocab.items():
        if not text:
            continue
        candidate = generated + text
        if any(word.startswith(candidate) for word in BOOLEANS):
            allowed.append(token_id)
    return allowed
