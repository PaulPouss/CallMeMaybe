import numpy as np
import re
"""Token masks used by the constrained decoder."""


def valid_name_token_ids(
    generated: str,
    names: list[str],
    vocab: dict[int, str],
    quote_id: int,
) -> list[int]:
    """Return the ids of tokens that keep `generated` a prefix of a name.

    Args:
        generated: Text of the function name produced so far.
        names: All available function names.
        vocab: Mapping token id -> token text.
        quote_id: Id of the closing double-quote token, allowed once
            `generated` is a complete function name.

    Returns:
        The ids of the allowed tokens.
    """
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


def pick_allowed_token(logits: list[float], allowed: list[int]) -> int:
    """Return the allowed token id with the highest logit.

    Args:
        logits: One score per token id, as returned by the model.
        allowed: Ids of the tokens that may be chosen.

    Returns:
        The id of the best allowed token.

    Raises:
        ValueError: If no token is allowed.
    """
    scores = np.array(logits)
    forbidden = np.ones(len(logits), dtype=bool)
    forbidden[allowed] = False
    scores[forbidden] = -np.inf
    return int(np.argmax(scores))


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
    """Return the ids of tokens that keep `generated` a valid number.

    Args:
        generated: Text of the number produced so far.
        vocab: Mapping token id -> token text.
        end_id: Id of the token that closes the value (',' or '}'),
            allowed once `generated` is a complete number.

    Returns:
        The ids of the allowed tokens.
    """
    allowed: list[int] = []
    for token_id, text in vocab.items():
        if not text:
            continue
        if is_number_prefix(generated + text):
            allowed.append(token_id)
    if is_complete_number(generated):
        allowed.append(end_id)
    return allowed
