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
