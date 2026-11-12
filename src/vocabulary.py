import json


def load_vocab(vocab_path: str) -> dict[int, str]:
    """Return a mapping token id -> token text.

    Raises:
        ValueError: If the vocabulary file is unreadable or malformed.
    """
    try:
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)

    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {vocab_path}: "
                         f"{error.msg}") from error

    except OSError as error:
        raise ValueError(f"Cannot read {vocab_path}: {error}") from error

    if not isinstance(vocab, dict):
        raise ValueError(
            f"Expected a JSON object in {vocab_path}, "
            f"got {type(vocab).__name__}"
        )

    inverse: dict[int, str] = {
        i: t.replace("Ġ", " ") for t, i in vocab.items()
    }
    return inverse
