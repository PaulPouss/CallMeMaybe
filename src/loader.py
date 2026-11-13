"""Reading and validation of the JSON input files."""

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .models import FunctionDef, PromptInput, FunctionCallResult

T = TypeVar("T", bound=BaseModel)


def read_json_list(path: Path) -> list[Any]:
    """Read a JSON file and check that its top level is a list.

    Args:
        path: Path to the JSON file.

    Returns:
        The decoded list. Its elements are not validated here.

    Raises:
        ValueError: If the file cannot be read, is not valid JSON,
            or does not contain a list at its top level.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path} "
            f"(line {error.lineno}, column {error.colno}): {error.msg}"
        ) from error
    except OSError as error:
        raise ValueError(f"Cannot read {path}: {error}") from error

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {path}, got {type(data).__name__}"
        )

    return data


def _load_models(path: Path, model: type[T]) -> list[T]:
    """Read a JSON list file and validate each element with a model.

    Args:
        path: Path to the JSON file.
        model: Pydantic model each element must satisfy.

    Returns:
        One validated model instance per element.

    Raises:
        ValueError: If the file is invalid or an element is malformed.
    """
    results: list[T] = []
    for index, item in enumerate(read_json_list(path)):
        try:
            results.append(model.model_validate(item))
        except ValidationError as error:
            raise ValueError(
                f"Invalid element #{index} in {path}: {error}"
            ) from error
    return results


def load_functions(path: Path) -> list[FunctionDef]:
    """Load and validate the function definitions."""
    return _load_models(path, FunctionDef)


def load_prompts(path: Path) -> list[PromptInput]:
    """Load and validate the test prompts."""
    return _load_models(path, PromptInput)


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write the function calls to a JSON file.

    Args:
        path: Output file path. Missing parent directories are created.
        results: The function calls to write.

    Raises:
        ValueError: If the file cannot be written.
    """
    data = [result.model_dump() for result in results]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    except OSError as error:
        raise ValueError(f"Cannot write {path}: {error}") from error
