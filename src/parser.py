"""Command-line interface: declares, parses and validates arguments."""

import argparse
from pathlib import Path

from pydantic import BaseModel, ValidationError, field_validator

from .config import (
    AVAILABLE_MODELS,
    DEFAULT_FUNCTIONS,
    DEFAULT_INPUT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    DEFAULT_VERBOSE,
)


class CliConfig(BaseModel):
    """Validated command-line configuration."""

    functions_definition: Path
    input: Path
    output: Path
    verbose: bool
    model: str

    @field_validator("functions_definition", "input")
    @classmethod
    def _must_be_existing_file(cls, path: Path) -> Path:
        """Check that an input path exists and is a regular file."""
        if not path.exists():
            raise ValueError(f"Input file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Expected a file: {path}")
        return path

    @field_validator("output")
    @classmethod
    def _must_not_be_directory(cls, path: Path) -> Path:
        """Check that the output path is not an existing directory."""
        if path.is_dir():
            raise ValueError(f"Output path cannot be a directory: {path}")
        return path


def _create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all options declared."""
    parser = argparse.ArgumentParser(
        description="Function calling with Qwen 0.6B"
    )
    parser.add_argument(
        "--functions_definition", type=Path, default=DEFAULT_FUNCTIONS,
        help="Path to the functions definition JSON file.",
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=DEFAULT_VERBOSE,
        help="Print each result as it is generated.",
    )
    parser.add_argument(
        "--model", choices=AVAILABLE_MODELS, default=DEFAULT_MODEL,
        help="Model to load.",
    )
    return parser


def parse_arguments(arguments: list[str] | None = None) -> CliConfig:
    """Parse the command line and validate it.

    Args:
        arguments: Arguments to parse. Defaults to sys.argv.

    Returns:
        The validated configuration.

    Raises:
        ValueError: If a path is unusable.
    """
    namespace = _create_parser().parse_args(arguments)
    try:
        return CliConfig(**vars(namespace))
    except ValidationError as error:
        messages = [
            str(item["msg"]).removeprefix("Value error, ")
            for item in error.errors()
        ]
        raise ValueError("; ".join(messages)) from error
