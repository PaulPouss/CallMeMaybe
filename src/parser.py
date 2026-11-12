"""Command-line interface: declares, parses and validates arguments."""

import argparse
from pathlib import Path

from .config import (
    AVAILABLE_MODELS,
    DEFAULT_FUNCTIONS,
    DEFAULT_INPUT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    DEFAULT_VERBOSE,
)


class Parser:
    """Declare, parse and validate the command-line arguments."""

    def __init__(self) -> None:
        """Build the underlying argparse parser once."""
        self._parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description="Function calling with Qwen 0.6B"
        )
        self._add_config_arguments(parser)
        return parser

    @staticmethod
    def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
        """Add configuration-related command-line arguments."""
        parser.add_argument(
            "--functions_definition",
            type=Path,
            default=DEFAULT_FUNCTIONS,
            help="Path to the functions definition JSON file.",
        )
        parser.add_argument(
            "--input",
            type=Path,
            default=DEFAULT_INPUT,
            help="Path to the input JSON file.",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=DEFAULT_OUTPUT,
            help="Path to the output JSON file.",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            default=DEFAULT_VERBOSE,
            help="Enable verbose output.",
        )
        parser.add_argument(
            "--model",
            choices=AVAILABLE_MODELS,
            default=DEFAULT_MODEL,
            help="Model to load.",
        )

    @staticmethod
    def validate_existing_file(path: Path) -> None:
        """Validate that a path exists and refers to a file.

        Args:
            path: Path to validate.

        Raises:
            ValueError: If the path does not exist or is not a file.
        """
        if not path.exists():
            raise ValueError(f"Input file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Expected a file: {path}")

    @staticmethod
    def validate_output_file(path: Path) -> None:
        """Validate an output file path.

        Ensures that the output path is not an existing directory and
        creates the parent directories if they do not exist.

        Args:
            path: Output file path.

        Raises:
            ValueError: If the output path is an existing directory.
        """
        if path.exists() and path.is_dir():
            raise ValueError(f"Output path cannot be a directory: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)

    def parse(self, arguments: list[str] | None = None) -> argparse.Namespace:
        """Parse the command line and validate the file paths.

        Args:
            arguments: Arguments to parse. Defaults to sys.argv.

        Returns:
            The parsed and validated arguments.

        Raises:
            ValueError: If an input path is unusable.
        """
        args = self._parser.parse_args(arguments)
        self.validate_existing_file(args.functions_definition)
        self.validate_existing_file(args.input)
        self.validate_output_file(args.output)
        return args