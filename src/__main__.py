"""Entry point: uv run python -m src."""

import sys

from .loader import load_functions, load_prompts
from .parser import Parser


def main() -> int:
    """Run the program.

    Returns:
        0 on success, 1 on a handled error.
    """
    try:
        args = Parser().parse()
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"{len(functions)} functions, {len(prompts)} prompts loaded")

    # Next modules: load the model, generate, write the results.
    return 0


if __name__ == "__main__":
    sys.exit(main())