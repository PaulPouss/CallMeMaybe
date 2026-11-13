"""Entry point: uv run python -m src."""

import sys

from llm_sdk.llm_sdk import Small_LLM_Model

from .generator import Generator
from .loader import load_functions, load_prompts, write_results
from .models import FunctionCallResult
from .parser import parse_arguments
from .vocabulary import load_vocab


def main() -> int:
    """Run the program.

    Returns:
        0 if every prompt succeeded, 1 otherwise.
    """
    try:
        config = parse_arguments()
        functions = load_functions(config.functions_definition)
        prompts = load_prompts(config.input)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    try:
        llm = Small_LLM_Model(model_name=config.model)
        vocab = load_vocab(llm.get_path_to_vocab_file())
        generator = Generator(llm=llm, functions=functions, vocab=vocab)
    except Exception as error:
        print(f"Error: cannot load the model: {error}", file=sys.stderr)
        return 1

    results: list[FunctionCallResult] = []
    failures = 0
    for item in prompts:
        try:
            result = generator.build_call(item.prompt)
        except Exception as error:
            failures += 1
            print(f"Warning: skipped {item.prompt!r}: {error}",
                  file=sys.stderr)
            continue
        results.append(result)
        if config.verbose:
            print(result.model_dump_json())

    try:
        write_results(config.output, results)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"{len(results)}/{len(prompts)} prompts written to {config.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
