"""Default values and constants for the command-line interface."""

from pathlib import Path

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")
DEFAULT_VERBOSE = False
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
AVAILABLE_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]
MAX_NUMBER_LENGTH = 24
MAX_STRING_LENGTH = 200
