from pathlib import Path

DATA_DIR = Path("data")

INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

DEFAULT_FUNCTIONS = INPUT_DIR / "functions_definition.json"
DEFAULT_INPUT = INPUT_DIR / "function_calling_tests.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "function_calling_results.json"

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
