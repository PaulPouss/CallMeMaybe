*This project has been created as part of the 42 curriculum by ppousser.*

# call me maybe

## Description

**call me maybe** is a function-calling tool: it translates a natural-language
request into a structured, machine-executable function call, using a small
language model (Qwen/Qwen3-0.6B).

Given the request `"What is the sum of 2 and 3?"`, the program does not answer
`5`. It produces the call that would compute it:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

Small models are unreliable at producing structured output when simply asked
to. This project solves that with **constrained decoding**: at every
generation step, the tokens that would break the JSON structure or the
function schema are removed before the model picks the next token. The output
is therefore valid JSON that matches the schema by construction, while the
model keeps full freedom over *which* function to call and *which* values to
extract.

The available functions are read from `functions_definition.json`, the
requests from `function_calling_tests.json`, and the results are written to
`function_calling_results.json`. Nothing about the functions or the requests
is hardcoded.

### Project structure

```
.
├── src/
│   ├── __main__.py     # entry point: orchestrates the run, handles errors
│   ├── config.py       # default paths, model list, length limits
│   ├── parser.py       # command-line arguments, validated by a Pydantic model
│   ├── models.py       # Pydantic models: FunctionDef, PromptInput, FunctionCallResult
│   ├── loader.py       # reading and validating input files, writing the output
│   ├── vocabulary.py   # token id -> token text mapping
│   ├── masks.py        # constrained-decoding rules (which tokens are allowed)
│   └── generator.py    # the generation loop driving the model
├── llm_sdk/            # provided SDK (unmodified)
├── data/input/         # example input files
├── Makefile
├── pyproject.toml
└── uv.lock
```

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)
- About 1.5 GB of free disk space for the model, plus the Python
  dependencies of the SDK (several GB)

### Installation

```bash
make install        # equivalent to: uv sync
```

The model is downloaded automatically on the first run.

If your home directory is small (as on some school machines), move the caches
and the virtual environment to a larger disk before installing:

```bash
export UV_CACHE_DIR=/path/to/large/disk/uv-cache
export HF_HOME=/path/to/large/disk/huggingface
export UV_PROJECT_ENVIRONMENT=/path/to/large/disk/venv
```

### Running

```bash
make run            # equivalent to: uv run python -m src
```

| Option | Default | Description |
|---|---|---|
| `--functions_definition` | `data/input/functions_definition.json` | Functions available to the model |
| `--input` | `data/input/function_calling_tests.json` | Requests to process |
| `--output` | `data/output/function_calling_results.json` | Output file (parent directories are created) |
| `--model` | `Qwen/Qwen3-0.6B` | Model to load (see `src/config.py` for the list) |
| `--verbose` | off | Print each result as soon as it is generated |

### Other Makefile rules

| Rule | Action |
|---|---|
| `make debug` | Run the program under `pdb` |
| `make lint` | `flake8 .` and `mypy .` with the flags required by the subject |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |
| `make clean` | Remove `__pycache__` and `.mypy_cache` |

## Example usage

Default paths:

```bash
uv run python -m src
```

Custom paths, with each result printed:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json \
    --verbose
```

Output (excerpt):

```json
[
  {
    "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "Hello 34 I'm 233 years old",
      "regex": "\\d+",
      "replacement": "NUMBERS"
    }
  }
]
```

Errors are reported in one line, without a traceback:

```
$ uv run python -m src --input missing.json
Error: Input file not found: missing.json

$ uv run python -m src --input broken.json
Error: Invalid JSON in broken.json (line 1, column 2): Expecting property name enclosed in double quotes
```

## Algorithm explanation

### Overview

For each request, the program builds a short prompt that lists the available
functions (name and description), shows two examples of complete calls using
a fictional function, then states the request and ends with the beginning of
the answer:

```
Function call: {"name": "
```

From there, generation alternates between two kinds of steps:

- **Fixed structure**, written directly by the program: `", "parameters": {`,
  the parameter keys, the separators. These parts are fully determined by the
  schema, so the model is never asked for them.
- **Free parts**, chosen by the model under a mask: the function name and each
  parameter value.

### The masking step

The SDK returns one logit per token id (151 936 for Qwen3-0.6B). For each free
step, a mask function returns the list of allowed token ids, and the next
token is chosen as follows:

1. build an array filled with `-inf`;
2. copy the model's logits into it at the allowed ids only;
3. take the `argmax` (greedy decoding).

A forbidden token keeps a logit of `-inf` and can never be selected, whatever
the model "wanted". This is what guarantees the structure.

### Vocabulary

`vocab.json` maps token text to token id. It is loaded once and inverted into
`id -> text`. The byte-level marker `Ġ` is replaced by a space, so that token
texts can be compared with ordinary strings. Token ids that do not appear in
the file (special tokens) have no known text and are therefore never allowed.

### Function name

A token is allowed if the text generated so far, followed by the token's text,
is still the beginning of at least one available function name. When the text
is exactly a complete name, the closing `"` is allowed too, so the model can
stop there (or continue if a longer name shares the same beginning, such as
`fn_get` and `fn_get_square_root`).

### Values, by type

The type of each parameter is read from the function definition.

- **number**: a token is allowed if the text stays a valid number prefix
  (regex `-?(\d+(\.\d*)?)?`). The closing token (`,` or `}`, depending on
  whether more parameters follow) is allowed only once the text is a complete
  number (`-?\d+(\.\d+)?`), which rejects `-` or `3.`. The value is converted
  to `float`.
- **string**: the content is checked by a small scanner that follows the JSON
  string rules: no raw double quote, no control character, and every `\` must
  start a valid escape sequence (`\"`, `\\`, `\n`, `\uXXXX`...). The string
  can only be closed when no escape sequence is pending. The generated text is
  then decoded with `json.loads`, so `\\d+` becomes the regex `\d+`.
- **boolean**: a token is allowed if the text stays a prefix of `true` or
  `false`. Generation stops as soon as one of them is complete, since nothing
  can extend it.

### Termination

Every loop is guaranteed to stop. Numbers are limited to 24 characters and
strings to 200: once the limit is reached and the value is complete, only the
closing token is allowed; if an escape sequence or a number is unfinished, only
single-character tokens are allowed, which can only complete it.

## Design decisions

- **One responsibility per module.** File reading (`loader.py`), data shapes
  (`models.py`), masking rules (`masks.py`) and the generation loop
  (`generator.py`) are separate. The masking rules are plain functions: they
  can be tested without loading the model.
- **Pydantic for every class**, as required: input files, command-line
  configuration, results, and the generator itself. Invalid input is rejected
  with a clear message before the model is loaded.
- **Validation before loading the model.** Paths and file contents are checked
  first, because loading the model takes several seconds.
- **Fixed JSON structure written directly.** The subject describes masking at
  every position, including braces and keys. At those positions only one
  token is valid, so the result would be identical; writing them directly
  avoids one model call per structural token. The output is the same, and
  faster.
- **No PyTorch in the project code.** `encode` returns a tensor, converted to
  a Python list immediately with `.tolist()`. The SDK's
  `get_logits_from_input_ids` expects a tensor, which is built with the
  `new_tensor` method of a tensor returned by the SDK itself. The project
  never imports `torch`, and only public SDK methods are used.
- **Greedy decoding.** The goal is a deterministic, reproducible answer, not
  varied text, so the best allowed token is always taken.
- **Fictional examples in the prompt.** The two few-shot examples use an
  invented function (`fn_example_replace`), so the prompt does not depend on
  the provided test files and does not bias the choice toward a real
  function. The name mask makes it impossible for the model to output the
  invented name.
- **One failed request does not stop the run.** If a request fails, a warning
  is printed, the other results are still written, and the exit code is 1.
- **Precomputation at startup.** The tokens that are valid inside a string
  (when no escape is pending) and the tokens able to close a string are
  computed once, from the vocabulary, instead of at every step.

## Performance analysis

Measured on the provided example files (11 requests, 5 functions):

| Metric | Result |
|---|---|
| Valid JSON | 100 %, by construction (and checked by Pydantic before writing) |
| Function selection | 11/11 |
| Argument extraction | 11/11 exact, with one debatable value (`"**"` as the replacement for "asterisks", where `"*"` would be expected) |
| Generation time | about 15 s for the 11 requests on a laptop, excluding model loading |

The main cost is the model call: the SDK recomputes the whole sequence at
every token, so time grows with the prompt length and with the number of
generated tokens. Mask computation is negligible thanks to the precomputed
token lists. Speed depends heavily on the hardware (GPU or CPU).

## Challenges faced

- **The SDK does not accept a list.** The subject documents
  `get_logits_from_input_ids(input_ids: List[int])`, but the provided SDK calls
  `.to()` on its argument, so it needs a tensor. Solved with `new_tensor`, as
  described above, without importing `torch` or modifying the SDK.
- **A very slow string mask.** The first version re-scanned the whole value for
  each of the ~150 000 tokens at every step, so long strings seemed to hang.
  Observation: when no escape sequence is pending, the validity of a token does
  not depend on what was written before. The list of valid tokens is now
  computed once at startup.
- **Values that never ended.** Regex values looped
  (`\d+\s+\d+\s+\d+...`) until the length limit. Printing the tokenization of
  the prompt examples showed the cause: the tokenizer merges the closing quote
  with its neighbors (`+",`), and the mask only allowed a lone `"`, a token the
  model had almost never seen at that position. The model wanted to stop but
  was not allowed to. Tokens made of complete content, a quote, then the
  beginning of what must follow in the JSON are now accepted as closing
  tokens. Total time dropped from 51 s to about 15 s, and all regexes became
  correct.
- **Escape sequences.** A regex such as `\d+` must be written `\\d+` in JSON.
  Forbidding backslashes would have made such values impossible, so the string
  mask follows the JSON escape rules instead.
- **Disk space.** The SDK dependencies and the model did not fit in a 5 GB home
  directory; caches and the virtual environment were moved with environment
  variables (see Instructions).

## Known limitations

- **Non-ASCII characters.** The vocabulary is byte-level: characters such as
  `é` are stored as byte sequences. Generating them in a string value may
  produce altered text.
- **Supported types.** `number`, `string` and `boolean`. Any other type is
  reported as an error for the request concerned.
- **Leading zeros.** A number such as `-00` is accepted by the mask; it is
  harmless because the value is converted to `float` before being written.
- **Long function lists.** All functions are listed in the prompt, so many
  functions make every model call slower.

## Testing strategy

- **Input validation**: each error case was run and checked to produce a
  single clear message and no traceback: missing file, directory instead of
  file, invalid JSON, JSON object instead of list, missing key, unknown
  command-line option, invalid `--model` value.
- **SDK exploration**: a throwaway script checked the shape of `encode`, the
  number of logits, the vocabulary format and the meaning of specific ids
  before any masking code was written.
- **Masks in isolation**: each mask was run on chosen prefixes and its allowed
  tokens printed, for example `""` → `f`, `fn`; `"fn_g"` → only tokens leading
  to `fn_greet` or `fn_get_square_root`; `"3."` → digits only, no closing
  token. The JSON string scanner was checked on eight cases (plain text,
  raw quote, pending backslash, escaped quote, invalid `\d`, valid `\\d`,
  incomplete and complete `\uXXXX`).
- **End to end**: the provided requests were run and the output file checked
  against the schema (keys, types, no extra content).
- **Static checks**: `make lint` (flake8 and mypy).

## Resources

- JSON grammar: [json.org](https://www.json.org/json-en.html) and
  [RFC 8259](https://datatracker.ietf.org/doc/html/rfc8259)
- Model: [Qwen/Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
- Tokenization (BPE):
  [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- Constrained generation:
  [Willard & Louf, *Efficient Guided Generation for Large Language Models*](https://arxiv.org/abs/2307.09702)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [NumPy documentation](https://numpy.org/doc/)
- Python [`re`](https://docs.python.org/3/library/re.html) and
  [`argparse`](https://docs.python.org/3/library/argparse.html) documentation

### Use of AI

Claude (Anthropic) was used throughout the project as a tutor and pair
programmer:

- **Understanding**: explanations of logits, constrained decoding, BPE
  tokenization, Pydantic, regular expressions and JSON escaping, usually by
  question and answer before writing code.
- **Design**: discussion of the module structure and of the generation
  pipeline.
- **Code**: parts of the code were written or completed with AI assistance,
  in particular the mask functions, the JSON string scanner, the generator
  class, the file loader and the Pydantic version of the command-line parser.
- **Debugging**: diagnosis of the disk-space issue, of the SDK expecting a
  tensor, of the slow string mask and of the tokenization problem behind the
  looping values.

Every piece of code was run, tested on the cases listed above, and reviewed
before being kept.
