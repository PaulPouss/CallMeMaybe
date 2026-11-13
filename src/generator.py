import json
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from llm_sdk.llm_sdk import Small_LLM_Model

from .config import MAX_NUMBER_LENGTH, MAX_STRING_LENGTH
from .masks import (
    BOOLEANS,
    is_complete_number,
    pick_allowed_token,
    scan_json_string,
    valid_boolean_token_ids,
    valid_name_token_ids,
    valid_number_token_ids,
    valid_string_token_ids,
    static_string_token_ids,
    closing_string_tokens,
)
from .models import FunctionCallResult, FunctionDef


class Generator(BaseModel):
    """Generate function calls token by token under constraint."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Small_LLM_Model
    functions: list[FunctionDef]
    vocab: dict[int, str]

    _quote_id: int = PrivateAttr(default=-1)
    _comma_id: int = PrivateAttr(default=-1)
    _brace_id: int = PrivateAttr(default=-1)
    _closers: dict[int, tuple[str, str]] = PrivateAttr(default_factory=dict)
    _string_ids: list[int] = PrivateAttr(default_factory=list)
    _ref: Any = PrivateAttr(default=None)

    def model_post_init(self, context: Any, /) -> None:
        """Compute the values derived from the fields, once."""
        self._quote_id = self._find_token('"')
        self._comma_id = self._find_token(",")
        self._brace_id = self._find_token("}")
        self._closers = closing_string_tokens(self.vocab)
        self._string_ids = static_string_token_ids(self.vocab)
        self._ref = self.llm.encode("x")

    def _find_token(self, text: str) -> int:
        """Return the id of the token whose text is exactly `text`."""
        for token_id, token_text in self.vocab.items():
            if token_text == text:
                return token_id
        raise ValueError(f"No token {text!r} in the vocabulary")

    @property
    def names(self) -> list[str]:
        """Names of all available functions."""
        return [function.name for function in self.functions]

    def _logits(self, ids: list[int]) -> list[float]:
        """Ask the model for the next-token logits."""
        logits: list[float] = self.llm.get_logits_from_input_ids(
            self._ref.new_tensor([ids])
        )
        return logits

    def _append_text(self, ids: list[int], text: str) -> None:
        """Encode fixed text written by us and append its ids in place."""
        if not text:
            return
        ids.extend(self.llm.encode(text)[0].tolist())

    def _build_prompt(self, request: str) -> str:
        """Build the text given to the model before the function name."""
        lines = ["Available functions:"]
        for function in self.functions:
            lines.append(f"- {function.name}: {function.description}")
        lines.append("")
        lines.append("Examples:")
        lines.append('Request: Remove every digit from "abc123"')
        lines.append(
            r'Function call: {"name": "fn_example_replace", '
            r'"parameters": {"text": "abc123", "pattern": "\\d+", '
            r'"replacement": ""}}'
        )
        lines.append("Request: Swap vowels for dots in 'banana'")
        lines.append(
            r'Function call: {"name": "fn_example_replace", '
            r'"parameters": {"text": "banana", "pattern": "[aeiou]", '
            r'"replacement": "."}}'
        )
        lines.append("")
        lines.append(f"Request: {request}")
        lines.append('Function call: {"name": "')
        return "\n".join(lines)

    def generate_function_name(self, prompt_ids: list[int]) -> str:
        """Let the model choose a function name under constraint."""
        ids = list(prompt_ids)
        generated = ""
        while True:
            allowed = valid_name_token_ids(
                generated, self.names, self.vocab, self._quote_id
            )
            token_id = pick_allowed_token(self._logits(ids), allowed)
            if token_id == self._quote_id:
                return generated
            ids.append(token_id)
            generated += self.vocab[token_id]

    def generate_number(self, ids: list[int], end_id: int) -> float:
        """Let the model write a number; `ids` is extended in place."""
        generated = ""
        while True:
            allowed = valid_number_token_ids(generated, self.vocab, end_id)
            if (len(generated) >= MAX_NUMBER_LENGTH
                    and is_complete_number(generated)):
                allowed = [end_id]
            token_id = pick_allowed_token(self._logits(ids), allowed)
            ids.append(token_id)
            if token_id == end_id:
                return float(generated)
            generated += self.vocab[token_id]

    def generate_string(
        self, ids: list[int], follow: str
    ) -> tuple[str, str]:
        """Let the model write a JSON string; `ids` is extended in place.

        Args:
            ids: Token ids of the sequence; must end with the opening quote.
            follow: JSON text that must come right after the closing quote.

        Returns:
            (decoded value, part of `follow` already written by the model).
        """
        generated = ""
        while True:
            complete = scan_json_string(generated)[1]
            allowed = valid_string_token_ids(
                generated, self.vocab, self._string_ids,
                self._closers, follow,
            )
            if len(generated) >= MAX_STRING_LENGTH:
                if complete:
                    allowed = [i for i in allowed if i in self._closers]
                else:
                    allowed = [i for i in allowed if len(self.vocab[i]) == 1]
            token_id = pick_allowed_token(self._logits(ids), allowed)
            ids.append(token_id)
            if complete and token_id in self._closers:
                before, after = self._closers[token_id]
                value: str = json.loads('"' + generated + before + '"')
                return value, after
            generated += self.vocab[token_id]

    def generate_boolean(self, ids: list[int]) -> bool:
        """Let the model write a JSON boolean; `ids` is extended in place."""
        generated = ""
        while generated not in BOOLEANS:
            allowed = valid_boolean_token_ids(generated, self.vocab)
            token_id = pick_allowed_token(self._logits(ids), allowed)
            ids.append(token_id)
            generated += self.vocab[token_id]
        return generated == "true"

    def build_call(self, prompt: str) -> FunctionCallResult:
        """Generate the full function call for one prompt.

        Raises:
            ValueError: If a parameter has an unsupported type.
        """
        ids = self.llm.encode(self._build_prompt(prompt))[0].tolist()
        name = self.generate_function_name(ids)
        function = next(f for f in self.functions if f.name == name)
        self._append_text(ids, name + '", "parameters": {')

        values: dict[str, float | str | bool] = {}
        params = list(function.parameters.items())
        for index, (key, param) in enumerate(params):
            is_last = index == len(params) - 1
            closing = "}" if is_last else ","
            if index > 0:
                self._append_text(ids, " ")
            self._append_text(ids, f'"{key}": ')
            if param.type == "number":
                end_id = self._brace_id if is_last else self._comma_id
                values[key] = self.generate_number(ids, end_id)
            elif param.type == "string":
                self._append_text(ids, '"')
                value, written = self.generate_string(ids, closing)
                values[key] = value
                self._append_text(ids, closing[len(written):])
            elif param.type == "boolean":
                values[key] = self.generate_boolean(ids)
                self._append_text(ids, closing)
            else:
                raise ValueError(
                    f"Unsupported type {param.type!r} for parameter "
                    f"{key!r} of {name}"
                )

        return FunctionCallResult(prompt=prompt, name=name, parameters=values)
