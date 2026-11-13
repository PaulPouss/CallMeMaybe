import numpy as np
from typing import Callable
from pathlib import Path

from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import BaseModel


class FunctionCallResult(BaseModel):
    name: str
    prompt: str
    parameters: dict[str, str | float | bool]
    pass


class PromptInput(BaseModel):
    prompt: str

    @classmethod
    def load_all(cls, path: Path) -> list["PromptInput"]:
        """Read a JSON file and return one PromptInput per element.

        Args:
            path: Path to the tests JSON file.

        Returns:
            The validated prompts.

        Raises:
            ValueError: If the file is not valid JSON, is not a list,
                or contains an invalid element.
        """


class PromptGenerator:
    """Builds JSON output for a function call, token by token,
    under constrained decoding.
    """

    def __init__(self, model: Small_LLM_Model, functions: list) -> None:
        self.model = model
        self.functions = functions
        self.vocab: dict[int, str] = self._load_vocab()
        self.input_ids: list[int] = []

    def _load_vocab(self) -> dict[int, str]:
        """Read the vocabulary and build token_id -> token_string."""
        raise NotImplementedError

    def _append_text(self, text: str) -> None:
        """Encode fixed text and append its token ids."""
        raise NotImplementedError

    def _append_token(self, token_id: int) -> None:
        """Append one token selected by the model."""
        self.input_ids.append(token_id)

    def _masked_logits(self, allowed_ids: set[int]) -> np.ndarray:
        """Get logits and mask all forbidden tokens."""
        raise NotImplementedError

    def _generate_constrained(
        self,
        allowed_ids_fn: Callable[[str], set[int]],
        stop_fn: Callable[[str], bool],
    ) -> str:
        raise NotImplementedError

    def generate_function_name(self) -> str:
        raise NotImplementedError

    def generate_number(self) -> float:
        raise NotImplementedError

    def generate_string(self) -> str:
        raise NotImplementedError

    def generate_boolean(self) -> bool:
        raise NotImplementedError

    def build_call(self, prompt: str) -> dict:
        raise NotImplementedError
