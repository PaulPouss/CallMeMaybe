"""Constrained generation of function calls."""

from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from llm_sdk.llm_sdk import Small_LLM_Model

from .masks import pick_allowed_token, valid_name_token_ids
from .models import FunctionDef


class Generator(BaseModel):
    """Generate function calls token by token under constraint."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Small_LLM_Model
    functions: list[FunctionDef]
    vocab: dict[int, str]

    _quote_id: int = PrivateAttr(default=-1)
    _ref: Any = PrivateAttr(default=None)

    def model_post_init(self, context: Any, /) -> None:
        """Compute the values derived from the fields, once."""
        quote_ids = [i for i, t in self.vocab.items() if t == '"']
        if not quote_ids:
            raise ValueError("No double-quote token in the vocabulary")
        self._quote_id = quote_ids[0]
        self._ref = self.llm.encode("x")

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

    def generate_function_name(self, prompt_ids: list[int]) -> str:
        """Let the model choose a function name under constraint.

        Args:
            prompt_ids: Token ids of the prompt, ending where the name starts.

        Returns:
            The chosen function name.
        """
        ids = list(prompt_ids)
        generated = ""
        while True:
            allowed = valid_name_token_ids(
                generated, self.names, self.vocab, self._quote_id
            )
            logits = self._logits(ids)
            token_id = pick_allowed_token(logits, allowed)
            if token_id == self._quote_id:
                return generated
            ids.append(token_id)
            generated += self.vocab[token_id]
