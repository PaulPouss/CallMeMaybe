"""Pydantic models describing the data handled by the program."""

from pydantic import BaseModel


class ParameterDef(BaseModel):
    """Type description of one parameter or of a return value."""

    type: str


class FunctionDef(BaseModel):
    """A function available to the LLM (one entry of the definitions)."""

    name: str
    description: str
    parameters: dict[str, ParameterDef]
    returns: ParameterDef


class PromptInput(BaseModel):
    """One natural-language prompt to translate into a function call."""

    prompt: str


class FunctionCallResult(BaseModel):
    """One function call produced for a prompt (one entry of the output)."""

    prompt: str
    name: str
    parameters: dict[str, float | str | bool]
