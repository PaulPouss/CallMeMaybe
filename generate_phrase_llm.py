import json
import torch
from llm_sdk.llm_sdk import Small_LLM_Model


# ---------------------------------------------------------
# 1. Choix du prochain token
# ---------------------------------------------------------

def choose_next_token(logits) -> int:
    """
    Choisit le token ayant le score le plus élevé.
    C'est du greedy decoding.
    """
    return max(
        range(len(logits)),
        key=lambda token_id: logits[token_id]
    )


# ---------------------------------------------------------
# 2. Génération token par token
# ---------------------------------------------------------

def generate(model, prompt: str, max_tokens: int = 200) -> str:
    """
    Génère du texte à partir d'un prompt.
    """

    input_ids = model.encode(prompt)

    generated_text = []

    for _ in range(max_tokens):

        logits = model.get_logits_from_input_ids(input_ids)

        next_token_id = choose_next_token(logits)

        next_token_text = model.decode([next_token_id])

        generated_text.append(next_token_text)

        next_token_tensor = torch.tensor(
            [[next_token_id]],
            dtype=input_ids.dtype,
            device=input_ids.device
        )

        input_ids = torch.cat(
            [input_ids, next_token_tensor],
            dim=1
        )

        # Vérifie si on a obtenu un JSON complet
        current_text = "".join(generated_text).strip()

        if current_text.startswith("{"):
            try:
                json.loads(current_text)
                return current_text
            except json.JSONDecodeError:
                pass

    return "".join(generated_text)


# ---------------------------------------------------------
# 3. Fonctions disponibles
# ---------------------------------------------------------

FUNCTIONS = [
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
            "a": {
                "type": "number"
            },
            "b": {
                "type": "number"
            }
        },
        "returns": {
            "type": "number"
        }
    },
    {
        "name": "fn_greet",
        "description": "Generate a greeting message for a person by name.",
        "parameters": {
            "name": {
                "type": "string"
            }
        },
        "returns": {
            "type": "string"
        }
    },
    {
        "name": "fn_reverse_string",
        "description": "Reverse a string and return the reversed result.",
        "parameters": {
            "s": {
                "type": "string"
            }
        },
        "returns": {
            "type": "string"
        }
    },
    {
        "name": "fn_get_square_root",
        "description": "Calculate the square root of a number.",
        "parameters": {
            "a": {
                "type": "number"
            }
        },
        "returns": {
            "type": "number"
        }
    },
    {
        "name": "fn_substitute_string_with_regex",
        "description": "Replace all occurrences matching a regex pattern in a string.",
        "parameters": {
            "source_string": {
                "type": "string"
            },
            "regex": {
                "type": "string"
            },
            "replacement": {
                "type": "string"
            }
        },
        "returns": {
            "type": "string"
        }
    }
]


# ---------------------------------------------------------
# 4. Construction du prompt
# ---------------------------------------------------------

def build_function_selection_prompt(user_prompt: str) -> str:
    """
    Construit le prompt envoyé au LLM pour choisir
    la fonction correspondant à la demande.
    """

    functions_json = json.dumps(
        FUNCTIONS,
        indent=2
    )

    return f"""
You are a function selector.

You have access to the following functions:

{functions_json}

Your task is to choose the function that can solve the user's request.

You must extract the values needed by the function from the user's request.

Return ONLY valid JSON.

The JSON must have exactly this format:

{{
  "name": "function_name",
  "arguments": {{}}
}}

Do not explain your answer.
Do not use markdown.
Do not add any text before or after the JSON.

User request:
{user_prompt}

Answer:
""".strip()


# ---------------------------------------------------------
# 5. Sélection d'une fonction
# ---------------------------------------------------------

def choose_function(model, user_prompt: str) -> str:
    """
    Demande au LLM quelle fonction utiliser
    et retourne sa réponse JSON.
    """

    prompt = build_function_selection_prompt(user_prompt)

    return generate(
        model,
        prompt,
        max_tokens=150
    )


# ---------------------------------------------------------
# 6. Programme principal
# ---------------------------------------------------------

def main():
    model = Small_LLM_Model()

    test_prompts = [
        "What is the sum of 2 and 3?",
        "What is the sum of 265 and 345?",
        "Greet shrek",
        "Greet john",
        "Reverse the string 'hello'",
        "Reverse the string 'world'",
        "What is the square root of 16?",
        "Calculate the square root of 144",
        'Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS',
        "Replace all vowels in 'Programming is fun' with asterisks",
        "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'",
    ]

    for prompt in test_prompts:
        print("\n" + "=" * 60)
        print(f"REQUEST: {prompt}")

        result = choose_function(model, prompt)

        print("MODEL:", result)

        try:
            call = json.loads(result)

            print("FUNCTION:", call["name"])
            print("ARGUMENTS:", call["arguments"])

        except (json.JSONDecodeError, KeyError) as e:
            print("ERROR:", e)
            print("RAW OUTPUT:", result)


if __name__ == "__main__":
    main()