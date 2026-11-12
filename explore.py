from llm_sdk.llm_sdk import Small_LLM_Model

from src.config import DEFAULT_FUNCTIONS
from src.loader import load_functions
from src.masks import valid_name_token_ids
from src.vocabulary import load_vocab

from src.generator import Generator


model = Small_LLM_Model()

# Vocabulaire : id -> texte
v = load_vocab(model.get_path_to_vocab_file())
print("vocab ->", len(v), repr(v[220]), repr(v[3838]), repr(v[17]))

# Noms de fonctions lus depuis le vrai fichier
functions = load_functions(DEFAULT_FUNCTIONS)
names = [function.name for function in functions]
print("names ->", names)

# Id du guillemet, retrouvé dans le vocabulaire
quote_id = next(i for i, t in v.items() if t == '"')
print("quote_id ->", quote_id)

# Masque du nom de fonction pour plusieurs débuts
for start in ["", "fn_", "fn_g", "fn_greet", "fn_get_square_root"]:
    ids_ok = valid_name_token_ids(start, names, v, quote_id)
    print(repr(start), "->", sorted(v[i] for i in ids_ok))


generator = Generator(llm=model, functions=functions, vocab=v)


def build_prompt(request: str) -> str:
    lines = ["Available functions:"]
    for function in functions:
        lines.append(f"- {function.name}: {function.description}")
    lines.append("")
    lines.append(f"Request: {request}")
    lines.append('Function call: {"name": "')
    return "\n".join(lines)


for request in [
    "What is the sum of 2 and 3?",
    "Greet shrek",
    "Reverse the string 'hello'",
    "What is the square root of 16?",
    "Replace all vowels in 'Programming is fun' with asterisks",
    "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat'",
]:
    prompt_ids = model.encode(build_prompt(request))[0].tolist()
    print(request, "->", generator.generate_function_name(prompt_ids))

from src.masks import valid_number_token_ids

comma_id = next(i for i, t in v.items() if t == ",")
for start in ["", "-", "3", "3.", "3.5", "-0"]:
    ids_ok = valid_number_token_ids(start, v, comma_id)
    texts = sorted(v[i] for i in ids_ok)
    print(repr(start), "->", len(texts), "tokens :", texts[:15])
