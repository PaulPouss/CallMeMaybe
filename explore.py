import time

from llm_sdk.llm_sdk import Small_LLM_Model

from src.config import DEFAULT_FUNCTIONS, DEFAULT_INPUT
from src.generator import Generator
from src.loader import load_functions, load_prompts
from src.vocabulary import load_vocab

model = Small_LLM_Model()
vocab = load_vocab(model.get_path_to_vocab_file())
functions = load_functions(DEFAULT_FUNCTIONS)
generator = Generator(llm=model, functions=functions, vocab=vocab)

start = time.time()
for item in load_prompts(DEFAULT_INPUT):
    result = generator.build_call(item.prompt)
    print(result.model_dump_json())
print(f"Total : {time.time() - start:.1f} s")

sample = r'"pattern": "\\d+", "replacement": ""}}'
print([vocab[i] for i in model.encode(sample)[0].tolist()])