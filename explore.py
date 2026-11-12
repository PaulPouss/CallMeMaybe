from llm_sdk.llm_sdk import Small_LLM_Model
import json
from src.vocabulary import load_vocab


model = Small_LLM_Model()
ref = model.encode("x")
ids = model.encode("What is the sum of 2 and 3?")[0].tolist()
print("ids ->", type(ids), ids)

ids_tensor = ref.new_tensor([ids])
logits = model.get_logits_from_input_ids(ids_tensor)
print("logits ->", type(logits), len(logits))

path = model.get_path_to_vocab_file()
print("vocab path ->", path)
with open(path, encoding="utf-8") as f:
    raw = f.read()
print(raw[:300])

vocab = json.loads(raw)
print("vocab size ->", len(vocab))
inverse = {i: t for t, i in vocab.items()}
print("220 ->", repr(inverse.get(220)), "| 3838 ->", repr(inverse.get(3838)))

v = load_vocab(model.get_path_to_vocab_file())
print(len(v), repr(v[220]), repr(v[3838]), repr(v[17]))
