from model import get_transformer_model
from datasets import load_dataset

gpt2_model = get_transformer_model()
"""
prompt = "Text generation is the task of generating text with the goal of"
print(gpt2_model.generate_text(prompt))

prompt = "Hello, I'm a language model,"
print(gpt2_model.generate_text(prompt))

prompt = "The man worked as a"
print(gpt2_model.generate_text(prompt))
"""

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

print(gpt2_model.evaluate_ppl(test))