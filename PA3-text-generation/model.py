from transformers import GPT2Tokenizer, GPT2TokenizerFast, GPT2LMHeadModel, pipeline, set_seed
from datasets import Dataset, load_dataset
import numpy as np


import torch
from tqdm import tqdm

def get_transformer_model():

	# Feel free to change models if having memory issue
	tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
	tokenizer.pad_token = tokenizer.eos_token
	model = GPT2LMHeadModel.from_pretrained("gpt2",
										 	output_attentions=True,
											attn_implementation="eager")

	# 'pt' for PyTorch, 'tf' for TensorFlow
	framework = 'pt'

	return TransformerModel(model, tokenizer, framework)


class TransformerModel(object):

	def __init__(self, model, tokenizer, framework='pt'):

		self.model = model
		self.tokenizer = tokenizer
		self.framework = framework

		##### Feel free to add more attributes here if needed #####
		self.generator = pipeline(
			'text-generation',
			model = model,
			tokenizer = tokenizer
		)
		set_seed(42)


	def generate_text(self, prompt, max_new_tokens=10, num_return_sequences=1):
		"""
		The method generates the complementary text for a given starting
		text, i.e., the prompt.

		Args:
			prompt: the starting text as a string
			max_length [optional]: the max length of the generated text

		Return:
			results: the generated text as a string.
		"""

		##### Your code here #####
	
		output = self.generator(
			prompt,
			max_new_tokens = max_new_tokens,
			num_return_sequences = num_return_sequences,
			temperature=0.7,
			top_k=5,
			top_p=0.9,
		)

		results = []
		for result in output:
			results.append(result['generated_text'])

		##### Code done #####c
		results = "\n".join(results)

		return results


	def evaluate_ppl(self, dataset):
		"""
		The method for evaluating the perplexity score on given datasets,
		e.g., WikiText-2.

		Args:
			dataset: a `datasets.Dataset' instance from Huggingface

		Return:
			score: A float number. The perplexity score.
		"""

		##### Your code here #####

		encodings = self.tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
		
		max_length = self.model.config.n_positions
		stride = 1024
		seq_len = encodings.input_ids.size(1)

		nll_sum = 0.0
		n_tokens = 0
		prev_end_loc = 0
		for begin_loc in tqdm(range(0, seq_len, stride)):
			end_loc = min(begin_loc + max_length, seq_len)
			trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
			input_ids = encodings.input_ids[:, begin_loc:end_loc]
			target_ids = input_ids.clone()
			target_ids[:, :-trg_len] = -100

			with torch.no_grad():
				outputs = self.model(input_ids, labels=target_ids)

				# loss is calculated using CrossEntropyLoss which averages over valid labels
				# N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
				# to the left by 1.
				neg_log_likelihood = outputs.loss

			# Accumulate the total negative log-likelihood and the total number of tokens
			num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
			batch_size = target_ids.size(0)
			num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
			nll_sum += neg_log_likelihood * num_loss_tokens
			n_tokens += num_loss_tokens

			prev_end_loc = end_loc
			if end_loc == seq_len:
				break

		avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
		ppl = torch.exp(avg_nll)

		##### Code done #####

		return ppl


	def get_template(self, doc, lbl):
		##### Write your own template below #####
		# template = 'Review: \"%s\"\nSentiment: %s' %(doc, lbl)
		##### Template done #####
		
		template = '\"%s\"\n(positive or negative): %s' %(doc, lbl)

		return template


	def fewshot_sentiment(self, trainSet, test_doc):
		"""
		Taking advantage of the language model to perform sentiment analysis.

		Args:
			trainSet: List of tuples. Each tuple is a pair of (document, label),
					  where `document` is a string of the entire document and 
					  label is either 'positive' or 'negative'
			test_doc: String. The test document.
		Return:
			prediction: String. The predicted sentiment, 'positive' or 
						'negative'.
		"""

		prompt = ''
		for (doc, lbl) in trainSet:
			prompt += self.get_template(doc, lbl)
			prompt += '\n###\n'

		prompt += self.get_template(test_doc, "")

		# 'positive'/'negative' plus an EoS token

		# print(prompt)

		prediction = self.generate_text(prompt, max_new_tokens=2)

		return prediction.split('\n###\n')[-1]


	def visualize_attention(self, trainSet, test_doc, layer=-1):
		"""
		(Bonus) Visualize how attention works in the fewshot sentiment analysis.

		Args:
			trainSet: List of tuples. Each tuple is a pair of (document, label),
					  where `document` is a string of the entire document and 
					  label is either 'positive' or 'negative'
			test_doc: String. The test document.
			layer: Integer. To speficify which attention layer to be visualized.
		Return:
			template: The template input to the language model.
			weights: 1D-Array. The attention score of each token in the template.
					 Values should be in [0,1], normalize if needed.
		"""

		prompt = ''
		for (doc, lbl) in trainSet:
			prompt += self.get_template(doc, lbl)
			prompt += '\n###\n'

		prompt += self.get_template(test_doc, "")

		##### Your code here #####

		inputs = self.tokenizer(prompt,
						  		return_offsets_mapping=True,
								return_special_tokens_mask=True,
								return_attention_mask=True,
						  		return_tensors="pt")

		outputs = self.model(**inputs)

		enc = inputs.encodings[0]
		
		attentions = outputs.attentions

		attn_layer = attentions[layer]

		# Option 1: Use attention from the last head, last token (standard trick)
		attn_weights = attn_layer[0, -1, -1]  # shape: [seq_len]
		print(np.shape(attn_weights))
		attn_weights = attn_weights.detach().numpy()

		# Map tokens to words using .words
		word_map = enc.words  # e.g., [None, 0, 1, 2, 3, 3, 4, None]

		# Group token attentions into word attentions
		word_attention = {}
		for i, word_idx in enumerate(word_map):
			if word_idx is not None:
				word_attention.setdefault(word_idx, []).append(attn_weights[i])

		# Average attentions across tokens for each word
		avg_word_attn = [np.mean(word_attention[i]) for i in sorted(word_attention)]

		##### Code done #####

		print(len(prompt.split()))
		print(len(avg_word_attn))

		assert len(prompt.split())==len(avg_word_attn)

		return prompt, avg_word_attn


	def finetune(self, trainSet):
		"""
		Taking advantage of the language model to perform sentiment analysis.

		Args:
			trainSet: List of tuples. Each tuple is a pair of (document, label),
					  where `document` is a string of the entire document and 
					  label is either 'positive' or 'negative'
		"""
		templates = [{"text": self.get_template(doc, lbl)} for doc, lbl in trainSet]
		dataset = Dataset.from_list(templates)
		# Use "left" truncation so that the sentiment is not truncated.
		map_tokenize = lambda x: self.tokenizer(x['text'], truncation_side='left')
		dataset = dataset.map(map_tokenize, batched=True)
		dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.1)

		##### Your code here #####



		##### Code done #####
