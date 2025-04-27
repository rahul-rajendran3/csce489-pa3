import argparse
from model import get_transformer_model
from datasets import load_dataset

if __name__ == '__main__':

	p = argparse.ArgumentParser()
	p.add_argument('-p', '--ppl', dest='ppl', action='store_true', default=False)
	p.add_argument('-n', '--num', dest='num', type=int, default=1)
	p.add_argument('-m', '--max', dest='max_token', type=int, default=10)
	args = p.parse_args()

	transformer_model = get_transformer_model()

	print(args.num)
	with open('prompt.txt', 'r+') as f:
		prompt = f.readlines()[0]

	if args.ppl:
		print('Task 2: Perplexity score')
		print('==============')
		wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
		ppl = transformer_model.evaluate_ppl(wikitext)
		print('Perplexity = %.4f.' %ppl)
		
	else:
		print('Task 1: Sentence generation')
		print('==============')
		generated = transformer_model.generate_text(
			prompt, max_new_tokens=args.max_token, num_return_sequences=args.num)
		print(generated)