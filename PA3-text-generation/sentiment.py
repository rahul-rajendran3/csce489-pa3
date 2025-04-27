import os, argparse
import random
from model import get_transformer_model
from datasets import load_dataset

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def loadSets(dataDir, max_length, shuffle=False):
	"""
	Load dataset from given directory. Return List of tuples. 
	Each tuple is a pair of (document, label), where `document` is a string of the 
	entire document and label is either 'positive' or 'negative'
	"""
	posTrainFileNames = os.listdir('%s/pos/' % dataDir)
	negTrainFileNames = os.listdir('%s/neg/' % dataDir)
	dataset = []
	for fileName in posTrainFileNames:
		dataset.append(
			(read_file('%s/pos/%s' % (dataDir, fileName), max_length), 'positive', fileName[2:5]))
	for fileName in negTrainFileNames:
		dataset.append(
			(read_file('%s/neg/%s' % (dataDir, fileName), max_length), 'negative', fileName[2:5]))
	if shuffle:
		random.shuffle(dataset)

	fname = [d[2] for d in dataset]
	dataset = [(d[0], d[1]) for d in dataset]
	return dataset, fname

def read_file(fileName, max_length):
	"""
	Read text in the given file. Return a single string for the document.
	"""
	contents = []
	with open(fileName) as f:
		for line in f:
			contents.append(line)

	# Limit max sentence length
	result = ('\n'.join(contents)).split(' ')[:max_length]
	return ' '.join(result)

def colorize(words, color_array):
	"""
	Visualize weight of each word
	Based on code at: https://stackoverflow.com/a/59249273.
	"""
	cmap = truncate_colormap(matplotlib.cm.YlOrRd, maxval=0.9)
	template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
	colored_string = ''
	for i, (word, color) in enumerate(zip(words, color_array)):
		color = matplotlib.colors.rgb2hex(cmap(color)[:3])
		colored_string += template.format(color, '&nbsp' + word + '&nbsp')
		if not (i+1)%25:
			colored_string += '<br>'

	with open('explained.html', 'w') as f:
		f.write(colored_string)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

if __name__ == '__main__':
	p = argparse.ArgumentParser()
	p.add_argument('-t', '--tune', dest='finetune', action='store_true', default=False)
	p.add_argument('-d', '--debug', dest='debug', action='store_true', default=False)
	p.add_argument('-L', '--length', dest='max_length', type=int, default=100)
	p.add_argument('-x', '--exp', dest='explain', action='store_true', default=False)
	args = p.parse_args()

	transformer_model = get_transformer_model()

	random.seed(42) # For reproductibility

	test_dir = 'test_examples_debug' if args.debug else 'test_examples'
	if args.finetune:
		trainSet, _ = loadSets('fewshot_examples', max_length=args.max_length, shuffle=True)
		testSet, fname = loadSets(test_dir, max_length=args.max_length, shuffle=False)
		transformer_model.finetune(trainSet)
		acc = 0
		for (doc, lbl), name in zip(testSet, fname):
			prompt = transformer_model.get_template(doc, "")
			pred = transformer_model.generate_text(prompt, max_new_tokens=1).split()
			if pred[-1] != lbl:
				if args.debug:
					print('%s file no. %s got: %s ... %s' %(
						lbl, name, ' '.join(pred[:10]), ' '.join(pred[-2:])))
				else:
					print('%s file no. %s incorrect' %(lbl, name))
			else:
				acc += 1.0
		print('Accuracy after finetuning: %.2f' %(acc/len(testSet)))
	else:
		trainSet, _ = loadSets('fewshot_examples', max_length=args.max_length, shuffle=True)
		testSet, fname = loadSets(test_dir, max_length=args.max_length, shuffle=False)
		acc = 0
		for (doc, lbl), name in zip(testSet, fname):
			pred = transformer_model.fewshot_sentiment(trainSet, doc).split()
			if pred[-1] != lbl:
				if args.debug:
					print('%s file no. %s got: %s ... %s' %(
						lbl, name, ' '.join(pred[:10]), ' '.join(pred[-2:])))
				else:
					print('%s file no. %s incorrect' %(lbl, name))
			else:
				acc += 1
		print('Fewshot accuracy: %.2f' %(acc/len(testSet)))

		if args.explain:
			doc, lbl = testSet[0]
			template, weights = transformer_model.visualize_attention(trainSet, doc)
			colorize(template.split(), weights)
