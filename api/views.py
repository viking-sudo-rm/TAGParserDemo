# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function

import tensorflow as tf
from nltk.tokenize import sent_tokenize, word_tokenize
import json, os, sys, pickle

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

PATH_TO_PARSER = "graph_parser"
cwd = os.getcwd()
os.chdir(PATH_TO_PARSER)

sys.path.insert(0, os.path.abspath("."))
import utils

def read_sents(sents_file):
	with open(sents_file) as fhand:
		return [line.split() for line in fhand]

def get_conllu(test_opts):
	outputs = []
	sents = read_sents(test_opts.text_test)
	stags = read_sents(test_opts.predicted_stags_file)
	pos = read_sents(test_opts.predicted_pos_file)
	arcs = read_sents(test_opts.predicted_arcs_file)
	rels = read_sents(test_opts.predicted_rels_file)
	with open(os.path.join(test_opts.base_dir, 'predicted_conllu', 'test.conllu'), 'wt') as fout:
		for sent_idx in xrange(len(sents)):
			output = ""
			sent = sents[sent_idx]
			stags_sent = stags[sent_idx]
			pos_sent = pos[sent_idx]
			arcs_sent = arcs[sent_idx]
			rels_sent = rels[sent_idx]
			for word_idx in xrange(len(sent)):
				line = [str(word_idx+1)]
				line.append(sent[word_idx])
				line.append('_')
				line.append(stags_sent[word_idx])
				line.append(pos_sent[word_idx])
				line.append('_')
				line.append(arcs_sent[word_idx])
				line.append(rels_sent[word_idx])
				line.append('_')
				output += '\t'.join(line)
				output += '\n'
			outputs.append(output)
	return outputs

def output_sents(sents, test_opts):
	sents = sent_tokenize(sents)
	sents = map(word_tokenize, sents)
	path = os.path.join(test_opts.base_dir, 'sents', 'test.txt')
	with open(path, 'wt') as fout:
		for sent in sents:
			fout.write(' '.join(sent))
			fout.write('\n')

with open('demo/configs/config_demo.pkl') as fin:
	opts = pickle.load(fin)
with open('demo/configs/config_demo_test.pkl') as fin:
	test_opts = pickle.load(fin)

# Load a session into memory
print("Loading saved parser session..")
graph = tf.Graph()
with graph.as_default():
	Model = getattr(utils, opts.model)
	model = Model(opts, test_opts)
	session = tf.Session()
	with session.as_default():
		saver = tf.train.Saver()
		saver.restore(session, test_opts.modelname)

os.chdir(cwd)

@csrf_exempt
def parse(request):

	"""
	API request to parse a sentence.
	"""

	try:
		args = json.loads(request.body)
	except:
		return JsonResponse(None, safe=False)
	if type(args) != dict or "text" not in args:
		return JsonResponse(None, safe=False)

	os.chdir(PATH_TO_PARSER)
	text = args["text"]
	output_sents(text, test_opts)
	model.run_epoch(session, True)
	conllu = get_conllu(test_opts)
	os.chdir(cwd)
	return JsonResponse({"conllu": conllu}, safe=False)