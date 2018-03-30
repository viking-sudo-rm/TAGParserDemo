# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function

import tensorflow as tf
from nltk.tokenize import sent_tokenize, word_tokenize
import json, os, sys, pickle, traceback

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.contrib.staticfiles.management.commands.runserver import Command as RunserverCommand

PATH_TO_PARSER = "graph_parser"

def read_sents(sents_file):
	with open(sents_file) as fhand:
		return [line.split() for line in fhand]

def get_parse(test_opts):
	outputs = []
	sent = read_sents(test_opts.text_test)[0]
	# FIXME: after the first sentence, parse output doesn't update!
	stags_sent, arcs_sent, rels_sent = model.predict(session)
	# arcs_sent = arcs[sent_idx]
	# rels_sent = rels[sent_idx]
	print(len(sent), sent)
	print(len(stags_sent), stags_sent)
	print(len(arcs_sent), arcs_sent)
	print(len(rels_sent), rels_sent)
	arcs = [{
		"start": min(i, int(arcs_sent[i]) - 1), # int(arcs_sent[i]) - 1
		"end": max(i, int(arcs_sent[i]) - 1), # i
		"dir": "right" if i < int(arcs_sent[i]) - 1 else "left", # right
		"label": rels_sent[i],
	} for i in xrange(len(sent)) if rels_sent[i] != "ROOT"]
	words = [{
		"tag": stags_sent[i],
		"text": sent[i],
	} for i in xrange(len(sent))]
	return {"arcs": arcs, "words": words}

def output_sents(text, test_opts):
	sent = word_tokenize(sent_tokenize(text)[0])
	# sents = map(word_tokenize, sents)
	path = os.path.join(test_opts.base_dir, 'sents', 'test.txt')
	with open(path, 'w') as fout:
		# for sent in sents:
		fout.write(' '.join(sent))
		fout.write('\n')

cwd = os.getcwd()
os.chdir(PATH_TO_PARSER)
sys.path.insert(0, os.path.abspath("."))
import utils

with open('demo/configs/config_demo.pkl') as fin:
	opts = pickle.load(fin)
with open('demo/configs/config_demo_test.pkl') as fin:
	test_opts = pickle.load(fin)

# Load a session into memory
print("Loading saved parser session..")
graph = tf.Graph()
with graph.as_default():
	print("Using {}.".format(opts.model))
	Model = getattr(utils, opts.model)
	model = Model(opts, test_opts)
	session = tf.Session()
	with session.as_default():
		session.run(tf.global_variables_initializer())
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
		return JsonResponse({"error": "no POST data"}, safe=False)
	if type(args) != dict or "text" not in args:
		return JsonResponse({"error": "no text"}, safe=False)

	os.chdir(PATH_TO_PARSER)
	text = args["text"]

	try:
		output_sents(text, test_opts)
		# model.run_epoch(session, testmode=True)
		# print(model.predict(session))
		parse = get_parse(test_opts)
		os.chdir(cwd)
		return JsonResponse({"parse": parse}, safe=False)
	except:
		os.chdir(cwd)
		print('-' * 60)
		traceback.print_exc(file=sys.stdout)
		print('-' * 60)
		return JsonResponse({"error": "internal error"}, safe=False)