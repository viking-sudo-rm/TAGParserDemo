# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function

import json, os, sys, pickle, traceback
import tensorflow as tf
from nltk.tokenize import sent_tokenize, word_tokenize
from boto.s3.connection import S3Connection
from boto.exception import S3ResponseError

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.contrib.staticfiles.management.commands.runserver import Command as RunserverCommand
from django.conf import settings

print("Updating path..")
sys.path.insert(0, os.path.abspath(settings.PARSER_DIR))
import utils
from utils.models.demo import Demo_Parser

# import graph_parser.utils
# from graph_parser.utils.models.demo import Demo_Parser

print("Loading saved parser session..")
print("Saved model in {}.".format(settings.MODEL_DIR))
graph = tf.Graph()
with graph.as_default():
	model = Demo_Parser(settings.DEMO_DIR)
	session = tf.Session()
	with session.as_default():
		session.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		saver.restore(session, settings.MODEL_DIR)

def word_tokenize_period(sent):
	words = word_tokenize(sent)
	if words[-1] not in ['.', '?']:
		words.append('.')
	return words

def get_parse(sents, session=session):
	# FIXME -- rn this assumes len(sents) == 1
	sent = sents[0]
	parses = model.run_on_sents(session, sents)
	stags = parses["stags"]
	arcs = map(int, parses["arcs"])
	rels = parses["rels"]
	arcs = [{
		"start": min(i, arcs[i] - 1), # int(arcs[i]) - 1
		"end": max(i, arcs[i] - 1), # i
		# "dir": "right" if i < arcs[i] - 1 else "left", # right
		"dir": "left" if i < arcs[i] - 1 else "right", # right
		"label": rels[i],
	} for i in xrange(len(sent)) if rels[i] != "ROOT"]
	words = [{
		"tag": stags[i],
		"text": sent[i],
	} for i in xrange(len(sent))]
	return {"arcs": arcs, "words": words}

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

	try:
		text = args["text"]
		sents = [word_tokenize_period(sent) for sent in sent_tokenize(text)]
		sents = sents[:1] # Restricted to first sentence
		parse = get_parse(sents)
		return JsonResponse({"parse": parse}, safe=False)
	except:
		print('-' * 60)
		traceback.print_exc(file=sys.stdout)
		print('-' * 60)
		return JsonResponse({"error": "internal error"}, safe=False)
