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

PARSER_DIR = "graph_parser"
DATA_DIR = "s3"
DEMO_DIR = os.path.join(DATA_DIR, "demo")
MODEL_DIR = os.path.join(DEMO_DIR, "Pretrained_Parser/best_model")

def download_bucket(bucket, path):

	if not os.path.exists(path):
		os.mkdir(path)
	bucket_list = bucket.list()

	for l in bucket_list:
		key_string = str(l.key)
		s3_path = os.path.join(path, key_string)
		try:
			print ("Current File is ", s3_path)
			os.makedirs(os.path.dirname(s3_path))
			l.get_contents_to_filename(s3_path)
			print ("Is file", os.path.isfile(s3_path))
		except (OSError, S3ResponseError) as e:
			pass
			print("Got error", e)
			# check if the file has been downloaded locally  
			if not os.path.exists(s3_path):
				try:
					os.makedirs(s3_path)
				except OSError as exc:
					# let guard againts race conditions
					import errno
					if exc.errno != errno.EEXIST:
						raise

BUCKET_NAME = os.environ["S3_BUCKET"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_ACCESS_SECRET_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
REGION = "s3.us-east-2.amazonaws.com"

print("Bucket name", BUCKET_NAME)
print("Access key", AWS_ACCESS_KEY_ID)
print("Secret key", AWS_ACCESS_SECRET_KEY)
print("Region", REGION)

print("Connecting to AWS..")
conn = S3Connection(AWS_ACCESS_KEY_ID, AWS_ACCESS_SECRET_KEY, host=REGION)
print("Getting bucket..")
bucket = conn.get_bucket(BUCKET_NAME)
print("Accessing bucket..")
download_bucket(bucket, DATA_DIR)
print("Bucket downloaded successfully!")

print("Updating path..")
sys.path.insert(0, os.path.abspath(PARSER_DIR))
import utils
from utils.models.demo import Demo_Parser

# import graph_parser.utils
# from graph_parser.utils.models.demo import Demo_Parser

print("Loading saved parser session..")
print("Saved model in {}.".format(MODEL_DIR))
graph = tf.Graph()
with graph.as_default():
	model = Demo_Parser(DEMO_DIR)
	session = tf.Session()
	with session.as_default():
		session.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		saver.restore(session, MODEL_DIR)

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
