# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os

from django.apps import AppConfig
from boto.s3.connection import S3Connection
from boto.exception import S3ResponseError

DATA_DIR = "s3"
DEMO_DIR = os.path.join(DATA_DIR, "demo")
MODEL_DIR = os.path.join(DEMO_DIR, "Pretrained_Parser/best_model")

BUCKET_NAME = os.environ["S3_BUCKET"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_ACCESS_SECRET_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
REGION = "s3.us-east-2.amazonaws.com"

class ApiConfig(AppConfig):

	name = 'api'

	def ready(self):

		print("Bucket name", BUCKET_NAME)
		print("Access key", AWS_ACCESS_KEY_ID)
		print("Secret key", AWS_ACCESS_SECRET_KEY)
		print("Region", REGION)

		print("Connecting to AWS..")
		conn = S3Connection(AWS_ACCESS_KEY_ID, AWS_ACCESS_SECRET_KEY, host=REGION)
		print("Getting bucket..")
		bucket = conn.get_bucket(BUCKET_NAME)
		print("Accessing bucket..")
		self.download_bucket(bucket, DATA_DIR)
		print("Bucket downloaded successfully!")


	def download_bucket(self, bucket, path):

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
