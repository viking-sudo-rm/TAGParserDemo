# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os

from django.apps import AppConfig
from django.conf import settings
from boto.s3.connection import S3Connection
from boto.exception import S3ResponseError

class ApiConfig(AppConfig):

	name = 'api'

	def ready(self):

		print("Bucket name", settings.BUCKET_NAME)
		print("Access key", settings.AWS_ACCESS_KEY_ID)
		print("Secret key", settings.AWS_ACCESS_SECRET_KEY)
		print("Region", settings.REGION)

		print("Connecting to AWS..")
		conn = S3Connection(settings.AWS_ACCESS_KEY_ID, settings.AWS_ACCESS_SECRET_KEY, host=settings.REGION)
		print("Getting bucket..")
		bucket = conn.get_bucket(settings.BUCKET_NAME)
		print("Accessing bucket..")
		self.download_bucket(bucket, settings.DATA_DIR)
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
