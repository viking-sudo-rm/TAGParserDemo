"""TAGParserDemo URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^api/', include("api.urls")),
    url(r'^', include("demo.urls")),
]

# FIXME not ideal
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
urlpatterns += staticfiles_urlpatterns()

#############################
# Download all the files ONCE
#############################

print("Starting stuff in urls.py")

import os
from boto.s3.connection import S3Connection
from boto.exception import S3ResponseError

DATA_DIR = "s3"
PARSER_DIR = "graph_parser"
DEMO_DIR = os.path.join(DATA_DIR, "demo")
GLOVE_DIR = os.path.join(DATA_DIR, "glovevector")
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
            l.get_contents_to_filename(s3_path)
        except (OSError, S3ResponseError) as e:
            pass
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