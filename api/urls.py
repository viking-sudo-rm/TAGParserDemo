from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^parse', views.parse, name='parse'),
]