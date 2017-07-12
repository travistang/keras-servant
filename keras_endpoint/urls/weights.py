from django.conf.urls import url
from ..api import keras_weights

urlpatterns = [
    url(r'^get/(?P<model_name>\w{1,256})/(?P<file_name>\w{1,256})/$',keras_weights.weights_get),
    url(r'^get/(?P<model_name>\w{1,256})/$',keras_weights.weights_get),
    url(r'^create/(?P<model_name>\w{1,256})/(?P<weight_name>\w{1,256})/$',keras_weights.upload_weights),
]
