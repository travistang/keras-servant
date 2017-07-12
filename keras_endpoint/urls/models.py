from django.conf.urls import url
from ..api import keras_models

urlpatterns = [
    url(r'^get/(?P<model_name>\w{0,256})/$', keras_models.models_get),
    url(r'^$',keras_models.models_get),
    url(r'^create/$',keras_models.models_save),
]
