from django.conf.urls import url
from ..api import dataset
urlpatterns = [
    url(r'^get/$',dataset.get),
    url(r'^get/(?P<dataset_name>\w{1,256})/$',dataset.get),
    url(r'^upload/(?P<dataset_name>\w{1,256})/$',dataset.upload),

]
