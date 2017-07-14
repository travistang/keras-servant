from django.conf.urls import url
from ..api import task

urlpatterns = [
    url(r'^predict/(?P<model_name>\w{1,256})/(?P<weight_name>\w{1,256})/$',task.create_predict_task),
    url(r'^train/(?P<dataset_name>\w{1,256})/(?P<model_name>\w{1,256})/$',task.create_train_task),
    url(r'^train/(?P<dataset_name>\w{1,256})/(?P<model_name>\w{1,256})/(?P<weight_name>\w{1,256})/$',task.create_train_task),
]
