from django.conf.urls import url
from ..api import task

urlpatterns =[
    url(r'^$',task.get_tasks),
    url(r'^(?P<name>\w{1,256})/$',task.get_tasks),
]
