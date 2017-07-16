# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from keras_endpoint.models import Task,PredictTask,TrainTask
from django.db import models

# Create your models here.

class Result(models.Model):
    class Meta:
        abstract = True
    date_created = models.DateTimeField(auto_now_add = True)
class PredictResult(Result):
    result = models.TextField()
    of_task = models.OneToOneField(PredictTask,primary_key = True,to_field = 'name')
class TrainResult(Result):
    result = models.FileField()
    of_task = models.OneToOneField(TrainTask,primary_key = True,to_field = 'name')
