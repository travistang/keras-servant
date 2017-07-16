# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

class KerasModel(models.Model):
    name = models.CharField(max_length = 256,primary_key = True)
    date_created = models.DateTimeField(auto_now_add = True)
    definition = models.TextField(blank = True,null = True)

class KerasModelWeights(models.Model):
    model = models.ForeignKey(KerasModel,on_delete = models.CASCADE)
    date_created = models.DateTimeField(auto_now_add = True)
    weight_name = models.CharField(max_length = 256)
    weight_file = models.FileField()

class Dataset(models.Model):
    name = models.CharField(max_length = 256,unique = True)
    date_created = models.DateTimeField(auto_now_add = True)
    dataset_file = models.FileField()

'''
    A task object represents a setting which is necessary for a prediction task.
'''
class Task(models.Model):
    name = models.CharField(max_length = 512,primary_key = True)
    weight = models.ForeignKey(KerasModelWeights, on_delete = models.CASCADE, null = True)
    date_created = models.DateTimeField(auto_now_add = True)
    completed = models.BooleanField(default = False)
    callback_url = models.URLField(null = True)

    class Meta:
        abstract = True

class PredictTask(Task):
    input = models.TextField()

'''
    A TrainTask object represents a setting that is essential for training a Keras model.

    The `weight` field becomes the representation of the initial weight.

    The `config` field is expected to be a JSON string that provides info about the the optimizers and loss function used.

    An example JSON for the `config` field is as follows:
        {
            "optimizer":
            {
                "name": "Adam",
                "lr": 1e-3,
                "decay": 1e-6,
            },

            "loss":
            {
                "name": "mse",
            },

            "callbacks":
            [
                {
                    "name": "CSVLogger",
                    "path": ...
                },
                ...
            ]
        }
'''

class TrainTask(Task):
    config = models.TextField()
    dataset = models.ForeignKey(Dataset,on_delete = models.CASCADE)

class TaskSuccessor(models.Model):
    previous_task = models.ForeignKey(TrainTask, related_name = 'previous_task')
    next_task = models.ForeignKey(TrainTask,related_name = 'next_task')

