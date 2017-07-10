# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

class KerasModel(models.Model):
    name = models.CharField(max_length = 256,primary_key = True)
    date_created = models.DateTimeField(auto_now_add = True)
    definition = models.TextField(blank = True,null = True)

class KerasModelWeights(models.Model):
    model = models.ForeignKey(KerasModel)
    date_created = models.DateTimeField(auto_now_add = True)
    weight_name = models.CharField(max_length = 256,unique = True)
    weight_file = models.FileField()

