# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin
from .models import *
# Register your models here.
admin.site.register(KerasModel)
admin.site.register(KerasModelWeights)
admin.site.register(Dataset)
admin.site.register(PredictTask)
admin.site.register(TrainTask)
admin.site.register(TaskSuccessor)
