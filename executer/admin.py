# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin
from .models import PredictResult,TrainResult
# Register your models here.
admin.site.register(PredictResult)
admin.site.register(TrainResult)
