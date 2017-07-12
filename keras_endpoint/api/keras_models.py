# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.db import models
from django.http import HttpResponse,JsonResponse

from ..models import KerasModel
from ..serializer import KerasModelSerializer

from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from ..utils import *
from ..broker.KerasModelBroker import KerasModelBroker

@api_view(['GET'])
def models_get(request,model_name = None):
    # if not model names are supplied, return a list of all models
    result = KerasModelBroker().get(model_name)
    return Response(result) if result else error_json_response_with_details("Model with name {} does not exists".format(model_name))

@api_view(['POST'])
def models_save(request):
    # check data structure
    serializer = KerasModelSerializer(data = request.data)
    if not serializer.is_valid():
        return error_json_response(serializer.errors)

    # checking passed. checking model defs
    model_def = serializer.validated_data['definition']
    model_name = serializer.validated_data['name']
    success,reason = KerasModelBroker().create(model_name,model_def)
    if success:
        return create_success_with_data(serializer.data)
    elif reason is KerasModelBroker.ERROR_ALREADY_EXISTS:
        return error_json_response_with_details("Model with name {} already exists".format(model_name))
    else:
        return error_json_response_with_details("Model definition is not valid")
