from keras.layers import *
from keras.models import *
from rest_framework.response import Response
from rest_framework import status

def empty_json_response():
    return Response({},status = status.HTTP_400_BAD_REQUEST)
def error_json_response_with_details(details):
    return Response({'detail': details},status = status.HTTP_400_BAD_REQUEST)

def error_json_response(details):
    return Response(details,status = status.HTTP_400_BAD_REQUEST)

def create_success_with_data(data):
    return Response(data,status = status.HTTP_201_CREATED)

def is_json_valid_keras_model(json_str):
    # check param type
    if not type(json_str) in [unicode,str]: return False
    # allow empty string
    if len(json_str.strip()) == 0: return True
    try:
        model_from_json(json_str)
        return True
    except Exception as e:
        return False

def create_model_from_json(json_str):
    if not is_json_valid_keras_model(json_str):
        return None
    return model_from_json(json_str)
