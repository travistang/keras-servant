import json
from keras_endpoint.models import PredictTask,TrainTask
from keras_endpoint.utils import numpy_to_json,json_to_numpy
from ..models import *

class ResultBroker(object):
    ERROR_RESULT_FOR_TASK_ALREADY_EXISTS = 'ResultBroker.ERROR_RESULT_FOR_TASK_ALREADY_EXISTS'
    ERROR_TASK_TYPE_MISMATCH = 'ResultBroker.ERROR_TASK_TYPE_MISMATCH'
    ERROR_CREATE_RESULT = 'ResultBroker.ERROR_CREATE_RESULT'
    ERROR_INVALID_RESULT = 'ResultBroker.ERROR_INVALID_RESULT'

    def create_predict_result_for_task(self,task,result_arr):
        if PredictResult.objects.filter(of_task = task).exists():
            return ResultBroker.ERROR_RESULT_FOR_TASK_ALREADY_EXISTS
        if type(task) != PredictTask:
            return ResultBroker.ERROR_TASK_TYPE_MISMATCH

        result_str = numpy_to_json(result_arr) if type(result_arr) is not list else [numpy_to_json(arr) for arr in result_arr] # handling single output vs multiple output models
        if not result_str:
            return  ResultBroker.ERROR_INVALID_RESULT

        try:
            result_model = PredictResult(of_task = task,result = result_str)
            result_model.save()
        except:
            return ResultBroker.ERROR_CREATE_RESULT
