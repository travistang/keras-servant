from ..models import Task,TrainTask,TaskSuccessor
from rest_framework.decorators import api_view
from rest_framework.response import Response
from ..broker.KerasModelWeightsBroker import KerasModelWeightsBroker
from ..broker.TaskBroker import TaskBroker
from ..utils import *
import json

@api_view(['POST'])
def create_predict_task(request,model_name,weight_name):
    # validate POST parameter
    if 'input' not in request.POST or 'name' not in request.POST:
        return error_json_response_with_details("Submission of prediction tasks requires \"input\" field and \"name\" field.")
    # extract relevant post parameter
    input_array = request.POST['input']
    task_name = request.POST['name']

    broker = TaskBroker()
    # see if the client requests to evaluate the result now
    if 'eval_now' in request.POST and request.POST['eval_now'].lower() == 'true':
        # result will be evaluated now...
        result,error = broker.evaluate(model_name,weight_name,input_array)
        if not error:
            return JsonResponse(json.dumps(result.tolist()))
    else:
        error = broker.add_predict_task(model_name,weight_name,task_name,input_array)

    # handle and report the errors here
    if not error:
        return create_success_with_data([])
    elif error == TaskBroker.ERROR_ARRAY_SHAPE_MISMATCH:
        return error_json_response_with_details("Input array can not be fed into the Keras model")
    elif error == TaskBroker.ERROR_CREATE_ARRAY:
        return error_json_response_with_details("Input JSON cannot be converted to NumPy array")
    else:
        return error_json_response_with_details("Fail save or evaluate task")


@api_view(['POST'])
def create_train_task(request,dataset_name,model_name,weight_name = None):
    if not contains_key(request.POST,['config','name']):
        return error_json_response_with_details("Submission of training tasks requires \"config\"  and \"name" "field")
    config = request.POST['config']
    task_name = request.POST['name']

    broker = TaskBroker()
    error = broker.add_train_task(model_name,weight_name,dataset_name,task_name,config)
    if not error:
        return create_success_with_data([])
    # TODO: handle possible errors here
    if error == TaskBroker.ERROR_CREATE_TASK:
        return error_json_response_with_details("Cannot create new task")
    if error == TaskBroker.ERROR_INVALID_CONFIG:
        return error_json_response_with_details("Invalid config JSON")
    if error == TaskBroker.ERROR_TASK_ALREADY_EXISTS:
        return error_json_response_with_details("Task already exists")
    if error == TaskBroker.ERROR_MODEL_DATASET_SHAPE_MISMATCH:
        return error_json_response_with_details("The input/output shape of the given model does not match with the shape of the given dataset")
    return error_json_response_with_details("Unknown error")
