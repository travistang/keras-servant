from ..models import KerasModelWeights
from ..broker.KerasModelWeightsBroker import KerasModelWeightsBroker
from ..utils import *
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import FileResponse

'''
    entry point that serves the information about models in 2 ways:
        1. When the client request weights/ point as well as the model name but did not specify the name of the weight, a "summary" of the weights of the model will be returned.
        2. When the client specifies the name of the weight model as well, then the weight file (.hf5) will be served.

'''
@api_view(['GET'])
def weights_get(request,model_name,file_name = None):
    broker = KerasModelWeightsBroker()
    result,reason = broker.get_weights_model(model_name,file_name) if not file_name else broker.get_weight_file(model_name,file_name)
    if not reason:
        # TODO: selectively return...
        return Response(result) if not file_name else serve_file(result,result.name,'Application/X-hdf')
    elif reason == KerasModelWeightsBroker.ERROR_NO_SUCH_MODEL:
        return error_json_response_with_details("no such model: {}".format(model_name))
    else:
        return error_json_response_with_details("weight file named {} for model {} not found".format(file_name,model_name))

'''
    Entry point for uploading weight (.hf5) files.
'''
@api_view(['POST'])
def upload_weights(request,model_name,weight_name):
    weight_file = request.FILES['file']
    if not weight_file:
        return error_json_response_with_details("No weight file uploaded")

    result = KerasModelWeightsBroker().add_weights(model_name,weight_file,weight_name)

    if not result:
        return create_success_with_data([])
    elif result == KerasModelWeightsBroker.ERROR_NO_SUCH_MODEL:
        return error_json_response_with_details("no model called {}".format(model_name))
    elif result == KerasModelWeightsBroker.ERROR_LOADING_WEIGHTS:
        return error_json_response_with_details("Weight uploaded does not match with the target Keras model")
    elif result == KerasModelWeightsBroker.ERROR_SAVING_WEIGHTS:
        return error_json_response_with_details("Error occured when trying to save the weight. This is probably because there is already a weight file with the same name associated to this model or running out of space.")

    elif result == KerasModelWeightsBroker.ERROR_WEIGHT_ALREADY_EXISTS:
        return error_json_response_with_details("weight with name {} for model {} already exists".format(weight_name,model_name))
    else:
        return error_json_response_with_details("Uploading failed with unknown reasons")

