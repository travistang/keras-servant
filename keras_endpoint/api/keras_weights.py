from ..models import KerasModelWeights
from ..broker.KerasModelWeightsBroker import KerasModelWeightsBroker
from ..utils import *
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

@api_view(['GET'])
def weights_get(request,model_name,file_name = None):
    result,reason = KerasModelWeightsBroker().get_weights_file(model_name,file_name)
    if not reason:
        return Response(result)
    elif reason == KerasModelWeightsBroker.ERROR_NO_SUCH_MODEL:
        return error_json_response_with_details("no such model: {}".format(model_name))
    else:
        return error_json_response_with_details("weight file named {} for model {} not found".format(file_name,model_name))



