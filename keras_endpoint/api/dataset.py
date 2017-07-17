from ..models import Dataset
from ..broker.DatasetBroker import DatasetBroker
from ..serializer import DatasetSerializer
from ..utils import *
from rest_framework.decorators import api_view
from rest_framework.response import Response


@api_view(['GET'])
def get(request,dataset_name = None):
    broker = DatasetBroker()
    if not dataset_name:
        return Response(broker.get_all_dataset_name())
    else:
        result,reason = broker.get_dataset(dataset_name)
        if not reason:
            return serve_file(result,dataset_name,'Application/X-hdf')
        else:
            return error_json_response_with_details("No dataset named {}".format(dataset_name))

@api_view(['POST'])
def upload(request,dataset_name):
    f = request.FILES['file']
    error = DatasetBroker().add(f,dataset_name)
    if error:
        if error == DatasetBroker.ERROR_DATASET_ALREADY_EXISTS:
            return error_json_response_with_details('dataset with name {} already exists'.format(dataset_name))
        if error == DatasetBroker.ERROR_INVALID_DATASET:
            return error_json_response_with_details('dataset uploaded has an unexpected format')
        if error == DatasetBroker.ERROR_INVALID_DATASET_NAME:
            return error_json_response_with_details('Cannot add a dataset with given name')
        if error == DatasetBroker.ERROR_UNABLE_TO_CREATE_DATASET:
            return error_json_response_with_details('Cannot create dataset')
        if error == DatasetBroker.ERROR_SAVING_DATASET:
            return error_json_response_with_details('Cannot save dataset')
        return error_json_response_with_details("Unknown error occured")

    return create_success_with_data([])
