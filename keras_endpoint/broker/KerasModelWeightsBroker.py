from ..utils import *
from ..models import KerasModelWeights
from ..serializer import KerasModelWeightsSerializer
from .KerasModelBroker import KerasModelBroker

'''
    Provide methods for interacting with KerasModelWeights models.
    All transations invoving this model should be done with instances of this class.
'''
class KerasModelWeightsBroker(object):
    ERROR_NO_SUCH_MODEL = 1
    ERROR_NO_SUCH_WEIGHT_FILE = 2
    ERROR_CANNOT_CREATE_MODEL = 3
    ERROR_LOADING_WEIGHTS = 4

    '''
        Retrieve all weight model objects by model name and optionally weight name.
        If file name is given, the method will return specified weight model with given model name and file
        otherwise it will return all weight models of the given model

        A tuple of (result,failed reasons) will be returned at all times,
        of which the result will be `None` if one of the following occur:
            1. There is no such model with the name provided
            2. There is no such weight file with the name provided

        If none of the above happens, `result` will not be `None` and failed_reasons will be `None`.

        If `serialize` is set to `True`, `result` will be serialized using `KerasModelWeightsSerializer`
        otherwise query results will be returned directly.
    '''
    def get_weights_model(self,model_name,file_name = None,serialize = True):
        # check if the related model exists
        if not KerasModelBroker().exists(model_name):
            return (None, KerasModelWeightsBroker.ERROR_NO_SUCH_MODEL)
        # if weight file name is not provided, return all weight files associated to the model with provided name
        if not file_name:
            result = KerasModelWeights.objects.filter(name = model_name)
            return (result if not serialize else KerasModelWeightsSerializer(result,many = True).data,None)
        # return specific weight file with provided model name and file name
        # will raise error if the specified weight file does not exist
        else:
            result = KerasModelWeights.objects.filter(name = model_name,weight_name = file_name)
            if not result.exists():
                return (None,KerasModelWeightsBroker.ERROR_NO_SUCH_WEIGHT_FILE)
            return (result if not serialize else KerasModelWeightsSerializer(result).data,None)

    '''
        Return the content of weight files given weight model objects
        This is supposed to be used fo loading weights for Keras models programmatically
    '''
    def get_weight_file(self,weight_models):
        pass
    def num_weights_file(self,model_name):
        result,reason = self.get_weights_file(mode_name,serialize = False)
        return (0,reason) if reason else (result.count(),None)

    def load_model_with_weights(self,model_name,file_name):
        # try to get the weight model
        weight_model,reason = self.get_weights_file(model_name,file_name,serialize = False)
        if reason:
            return (None,reason)
        model = KerasModelBroker().get_model_instance(model_name)
        if not model:
            return (None,KerasModelWeightsBroker.ERROR_CANNOT_CREATE_MODEL)
        try:
            model.load_weights(weight)
            return (model,None)
        except Exception as e:
            return (None,KerasModelWeightsBroker.ERROR_LOADING_WEIGHTS)

