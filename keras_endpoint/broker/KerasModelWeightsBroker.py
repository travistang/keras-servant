from ..utils import *
from ..models import KerasModelWeights
from ..serializer import KerasModelWeightsSerializer
from .KerasModelBroker import KerasModelBroker
from ..keras_verifier import *
from django.core.files import File
from os.path import isfile,isdir,exists,dirname,splitext
from os import makedirs,remove
'''
    Provide methods for interacting with KerasModelWeights models.
    All transations invoving this model should be done with instances of this class.
'''
class KerasModelWeightsBroker(object):
    ERROR_NO_SUCH_MODEL = 1
    ERROR_NO_SUCH_WEIGHT_FILE = 2
    ERROR_CANNOT_CREATE_MODEL = 3
    ERROR_LOADING_WEIGHTS = 4
    ERROR_SAVING_WEIGHTS = 5
    ERROR_WEIGHT_ALREADY_EXISTS = 6
    '''
        Retrieve all weight model objects by model name and optionally weight name.
        If file name is given, the method will return specified weight model with given model name and file
        otherwise it will return the name of all the weights of the given model

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
            result = KerasModelWeights.objects.filter(model = model_name)
            return (result if not serialize else KerasModelWeightsSerializer(result,many = True).data,None)
        # return specific weight file with provided model name and file name
        # will raise error if the specified weight file does not exist
        else:
            result = KerasModelWeights.objects.filter(model = model_name,weight_name = file_name)
            if not result.exists():
                return (None,KerasModelWeightsBroker.ERROR_NO_SUCH_WEIGHT_FILE)
            # because when it comes to here there should be exactly one result in the query set. So simply return the first one here.
            return (result[0] if not serialize else KerasModelWeightsSerializer(result).data,None)

    '''
        Return the file object for weight files given weight model name and file_name
        This is supposed to be used fo loading weights for Keras models programmatically
    '''
    def get_weight_file(self,model_name,file_name):
        weight_model, reason = self.get_weights_model(model_name,file_name,False)
        if reason:
            return (None, reason)
        return (weight_model.weight_file.file,None)
    '''
        Return the number of weight models associated to the given model
    '''
    def num_weights_file(self,model_name):
        result,reason = self.get_weights_file(mode_name,serialize = False)
        return (0,reason) if reason else (result.count(),None)

    '''
        Return a keras model with loaded with specified weight filename.
    '''
    def load_model_with_weights(self,model_name,file_name):
        # try to get the weight model
        model = KerasModelBroker().get_model_instance(model_name)
        if not model:
            return (None,KerasModelWeightsBroker.ERROR_NO_SUCH_MODEL)
        # try to get the weight file
        weight_file, reason = self.get_weight_file(model_name,file_name)
        if reason:
            return (None,reason)
        try:
            model.load_weights(weight_file)
            return (model,None)
        except Exception as e:
            return (None,KerasModelWeightsBroker.ERROR_LOADING_WEIGHTS)

    '''
        Handle weights uploading.

        It takes the name of a saved Kerasl Model as well as the File instance retrieved from the request (e.g. request.FILE[...])
        and tries to add new KerasModelWeights instance accordingly.

        If the new object can be added to the database successfully, the method will return `None`, otherwise it will return `KerasModelWeightsBroker.ERROR_SAVING_WEIGHTS`
    '''
    def add_weights(self,model_name,weight_file,file_name):
        model = KerasModelBroker().get(model_name,False)

        if not model:
            return KerasModelWeightsBroker.ERROR_NO_SUCH_MODEL

        # check if weight with the same name exists
        if KerasModelWeights.objects.filter(model = model,weight_name = file_name).exists():
            return KerasModelWeightsBroker.ERROR_WEIGHT_ALREADY_EXISTS

        # create weight model object
        weight_model = KerasModelWeights(model = model,weight_name = file_name)
        # create weight file itself
        weight_file_path = 'weights/{}/{}.h5'.format(model_name,file_name)
        # try not to the the client write anywhere other than the weights/ directory...
        if not is_subdir(weight_file_path,'weights/'):
            return KerasModelWeightsBroker.ERROR_SAVING_WEIGHTS

        # check if the file in the same location exists
        # if yes, that means a weight file with the same name for the same model has been created before
        if isfile(weight_file_path):
            return KerasModelWeightsBroker.ERROR_WEIGHT_ALREADY_EXISTS

        # create the directories if they are not exist yet
        create_dir_if_not_exists(weight_file_path)
        # copy the source file to dest
        with open(weight_file_path,'w+') as storage_file:
            if not copy_file(weight_file,storage_file):
                return KerasModelWeightsBroker.ERROR_SAVING_WEIGHTS

        # close and delete the temporary file
        weight_file.close()

        # verify the file is indeed a valid weight file for the given model
        if not verify_weight(model,weight_file_path):
            remove(weight_file_path)
            return KerasModelWeightsBroker.ERROR_LOADING_WEIGHTS

        #weight_model.weight_file.save(file_name,File(storage_file))
        weight_model.weight_file.name = weight_file_path
        # everything is fine. Save the object and return
        weight_model.save()

