from ..utils import *
from ..models import KerasModel
from ..serializer import KerasModelSerializer

class KerasModelBroker(object):
    # list of errors that will occur in the following instance methods
    ERROR_ALREADY_EXISTS = 1
    ERROR_INVALID_DEFINITION = 2

    def num_model(self):
        return KerasModel.objects.all().count()

    def get(self,name,serialize = True):
        if not name:
            result = KerasModel.objects.all()
            return result if not serialize else KerasModelSerializer(result,many = True).data
        else:
            try:
                result = KerasModel.objects.get(name = name)
                return result if not serialize else KerasModelSerializer(result).data
            except:
                return None

    def exists(self,name):
        return self.get(name,False) is not None

    def create(self,name,definition):
        if self.exists(name):
            return (False,KerasModelBroker.ERROR_ALREADY_EXISTS)
        if not is_json_valid_keras_model(definition):
            return (False,KerasModelBroker.ERROR_INVALID_DEFINITION)
        new_model = KerasModel(name = name,definition = definition)
        new_model.save()
        return (True,None)

    def get_model_instance(self,name):
        model = self.get(name,False)
        if not model:
            return None
        return create_model_from_json(json_str)

