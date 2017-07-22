from ..utils import *
from ..models import PredictTask,TrainTask,TaskSuccessor
from ..broker.KerasModelBroker import KerasModelBroker
from ..broker.DatasetBroker import DatasetBroker
from ..broker.KerasModelWeightsBroker import KerasModelWeightsBroker
import numpy as np
import json
from keras.optimizers import *
from keras.losses import *
from ..serializer import PredictTaskSerializer,TrainTaskSerializer

class TaskBroker(object):
    ERROR_CREATE_ARRAY = 1
    ERROR_ARRAY_SHAPE_MISMATCH = 2
    ERROR_TASK_ALREADY_EXISTS = 3
    ERROR_INVALID_CONFIG = 4
    ERROR_CREATE_TASK = 5
    ERROR_MODEL_DATASET_SHAPE_MISMATCH = 6
    ERROR_TASK_DOES_NOT_EXIST = 'ERROR_TASK_DOES_NOT_EXIST'
    def get_task_by_name(self,name,serialize = False):
        try:
            result = PredictTask.objects.get(name = name)
            return result if not serialize else PredictTaskSerializer(result)
        except:
            try:
                result = TrainTask.objects.get(name = name)
                return result if not serialize else TrainTaskSerializer(result)
            except:
                return None
    def get_tasks(self,serialize = False):
        predicts,trains = PredictTask.objects.all(),TrainTask.objects.all()
        return (predicts,trains) if not serialize else (PredictTaskSerializer(predicts,many = True),TrainTaskSerializer(trains,many = True))

    '''
        Create a predict task
    '''
    def add_predict_task(self,model_name,weight_name,task_name,inp_str):
        inp_arr =  self.array_from_json(inp_str)
        if inp_arr is None:
            return TaskBroker.ERROR_CREATE_ARRAY

        inp_arr = self.prepare_model_input(model_name,inp_str)
        if not inp_arr:
            return TaskBroker.ERROR_ARRAY_SHAPE_MISMATCH

        if PredictTask.objects.filter(name = task_name).exists():
            return TaskBroker.ERROR_TASK_ALREADY_EXISTS

        weight_model,error = KerasModelWeightsBroker().get_weights_model(model_name,weight_name,False)
        if error:
            return error

        try:
            task = PredictTask(name = task_name,weight = weight_model,input = inp_arr)

            task.save()
        except:
            return TaskBroker.ERROR_CREATE_TASK

    '''
        Auxillary function for parsing the JSON string of the configuration of the training task. This includes the parameters for model.compile(...) call (e.g. 'loss','optimizers') and model.fit(...) call (e.g. 'epoch','batch_size','callbacks')
    '''
    def parse_config(self,config_str):
        try:
            config = json.loads(config_str)
        except:
            return None

        # check compulsory keys
        if not contains_key(config,['optimizer','loss','batch_size','epoch']):
            return None

        # 1. check optimizer config
        # 1.1 expect name , lr, (decay) exists
        optimizer_required_fields = ['lr','name']
        if not contains_key(config['optimizer'],optimizer_required_fields):
            return None

        # 1.2 check name is one of the known optimizer name.
        optimizer_names = ['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam']
        if not contains_one_of(config['optimizer']['name'],optimizer_names):
            return None

        # 1.3 retrieve optimizer class
        # convert the string to Keras optimizer class
        # TODO: should be safe to eval. But is this for sure?
        optimizer = eval(config['optimizer']['name'])

        # 1.4 get Optimizer instance
        # delete the name entry before applying to the optimizer constructor, because 'name' is not one of the parameters
        del config['optimizer']['name']
        # apply all parameters into the optimizer. If extra parameters are given the values will be detected here. 
        try:
            optimizer = apply_dict(optimizer,config['optimizer'])
        except:
            return None

        # 2. check loss function
        losses = [
                    'mse','mean_squared_error',
                    'mae','mean_absolute_error',
                    'mean_absolute_percentage_error',
                    'mean_squared_logarithmic_error',
                    'squared_hinge',
                    'hinge',
                    'categorical_hinge',
                    'logcosh',
                    'categorical_crossentropy',
                    'sparse_categorical_crossentropy',
                    'binary_crossentropy',
                    'kullback_leibler_divergence',
                    'poisson',
                    'cosine_proximity',
               ]
        if not contains_one_of(config['loss'],losses):
           return None

        loss = config['loss']

        # 3. check epoch
        try:
            epoch = int(config['epoch'])
        except:
            return None # invalid epoch: cannot be converted to int

        # 4. check batch_size
        try:
            batch_size = int(config['batch_size'])
        except:
            return None # invalid batch_size: cannot be converted to int

        # 5. check additional callbacks
        callbacks = None
        if 'callbacks' in config:
            callbacks = [
                    'BaseLogger',
                    'TerminateOnNaN',
                    'ProgbarLogger',
                    'History',
                    'ModelCheckpoint',
                    'EarlyStopping',
                    #'RemoteMonitor',
                    'LearningRateScheduler',
                    #'TensorBoard',
                    'ReduceLROnPlateau',
                    #'CSVLogger',
                    #'LambdaCallback', # not supported at the moment
            ]
            # 5.1 check if names in config['callbacks'] and they are one of the recognized callbacks
            if not all('name' in callback and callback['name'] in callbacks for callback in config['callbacks']) or not type(config['name']) == list:
                return None

            # 5.2 retrieve callback constructors
            callback_constructors = [eval(callback['name']) for callback in config['callbacks']]

            # 5.3 remove callback['name'] and apply parameters to constructors
            callbacks = []
            try:
                for callback in config['callbacks']:
                    del callback['name']
                    callbacks.append(apply_dict(callback_constructors[i],callback))
            except:
                pass # some arguments for the callbacks are invalid, but that is fine
        # 6. return all parsed parameters
        result = {
                    'compile':{'loss':loss,'optimizer': optimizer},
                    'fit': {'epoch': epoch,'batch_size': batch_size}
                }
        if callbacks:
            result['fit']['callbacks'] = callbacks
        return result

    '''
        Create a training task
    '''
    def add_train_task(self,model_name,weight_name,dataset_name,task_name,config_str,from_task = None):
        # check task name
        train_model = self.get_task_by_name(task_name)
        if train_model:
            return TaskBroker.ERROR_TASK_ALREADY_EXISTS

        # check model and weight
        weight_model, error = KerasModelWeightsBroker().get_weights_model(model_name,weight_name,serialize = False)
        if error:
            return error

        # check dataset
        dataset_model, error = DatasetBroker().get_dataset(dataset_name,get_file = False)
        if error:
            return error

        # check dataset dimension and that of the model's...
        model_in_shape,model_out_shape = KerasModelBroker().get_io_shapes(model_name)
        data_in_shape,data_out_shape = DatasetBroker().get_io_shapes(dataset_name)
        if not is_model_data_shape_consistent(model_in_shape,model_out_shape,data_in_shape,data_out_shape):
            return TaskBroker.ERROR_MODEL_DATASET_SHAPE_MISMATCH

        # TODO: also add the errors from the above
        # check config string...
        config = self.parse_config(config_str)
        if not config:
            return TaskBroker.ERROR_INVALID_CONFIG
        # otherwise add it to the database
        try:
            task = TrainTask(name = task_name,config = config_str,dataset = dataset_model,weight = weight_model)
            task.save()
        except Exception as e:
            return TaskBroker.ERROR_CREATE_TASK
        # and take care of the Task successor issue
        if from_task:
            successor = self.get_task_by_name(from_task)
            if not successor:
                return # no big deal here
            relation = TaskSuccessor(successor,task)
            relation.save()

    '''
        Evaluate the model right away...
    '''
    def evaluate(self,model_name,weight_name,inp_str):
        inp_arr =  self.array_from_json(inp_str)
        if not inp_err:
            return (None,TaskBroker.ERROR_CREATE_ARRAY)

        inp_arr = self.prepare_model_input(model_name,inp_str)
        if not inp_arr:
            return (None,TaskBroker.ERROR_ARRAY_SHAPE_MISMATCH)
        model,error = KerasModelWeightsBroker.load_model_weight_weights(model,weight_name)
        if error:
            return (None,error)
        result = model.predict(inp_arr)
        return (result,None)

    '''
        Auxillary function for checking whether a serialized array can be fed into the model with provided name.

        This function assumes that the model already exists and the input string is a valid JSON containing a numpy array. If it is not the case exceptions will be thrown.

        The expected model input shapes will be (None,....) or [(None,...),(None,...)]
        To determine the input array can be fed into the model:
             -First check the input shape of the model.
                -If the model input shape is a list...
                    -Get total number of inputs expected by the model
                    -Parse the input array, check the first level
                    -If the first level does not have the same number of element as the number of input, they do not match
                    -Check the corresponding input to the inner array.
                    -If any does not match, the whole thing does not match
                    -Check the first dim. of each sub-array, if match, then OK
                -Otherwise, check the shape matches...

    '''
    def prepare_model_input(self,model_name,inp_str):
        def is_corresponding_input_OK(expected_shape,arr):
            arr = np.array(arr)
            dif = len(arr.shape) - len(expected_shape)
            if dif == 0:
                return arr.shape[1:] == expected_shape[1:]
            elif dif == -1:
                return arr.shape == expected_shape[1:]
            else:
                return False
        def as_model_input(expected_shape,arr):
            arr = np.array(arr)
            dif = len(arr.shape) - len(expected_shape)
            if dif == 0:
                return arr
            elif dif == -1:
                return np.expand_dims(arr,0)

        inp_shape,out_shape = KerasModelBroker().get_io_shapes(model_name)
        input_arr = self.array_from_json(inp_str)
        if type(input_arr) != list: # check if it is structured
            return None
        if type(inp_shape) == list: # 1.1
            num_input = len(inp_shape)
            if num_input != len(input_arr):
                return None

            if not all(is_corresponding_input_OK(expected_shape,sub_array) for expected_shape,sub_array in zip(inp_shape,input_arr)):
                return None
            inputs = [as_model_input(expected_shape,arr) for expected_shape,arr in zip(inp_shape,input_arr)]
            if not reduce(lambda a,b: a if a == b else None,[arr.shape[0] for arr in inputs]):
                return None
            return [inp.tolist() for inp in inputs] # return to normal list for retrieval later on
        else:
            inputs = as_model_input(inp_shape,input_arr)
            if not inputs:
                return None
            if inputs.dtype == object:
                return None # ragged array
            # should be fine then..
            return inputs
    def array_from_json(self,json_str):
        try:
            arr = json.loads(json_str)
            return arr
        except:
            return None

    def mark_as_completed(self,name,isFalse = False):
        task = self.get_task_by_name(name)
        if not task:
            print 'no such task'
            return TaskBroker.ERROR_TASK_DOES_NOT_EXIST
        try:
            task.completed = not isFalse
            task.save()
        except:
            print 'unknown error'
            return TaskBroker.ERROR_TASK_DOES_NOT_EXIST
