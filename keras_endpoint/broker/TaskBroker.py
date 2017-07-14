from ..utils import *
from ..models import PredictTask,TrainTask,TaskSuccessor
from ..broker.KerasModelBroker import KerasModelBroker
from ..broker.DatasetBroker import DatasetBroker
from ..broker.KerasModelWeightsBroker import KerasModelWeightsBroker
import numpy as np
import json
from keras.optimizers import *
from keras.losses import *

class TaskBroker(object):
    ERROR_CREATE_ARRAY = 1
    ERROR_ARRAY_SHAPE_MISMATCH = 2
    ERROR_TASK_ALREADY_EXISTS = 3
    ERROR_INVALID_CONFIG = 4
    ERROR_CREATE_TASK = 5
    ERROR_MODEL_DATASET_SHAPE_MISMATCH = 6

    def get_task_by_name(self,name):
        try:
            return PredictTask.objects.get(name = name)
        except:
            try:
                return TrainTask.objects.get(name = name)
            except:
                return None
    '''
        Create a predict task
    '''
    def add_predict_task(self,model_name,weight_name,task_name,inp_str):
        inp_arr =  self.array_from_json(inp_str)
        if not inp_arr:
            return TaskBroker.ERROR_CREATE_ARRAY

        inp_arr = self.prepare_model_input(model_name,inp_str)
        if not inp_arr:
            return TaskBroker.ERROR_ARRAY_SHAPE_MISMATCH

        if PredictTask.objects.filter(name = task_name).exists():
            return TaskBroker.ERROR_TASK_ALREADY_EXISTS

        weight_model,error = KerasModelWeightsBroker().get_weights_model(model_name,weight_name,False)
        print type(weight_model)
        if error:
            return error

        try:
            task = PredictTask(name = task_name,weight = weight_model,input = inp_str)

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
            if not all('name' in callback and callback['name'] in callbacks for callback in config['callbacks']):
                return None

            # 5.2 retrieve callback constructors
            callbacks_constructors = [eval(callback['name']) for callback in config['callbacks']]

            # 5.3 remove callback['name'] and apply parameters to constructors
            callbacks = []
            for callback in config['callbacks']:
                del callback['name']
                callbacks.append(apply_dict(callback_constructors[i],callback))

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
            raise e
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

        To determine the input array can be fed into the model:
            1. First check the dimension of the array. Let dim(x) be the dimension of x. Then dim(inp_arr) == dim(model_input) || dim(inp_arr) == dim(model_input) - 1.
            1.1 If dim(inp_arr) == dim(model_input), then inp_arr[0].shape == model_input.shape[1:]
            1.2 If dim(inp_arr) == dim(model_input) - 1, then inp_arr.shape == model_input.shape[1:]
            2.1 If dim(inp_arr) == dim(model_input), then return inp_arr
            2.2 If dim(inp_arr) == dim(model_input) - 1, then return inp_arr.expand_dims(0)
            3. If None of the case above is true, return `None`

    '''
    def prepare_model_input(self,model_name,inp_str):
        inp_shape,out_shape = KerasModelBroker().get_io_shapes(model_name)
        input_arr = self.array_from_json(inp_str)
        arr_shape = input_arr.shape
        arr_dim = len(arr_shape)
        inp_dim = len(inp_shape)

        if arr_dim == inp_dim and arr_shape[1:] == inp_shape[1:]:
            return input_arr
        if arr_dim == inp_dim - 1 and arr_shape == inp_shape[1:]:
            return np.expand_dims(0,input_arr)
        # neighter of the case above is true. The array cannot be fed into the model
        return None

    def array_from_json(self,json_str):
        try:
            arr = json.loads(json_str)
            return np.array(arr)
        except:
            return None
