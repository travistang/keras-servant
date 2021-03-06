from .broker.ResultBroker import ResultBroker
from keras_endpoint.broker.TaskBroker import TaskBroker
from keras_endpoint.models import *
from .models import PredictResult,TrainResult
from keras_endpoint.broker.DatasetBroker import DatasetBroker
from keras_endpoint.broker.KerasModelBroker import KerasModelBroker
from keras_endpoint.broker.KerasModelWeightsBroker import KerasModelWeightsBroker
from django.db.models import Min,Max
from keras.models import model_from_json
from keras_endpoint.utils import *
import numpy as np
from random import shuffle
import h5py
import json
'''
    Auxillary functions
    Do not pass these to the crontab...
'''
def get_oldest_task():
    # get the "oldest" date_created from both PredictTask and TrainTask
    # then check if any doesnt exists, if not return None
    # both incomplete predict task and train task exists
    oldest_predict_task = PredictTask.objects.filter(completed = False).aggregate(Min('date_created'))['date_created__min']
    oldest_train_task = TrainTask.objects.filter(completed = False).aggregate(Min('date_created'))['date_created__min']

    if not oldest_predict_task and not oldest_train_task:
            return None
    if not oldest_predict_task: # have train task
        return TrainTask.objects.filter(completed = False,date_created = oldest_train_task)[0]
    if not oldest_train_task:
        return PredictTask.objects.filter(completed = False,date_created = oldest_predict_task)[0]
    # both tasks exists, get the oldest one
    if oldest_train_task > oldest_predict_task: # oldest predict task is older than oldest train task
        return PredictTask.objects.filter(completed = False,date_created = oldest_predict_task)[0]
    else:
        return TrainTask.objects.filter(completed = False,date_created = oldest_train_task)[0]

def get_data_generator(dataset,dataset_size,batch_size):
    X = dataset['data']['X']
    y = dataset['data']['y']
    while True:
        # prepare indicies for this round
        # shuffle the indices and group by batch size. indices should be [(i1,i2,...in),(in+1,...)...]
        # https://stackoverflow.com/questions/1624883/alternative-way-to-split-a-list-into-groups-of-n
        indicies = range(dataset_size)
        shuffle(indicies)
        indicies = zip(*(iter(indicies),) * batch_size)
        # for each batch of indices (i1,i2...in)...
        for batch_indicies in indicies:
            # TODO: fix this for multiple input output models
            batch_X = [np.array([X[i][b] for b in batch_indicies]) for i in X]
            batch_y = [np.array([y[i][b] for b in batch_indicies]) for i in y]
            yield batch_X,batch_y

def perform_predict_task(task):
    try:
        weight_model = task.weight
        weight_file = weight_model.weight_file.file
        input = json.loads(task.input)
        model_def = weight_model.model.definition
        model = model_from_json(model_def)
        result = model.predict(np.array(input) if type(model.input_shape) == tuple else [np.array(inp) for inp in input])
        print 'result for evaluating task {} : {}'.format(task.name,result)
        return result
    except:
        raise
        return None

def perform_train_task(task):
    try:
        weight_model = task.weight
        weight_file = weight_model.weight_file.file
        config = TaskBroker().parse_config(task.config)

        f = h5py.File(task.dataset.dataset_file.name,'r')
        model = model_from_json(weight_model.model.definition)
        # compile the model
        apply_dict(model.compile,config['compile'])

        # unpack and eval necessary variables for training 
        dataset_size = DatasetBroker().get_dataset_size(task.dataset.name)
        batch_size = config['fit']['batch_size']
        epoch = config['fit']['epoch']
        steps = dataset_size / batch_size
        callbacks = config['fit']['callbacks'] if 'callbacks' in config['fit'] else None

        # train and return the history object
        result = model.fit_generator(get_data_generator(f,dataset_size,batch_size),steps,epoch,callbacks = callbacks)

        f.close()
        return result

    except:
        raise
        return None

def save_train_result(task,result):
    name = task.name
    result = result.__dict__
    model = result['model']
    weight_path = 'result/train/{}_weights.h5'.format(name)
    model.save_weights(weight_path)
    if 'model' in result:
        del result['model']
    result = json.dumps(result)
    result_model = TrainResult(of_task = task,result = weight_path,history = result)
    result_model.save()
    return result_model
'''
    Main loop
'''
def cron_loop():
    while True:
        # 1. get the latest job to be performed
        task = get_oldest_task()
        if not task:
            return

        # 2. load and perform the task
        if type(task) == PredictTask:
            print 'predict...'
            result = perform_predict_task(task)
            # 3. store to the result
            if result is None: break # some error occured. Await for further handling

            error = ResultBroker().create_predict_result_for_task(task,result)
            # 4. mark as completed
            if not error:
                error = TaskBroker().mark_as_completed(task.name)
                if error:
                    print error
            else: 
                print error
        else:
            print 'train...'
            result = perform_train_task(task)
            if not result:
                # what to do?
                continue
            # 3. store to the result
            error = TaskBroker().mark_as_completed(task.name)
            if error:
                TaskBroker().mark_as_completed(task.name,True)
                print 'Unable to mark the task as completed'
                continue

            result = save_train_result(task,result)
            if not result:
                print 'Cannot save Train result,reverting mark as completed action'
                TaskBroker().mark_as_completed(task.name,True)

            break
