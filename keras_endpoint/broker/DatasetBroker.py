from ..models import Dataset
from ..serializer import DatasetSerializer
from ..utils import *
import os
class DatasetBroker(object):
    ERROR_DATASET_DOES_NOT_EXIST = 1
    ERROR_DATASET_ALREADY_EXISTS = 2
    ERROR_INVALID_DATASET_NAME = 3
    ERROR_INVALID_DATASET = 4
    ERROR_UNABLE_TO_CREATE_DATASET = 5
    ERROR_SAVING_DATASET = 6
    def get_all_dataset_name(self):
        serializer = DatasetSerializer(Dataset.objects.all(),many= True)
        return serializer.data

    '''
        Return the file of the dataset
    '''
    def get_dataset(self,name):
        dataset = Dataset.objects.filter(name = name)
        if not dataset.exists():
            return (None,DatasetBroker.ERROR_DATASET_DOES_NOT_EXIST)
        else:
            return (dataset[0].dataset_file.file,None)
    '''
        Determine if the given file complies with the format of datasets
        that will be passed to Keras models

        The file should be in HDF5 format. There is exactly one subgroup called 'data' in it and there are exactly two datasets inside: 'X' and 'y'.

    '''
    def is_valid_dataset(self,f):
        import h5py
        print 'validating file with path {}'.format(f)
        try:
            f = h5py.File(f,'r')
        except Exception as e:
            print e
            return False
        # It is a h5 file. Now check the keys
        if 'data/X' not in f or 'data/y' not in f:
            print 'something not in f'
            return False
        # The dataset structure is as expected. Checking if there are more layers
        if type(f['data/X']) != type(f['data/y']) or type(f['data/y']) != h5py._hl.dataset.Dataset:
            print 'type mismatch'
            return False

        # Then check the dataset length
        x_shape = f['data/X'].shape
        y_shape = f['data/y'].shape
        if x_shape[0] != y_shape[0]:
            return False

        # everything is fine. Good to go
        return True

    def add(self,f,file_name):
        dataset_path = 'datasets/{}.h5'.format(file_name)
        # create directory if not exist
        create_dir_if_not_exists(dataset_path)

        if os.path.isfile(dataset_path) or self.get_dataset(file_name)[0] is not None:
            return DatasetBroker.ERROR_DATASET_ALREADY_EXISTS

        if not is_subdir('datasets/',dataset_path):
            return DatasetBroker.ERROR_INVALID_DATASET_NAME

        # try copying the temporary file to the right place
        try:
            with open(dataset_path,'w+') as ft:
                copy_file(f,ft)
        except Exception as e:
            print e
            return DatasetBroker.ERROR_UNABLE_TO_CREATE_DATASET

#        # and delete the temp file
#        os.remove(f)

        # then check the file itself
        if not self.is_valid_dataset(dataset_path):
            return DatasetBroker.ERROR_INVALID_DATASET

        # check if the new file is created correctly
        if not os.path.isfile(dataset_path):
            return DatasetBroker.ERROR_UNABLE_TO_CREATE_DATASET

        # create and configure the new Dataset model
        try:
            dataset_model = Dataset(name = file_name)
            dataset_model.dataset_file.name = dataset_path

            dataset_model.save()
        except:
            # the model cannot be created for whatever reason
            # remove the saved dataset file
            os.remove(dataset_path)
            return DatasetBroker.ERROR_SAVING_DATASET
