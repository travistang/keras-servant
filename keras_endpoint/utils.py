from keras.layers import *
from keras.models import *
from rest_framework.response import Response
from rest_framework import status
import os
import errno
def empty_json_response():
    return Response({},status = status.HTTP_400_BAD_REQUEST)
def error_json_response_with_details(details):
    return Response({'detail': details},status = status.HTTP_404_NOT_FOUND)

def error_json_response(details):
    return Response(details,status = status.HTTP_400_BAD_REQUEST)

def create_success_with_data(data):
    return Response(data,status = status.HTTP_201_CREATED)

def is_json_valid_keras_model(json_str):
    # check param type
    if not type(json_str) in [unicode,str]: return False
    # allow empty string
    if len(json_str.strip()) == 0: return True
    try:
        model_from_json(json_str)
        return True
    except Exception as e:
        return False

def create_model_from_json(json_str):
    if not is_json_valid_keras_model(json_str):
        return None
    return model_from_json(json_str)
def create_dir_if_not_exists(pathname):
    if not os.path.exists(os.path.dirname(pathname)):
        try:
            os.makedirs(os.path.dirname(pathname))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
# ref: https://stackoverflow.com/questions/3812849/how-to-check-whether-a-directory-is-a-sub-directory-of-another-directory
def is_subdir(path, directory):
        path = os.path.realpath(path)
        directory = os.path.realpath(directory)
        relative = os.path.relpath(path, directory)
        return not relative.startswith(os.pardir + os.sep)

def copy_file(f_from,f_to):
    try:
        f_to.writelines([l for l in f_from.readlines()])
        return True
    except:
        # some exceptions can be raised from the code above. Such as the mode of the file objects are too restrictive,out of spaces...
        return False

def serve_file(f,filename,mime_type):
    from django.http import HttpResponse
    response = HttpResponse(f,content_type = mime_type)
    response['Content-Disposition'] = 'attachment;filename="{}"'.format(filename)
    return response

# returns True if dict contains all keys in "keys"
def contains_key(dict, keys):
    return all(x in dict for x in keys)

# returns True if dict contains one of the keys in "keys"
def contains_one_of(dict,keys):
    return any(x in dict for x in keys)

# apply all parameters inside a dict object to function f
def apply_dict(f,dict):
    return f(**dict)

def is_model_data_shape_consistent(m_in,m_out,d_in,d_out):
    # preprocessing: remove batch size / num data
    merge_shape = lambda shapes: tuple(filter(lambda d: d is not None,np.array(shapes).flatten().tolist()))

    # apply to the model's IO shape
    m_in = m_in[1:] if type(m_in) is tuple else merge_shape(m_in)
    m_out = m_out[1:] if type(m_out) is tuple else merge_shape(m_out)

    # retrieve shapes for each records in the dataset
    d_in = d_in[1:]
    d_out = d_out[1:]

    print m_in,m_out,d_in,d_out
    # apply checking and return result
    return m_in == m_out and d_in == d_out
