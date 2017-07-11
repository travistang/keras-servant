from keras.models import *
from keras.layers import *

'''
    Verify that a json string can be used to create a Keras model
'''
def verify_model(model_json):
    try:
        model_from_json(model_json)
        return True
    except:
        return False
'''
    Verify that the weight can be used with the given Keras model
'''
def verify_weight(model,weight_file_str):
    try:
        model = model_from_json(model.definition)
        model.load_weights(weight_file_str)
        return True
    except Exception as e:
        return False

