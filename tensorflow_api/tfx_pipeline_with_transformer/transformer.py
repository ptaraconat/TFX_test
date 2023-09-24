import tensorflow_transform as tft
#from trainer import * ### In order to get global variables 

_INPUT_KEYS = ['x1', 'x2']
_OUTPUT_KEYS = ['y1', 'y2']

def preprocessing_fn(inputs):
    '''
    inputs: map from feature keys to raw not-yet-transformed features.
    Returns : 
    Map from string feature key to transformed feature.
    '''
    outputs = {}
    for key in _INPUT_KEYS : 
       outputs[key] = tft.scale_to_z_score(inputs[key])
    for key in _OUTPUT_KEYS : 
       outputs[key] = inputs[key]
    return outputs 

