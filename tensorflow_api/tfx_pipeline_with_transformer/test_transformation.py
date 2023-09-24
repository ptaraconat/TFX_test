import tensorflow as tf 
import tensorflow_transform as tft 
import numpy as np 
import os as os 
#from transformer import * 

# Define transformation function (the same as in transformer.py)
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

# Dataset parser function 
def _parse_proto(example_proto): 
	# Create features description 
    feature_description = {'x1' : tf.io.FixedLenFeature([], dtype = tf.float32),
                           'x2' : tf.io.FixedLenFeature([], dtype = tf.float32),
                           'y1' : tf.io.FixedLenFeature([], dtype = tf.float32),
                           'y2' : tf.io.FixedLenFeature([], dtype = tf.float32), }
	# Parse the input 'tf.Example' proto using the dictionnary 
    return tf.io.parse_single_example(example_proto,feature_description)

# Extract column function 
def _get_column(example_proto, col_name):
    return example_proto[col_name]

data_path = '/workspaces/TFX_test/tfx_pipeline_with_transformer/pipelines/regression/CsvExampleGen/examples/1/Split-train/data_tfrecord-00000-of-00001.gz'
raw_data = tf.data.TFRecordDataset(data_path,compression_type="GZIP")
raw_data = raw_data.map(_parse_proto)
#
data_path = '/workspaces/TFX_test/tfx_pipeline_with_transformer/pipelines/regression/Transform/transformed_examples/4/Split-train/transformed_examples-00000-of-00001.gz'
transformed_raw_data = tf.data.TFRecordDataset(data_path, compression_type = 'GZIP')
transformed_raw_data = transformed_raw_data.map(_parse_proto)
#

raw_x1 = raw_data.map(lambda x: _get_column(x, 'x1'))
raw_x2 = raw_data.map(lambda x: _get_column(x, 'x2'))
raw_y1 = raw_data.map(lambda x: _get_column(x, 'y1'))
raw_y2 = raw_data.map(lambda x: _get_column(x, 'y2'))

trans_x1 = transformed_raw_data.map(lambda x: _get_column(x, 'x1'))
trans_x2 = transformed_raw_data.map(lambda x: _get_column(x, 'x2'))
trans_y1 = transformed_raw_data.map(lambda x: _get_column(x, 'y1'))
trans_y2 = transformed_raw_data.map(lambda x: _get_column(x, 'y2'))

print(np.all(np.array(list(raw_x1)) == np.array(list(trans_x1))))
print(np.all(np.array(list(raw_x2)) == np.array(list(trans_x2))))

print(np.all(np.array(list(raw_y1)) == np.array(list(trans_y1))))
print(np.all(np.array(list(raw_y2)) == np.array(list(trans_y2))))

# Perform data transformation manually 
# Load the exported transform function
transform_fn_dir = '/workspaces/TFX_test/tfx_pipeline_with_transformer/pipelines/regression/Transform/transform_graph/4'  # Replace with the actual path
transform_fn = tft.TFTransformOutput(transform_fn_dir)
# Apply the loaded preprocessing_fn to the new dataset
def apply_transform(element):
    return transform_fn.transform_raw_features(element)

print(raw_data)
man_trans_data = raw_data.map(apply_transform)
man_x1 = man_trans_data.map(lambda x: _get_column(x, 'x1'))
man_x2 = man_trans_data.map(lambda x: _get_column(x, 'x2'))

for element in man_trans_data:
    print(element)
print(np.array(list(raw_x1)[:10]))
print(np.array(list(trans_x1)[:10]))
print(np.array(list(man_x1)[:10]))

#print(np.all(np.array(list(man_x1)) == np.array(list(trans_x1))))
#print(np.all(np.array(list(man_x2)) == np.array(list(trans_x2))))

print('terminated')