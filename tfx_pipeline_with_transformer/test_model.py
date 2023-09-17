from sklearn.feature_selection import r_regression
import os as os 
import numpy as np 
import tensorflow as tf 

PIPELINE_NAME = "regression"
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)

# Replace 'exported_model_directory' with the actual path to your exported model.
exported_model_directory = SERVING_MODEL_DIR+'/1694439698'

# Load the model
loaded_model = tf.saved_model.load(exported_model_directory)

# Now you can use the 'loaded_model' for inference or further training.
print(loaded_model.signatures)

# Get data 
N = 20
x1 = np.linspace(0,1,N)
x2 = np.linspace(0.,2,N)
y1 = x1**2. + 4*x2
y2 = x1*x2
# Set input data for inference 
x1_tensor = tf.constant(x1, dtype=tf.float32, name = 'x1') # For newer versions of
x2_tensor = tf.constant(x2, dtype=tf.float32, name = 'x2') # For newer versions of
new_shape = (N,1)
x1_tensor = tf.reshape(x1_tensor, new_shape)
x2_tensor = tf.reshape(x2_tensor, new_shape)
# Perform inference 
output = loaded_model([x1_tensor, x2_tensor])
y1_hat = np.squeeze(output[0].numpy())
y2_hat = np.squeeze(output[1].numpy())
# Assess and print performance 
print(y1)
print(y1_hat)
corr_y1 = r_regression(y1.reshape(-1,1), y1_hat.reshape(-1,1))
print(corr_y1)
#
print(y2)
print(y2_hat)
corr_y2 = r_regression(y2.reshape(-1,1), y2_hat.reshape(-1,1))
print(corr_y2)


# test model on transform output 
data_path = '/workspaces/TFX_test/tfx_pipeline_with_transformer/pipelines/regression/Transform/transformed_examples/4/Split-train/transformed_examples-00000-of-00001.gz'
data_path = '/workspaces/TFX_test/tfx_pipeline_with_transformer/pipelines/regression/CsvExampleGen/examples/1/Split-train/data_tfrecord-00000-of-00001.gz'
dataset = tf.data.TFRecordDataset(data_path,compression_type="GZIP")
# Dataset parser function 
def _parse_proto(example_proto): 
	# Create features description 
    feature_description = {'x1' : tf.io.FixedLenFeature([], dtype = tf.float32),
                           'x2' : tf.io.FixedLenFeature([], dtype = tf.float32),
                           'y1' : tf.io.FixedLenFeature([], dtype = tf.float32),
                           'y2' : tf.io.FixedLenFeature([], dtype = tf.float32), }
	# Parse the input 'tf.Example' proto using the dictionnary 
    return tf.io.parse_single_example(example_proto,feature_description)
dataset = dataset.map(_parse_proto)
_INPUT_KEYS = ['x1', 'x2']
_OUTPUT_KEYS = ['y1', 'y2']
def _make_features_label_tuples(example_proto):
    features = {feature: example_proto[feature] for feature in _INPUT_KEYS}
    labels = {label: example_proto[label] for label in _OUTPUT_KEYS}
    return features
dataset = dataset.map(_make_features_label_tuples)
def _get_column(example_proto, col_name):
    return example_proto[col_name]
x1_data = dataset.map(lambda x: _get_column(x, 'x1'))
x2_data = dataset.map(lambda x: _get_column(x, 'x2'))
x1_arr = np.array(list(x1_data))
x2_arr = np.array(list(x2_data))
# Set input data for inference 
x1_tensor = tf.constant(x1_arr, dtype=tf.float32, name = 'x1') # For newer versions of
x2_tensor = tf.constant(x2_arr, dtype=tf.float32, name = 'x2') # For newer versions of
new_shape = (len(x1_arr),1)
x1_tensor = tf.reshape(x1_tensor, new_shape)
x2_tensor = tf.reshape(x2_tensor, new_shape)
# Perform inference 
output = loaded_model([x1_tensor, x2_tensor])
# Assess and print performance 
print(y1)
print(y1_hat)
corr_y1 = r_regression(y1.reshape(-1,1), y1_hat.reshape(-1,1))
print(corr_y1)
#
print(y2)
print(y2_hat)
corr_y2 = r_regression(y2.reshape(-1,1), y2_hat.reshape(-1,1))
print(corr_y2)

#output = loaded_model(dataset)