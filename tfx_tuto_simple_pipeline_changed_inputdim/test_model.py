from sklearn.feature_selection import r_regression
import os as os 
import numpy as np 
import tensorflow as tf 

PIPELINE_NAME = "regression"
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)

# Replace 'exported_model_directory' with the actual path to your exported model.
exported_model_directory = SERVING_MODEL_DIR+'/1694353309'

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

x1_tensor = tf.constant(x1, dtype=tf.float32, name = 'x1') # For newer versions of
x2_tensor = tf.constant(x2, dtype=tf.float32, name = 'x2') # For newer versions of
new_shape = (N,1)
x1_tensor = tf.reshape(x1_tensor, new_shape)
x2_tensor = tf.reshape(x2_tensor, new_shape)

output = loaded_model([x1_tensor, x2_tensor])
print(output)
y1_hat = np.squeeze(output[0].numpy())
y2_hat = np.squeeze(output[1].numpy())

print(y1)
print(y1_hat)
corr_y1 = r_regression(y1.reshape(-1,1), y1_hat.reshape(-1,1))
print(corr_y1)

print(y2)
print(y2_hat)
corr_y2 = r_regression(y2.reshape(-1,1), y2_hat.reshape(-1,1))
print(corr_y2)
