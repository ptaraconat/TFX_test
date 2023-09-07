import tensorflow as tf 

data_path = '/workspaces/TFX_test/tfx_tuto_simple_pipeline_changed_inputdim/pipelines/regression/CsvExampleGen/examples/1/Split-train/data_tfrecord-00000-of-00001.gz'

raw_data = tf.data.TFRecordDataset(data_path,compression_type="GZIP")
# Print first example 
for example in raw_data.take(1):
    print(repr(example))
# 
# Create features description 
feature_description = {'x1' : tf.io.FixedLenFeature([], dtype = tf.float32),
					   'x2' : tf.io.FixedLenFeature([], dtype = tf.float32),
					   'y1' : tf.io.FixedLenFeature([], dtype = tf.float32),
					   'y2' : tf.io.FixedLenFeature([], dtype = tf.float32), }

def _parse_proto(example_proto): 
	# Parse the input 'tf.Example' proto using the dictionnary 
	return tf.io.parse_single_example(example_proto,feature_description)

# Map method parsed raw dataset into a parsed dataset 
parsed_dataset = raw_data.map(_parse_proto)

# Read and print data from dataset 
for parsed_record in parsed_dataset.take(1):
	print((parsed_record))
	print(parsed_record['x1'])
	
# 
def set_features_labels_tuple(elem):
	features = {'x1' : elem['x1'],
			 'x2' : elem['x2']}
	labels = {'y1' : elem['y1'],
			 'y2' : elem['y2']}
	return features, labels

dataset = parsed_dataset.map(set_features_labels_tuple) 
dataset = dataset.shuffle(1000).batch(20)
# Read and print data from dataset 
for record in dataset.take(1):
	print((record))
	print(record[0]["x1"])

def mlp_model():
	inputs = [tf.keras.layers.Input(shape=(1,), name=f) for f in ['x1', 'x2']]
	d = tf.keras.layers.concatenate(inputs)
	d = tf.keras.layers.Dense(10, activation = 'relu')(d)
	output_1 = tf.keras.layers.Dense(1, activation = 'linear', name = 'y1')(d)
	output_2 = tf.keras.layers.Dense(1, activation = 'linear', name = 'y2')(d)
	model = tf.keras.Model(inputs = inputs, outputs = [output_1, output_2])
	model.compile(optimizer = 'adam', loss = 'mse')
	model.summary()
	return model

model = mlp_model()

model.fit(dataset, epochs = 20 )

import numpy as np 
print(model({'x1': np.array([0,1]),
			 'x2': np.array([0,2])}))