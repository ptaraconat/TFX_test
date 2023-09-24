import numpy as np 
import tensorflow as tf 

FEATURES_COLUMNS = ['input1', 'input2', 'input3']
LABEL_COLUMNS = ['output1', 'output2', 'output3']

def get_model(): 
    input1 = tf.keras.layers.Input(shape=(1,), name='input1')
    input2 = tf.keras.layers.Input(shape=(1,), name='input2')
    path1 = tf.keras.layers.concatenate([input1, input2])
    path1 = tf.keras.layers.Dense(5, activation = 'linear')(path1)
    path1 = tf.keras.layers.Dense(3, activation = 'linear')(path1)
    #
    input3 = tf.keras.layers.Input(shape=(3,), name='input3')
    path2 = tf.keras.layers.Dense(6, activation = 'linear')(input3)
    path2 = tf.keras.layers.Dense(3, activation = 'linear')(path2)
    # Merge both paths 
    merge = tf.keras.layers.concatenate([path1,path2])
    #output = tf.keras.layers.Dense(3, activation = 'linear')(merge)
    outputs = [tf.keras.layers.Dense(1, name = f)(merge) for f in LABEL_COLUMNS]
    model = tf.keras.Model(inputs = [input1, input2, input3], outputs = outputs)
    return model

def model_training_inputer(data_proto):
    features = {input : data_proto[input] for input in FEATURES_COLUMNS}
    labels = {output : data_proto[output] for output in LABEL_COLUMNS}
    return features, labels 

def inference_inputer(data_proto): 
    #features = [data_proto[input] for input in FEATURES_COLUMNS]
    features = {input : data_proto[input] for input in FEATURES_COLUMNS}
    return features

N = 60000
in1 = np.linspace(0,10, N)
in2 = np.linspace(0,22, N)
in3_a = np.linspace(-2,-1, N)
in3_b = np.linspace(0,1,N)
in3_c = np.linspace(0,1,N)
in3 = np.stack((in3_a, 4*in3_b, in3_c), axis = 1)

out1 = in1**3 + in2 
out2 = in1**2 + in2**3
out3 = in2  - 4 + np.dot(in3,np.array([1,1,1]))

data_dict = {'input1' : in1,
             'input2' : in2,
             'input3' : in3,
             'output1' : out1,
             'output2' : out2,
             'output3' : out3}

loaded_dataset = tf.data.Dataset.from_tensor_slices(data_dict)
dataset = loaded_dataset.map(model_training_inputer)
dataset = dataset.shuffle(N)
dataset = dataset.batch(200)

model = get_model()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),loss='mse')
model.summary()
model.fit(dataset)

test_dataset = loaded_dataset.map(inference_inputer)
test_dataset = test_dataset.batch(200)
tensor_element = tf.data.experimental.get_single_element(test_dataset.take(1))

prediction = model(tensor_element)
print(tensor_element['input3'].shape)


