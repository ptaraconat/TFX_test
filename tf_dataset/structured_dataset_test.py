import numpy as np 
import tensorflow as tf 

FEATURES_COLUMNS = ['input1', 'input2', 'input3']
LABEL_COLUMNS = ['output1', 'output2', 'output3']

def model_training_inputer(data_proto):
    features = [data_proto[col] for col in FEATURES_COLUMNS]
    col_2d = data_proto['input4']
    labels = [data_proto[col] for col in LABEL_COLUMNS]
    return features, labels 

N = 60000
in1 = np.linspace(0,10, N)
in2 = np.linspace(0,22, N)
in3 = np.linspace(-2,-1, N)
in4_a = np.linspace(0,1,N)
in4_b = np.linspace(0,1,N)
in4 = np.stack((in4_a, 4*in4_b), axis = 1)

out1 = in1**3 + in2 - 3*in3
out2 = in1**2 + in2**3 + in3
out3 = in2 + in3 - 4

data_dict = {'input1' : in1,
             'input2' : in2,
             'input3' : in3,
             'input4' : in4,
             'output1' : out1,
             'output2' : out2,
             'output3' : out3}

dataset = tf.data.Dataset.from_tensor_slices(data_dict)
print(dataset)
dataset = dataset.map(model_training_inputer)
print(dataset)

dataset_bis = tf.data.Dataset.from_tensor_slices([in1,in2,in3,out1,out2,out3])
print(dataset_bis)
