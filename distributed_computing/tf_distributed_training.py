import tensorflow as tf 
import multiprocessing
import numpy as np 
import time 

BATCH_SIZE_PER_REPLICA = 200
N_EPOCHS = 20 
N = 200000

def features_label_split(element_proto):
    features = {feature : element_proto[feature] for feature in ['x1', 'x2']}
    labels = {label : element_proto[label] for label in ['y1', 'y2']}
    return features, labels

def get_model(): 
    inputs = [tf.keras.layers.Input(shape=(1,), name=f) for f in ['x1', 'x2']]
    d = tf.keras.layers.concatenate(inputs)
    for _ in range(8):
        d = tf.keras.layers.Dense(8, activation='relu')(d)
        outputs = [tf.keras.layers.Dense(1, name = f)(d) for f in ['y1', 'y2']]
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),loss='mse')
    model.summary()
    return model

def get_cpu_names():
    return multiprocessing.cpu_count(), [f"/cpu:{i}" for i in range(multiprocessing.cpu_count())]

cpu_count, cpu_names = get_cpu_names()
print(f"Number of CPUs: {cpu_count}")
print(f"CPU Names: {cpu_names}")

# Define Data 
x1 = np.linspace(0,1,N)
x2 = np.linspace(0,2,N)
y1 = x1**2. + 4*x2
y2 = x1*x2
data_dict = {'x1' : x1,
             'x2' : x2,
             'y1' : y1,
             'y2' : y2}
dataset = tf.data.Dataset.from_tensor_slices(data_dict)
dataset= dataset.map(features_label_split)
dataset = dataset.shuffle(N)
##########################

# Model training without distributing 
model = get_model()
start1 = time.time()
model.fit(dataset.batch(BATCH_SIZE_PER_REPLICA), epochs = N_EPOCHS, verbose = 1)
end1 = time.time()

#######################
strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0", "/cpu:1"])
#strategy = strategy = tf.distribute.MultiWorkerMirroredStrategy(devices=["/cpu:0", "/cpu:1"])
# Compute a global batch size using a number of replicas.
global_batch_size = (BATCH_SIZE_PER_REPLICA *
                     strategy.num_replicas_in_sync)
with strategy.scope(): 
    model1 = get_model()
    start2 = time.time()
    model1.fit(dataset.batch(global_batch_size), epochs = N_EPOCHS,verbose = 1)
    end2 = time.time()

print('fit1 ', end1 - start1)
print('fit2 ', end2 - start2)
print(global_batch_size)