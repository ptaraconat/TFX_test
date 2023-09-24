import tensorflow as tf 
from tensorflow.keras import layers

def normalize(data_proto):
    img = data_proto['image']
    print(img.shape)
    label = data_proto['label']
    img = tf.cast(img, tf.float32)/255.
    return {'image' : img, 'label' : tf.one_hot(label,10)} #img, tf.one_hot(label,10)

def model_inference_inputer(data_proto):
    img = data_proto['image']
    return img

def model_training_inputer(data_proto):
    img = model_inference_inputer(data_proto)
    label = data_proto['label']
    return (img,label)

#### Dataset that is define from dict 
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
dict = {'image' : X_train,
        'label' : y_train}
dataset = tf.data.Dataset.from_tensor_slices(dict)
dataset = dataset.map(normalize)
print(dataset)
for element in dataset.take(3) : 
    print(element['image'].numpy().shape, element['label'].numpy())
#### Map dataset so that we get a tuple Features/Labels
train_dataset = dataset.map(model_training_inputer)
train_dataset = train_dataset.shuffle(10000).batch(200)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
print(train_dataset)

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape = (28,28,)),
                             tf.keras.layers.Dense(128,activation = 'relu'),
                             tf.keras.layers.Dense(10)])

model.compile(optimizer = 'adam',
              loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True))

model.summary()
model.fit(train_dataset,epochs = 1)

#### Perform inference 
test_dataset = dataset.map(model_inference_inputer).batch(200)
print(test_dataset)
tensor_element = tf.data.experimental.get_single_element(test_dataset.take(1))

print(tensor_element)
print(model(tensor_element))