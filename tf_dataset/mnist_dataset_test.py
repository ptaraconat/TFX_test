import tensorflow as tf 

def normalize(img, label):
    img = tf.cast(img, tf.float32)/255.
    return img, tf.one_hot(label,10)

def get_img_only(img, label):
    return img

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print(type(X_train))
print(X_train.shape)
print(type(y_train))
print(y_train.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
print(train_dataset)
train_dataset = train_dataset.map(normalize)
print(train_dataset)
train_dataset = train_dataset.shuffle(10000).batch(200)
print(train_dataset)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE) # ???? 
#for (img, label) in train_dataset : 
#	print(img.numpy().shape, label.numpy().shape)
print(train_dataset)

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape = (28,28,)),
                             tf.keras.layers.Dense(128,activation = 'relu'),
                             tf.keras.layers.Dense(10)])

model.compile(optimizer = 'adam',
              loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True))

model.summary()
model.fit(train_dataset,epochs = 1)

# Make some predictions 
test_dataset = tf.data.Dataset.from_tensor_slices((X_test,y_test))
test_dataset = test_dataset.map(normalize)
test_dataset = test_dataset.map(get_img_only)
test_dataset = test_dataset.batch(200)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
#dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# Cannot pas a data set for prediction must get a tensor instead, this is done as follows. 
tensor_element = tf.data.experimental.get_single_element(test_dataset.take(1))

print(tensor_element)
print(model(tensor_element))
