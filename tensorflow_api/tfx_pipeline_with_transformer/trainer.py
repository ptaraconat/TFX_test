from typing import List
from absl import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils
import tensorflow_transform as tft

from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2

import os as os 

_INPUT_KEYS = ['x1', 'x2']
_OUTPUT_KEYS = ['y1', 'y2']

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10

# Since we're not generating or creating a schema, we will instead create
# a feature spec.  Since there are a fairly small number of features this is
# manageable for this dataset.
_FEATURE_SPEC = {
    **{
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
           for feature in _INPUT_KEYS + _OUTPUT_KEYS
       }
}


def _input_fn_former(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    schema: schema of the input data.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size),
      schema=schema).repeat()

def _input_fn(file_pattern, data_accessor, schema, batch_size) :
  print('########## File Pattern #############')
  print(file_pattern)
  list_dir = os.listdir(file_pattern[0][:-1])
  file_path = file_pattern[0][:-1]+list_dir[0]
  print(file_path)
  dataset = tf.data.TFRecordDataset(file_path,compression_type="GZIP")
  print('########## Dataset #############')
  print(dataset) 

  def _parse(example_proto):
     feature_spec = tft.tf_metadata.schema_utils.schema_as_feature_spec(schema).feature_spec
     example = tf.io.parse_example(example_proto, feature_spec)
     return example 
  dataset = dataset.map(_parse) 
  print('########## Dataset parsed #############')
  print(dataset) 

  def _make_features_label_tuples(example_proto):
    features = {feature: example_proto[feature] for feature in _INPUT_KEYS}
    labels = {label: example_proto[label] for label in _OUTPUT_KEYS}
    return features, labels

  dataset = dataset.map(_make_features_label_tuples) 
  print('########## Dataset tuple #############')
  print(dataset) 
  
  #dataset = dataset.shuffle(1000).batch(20)
  #print('########## Dataset suffle and batch #############')
  #print(dataset) 
  # Shuffle and batch the dataset
  dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  print('########## Dataset suffle and batch and prefetch #############')
  print(dataset) 
  return dataset.repeat()
     
   

def _input_fn2(file_pattern, data_accessor, schema, batch_size):
    """Create a tf.data.Dataset for training or evaluation.

    Args:
        file_pattern: A file pattern specifying the input files.
        data_accessor: A DataAccessor for reading data.
        schema: The schema of the dataset.
        batch_size: Batch size for the dataset.
        feature_columns: List of feature columns.
        label_columns: List of label columns.

    Returns:
        A tf.data.Dataset that provides (features, labels) batches.
    """
    dataset = data_accessor.tf_dataset_factory(file_pattern,tfxio.TensorFlowDatasetOptions(batch_size=batch_size),schema=schema)
  
    # Decode the TFRecords and parse them according to the schema
    def _make_features_labels_tuples(example_proto):
        #feature_spec = tft.tf_metadata.schema_utils.schema_as_feature_spec(schema).feature_spec
        #example = tf.io.parse_example(example_proto, feature_spec)
        # Extract features and labels
        features = {feature: example_proto[feature] for feature in _INPUT_KEYS}
        labels = {label: example_proto[label] for label in _OUTPUT_KEYS}
        return features, labels

    dataset = dataset.map(_make_features_labels_tuples)
    # Batch the dataset
    #dataset = dataset.batch(batch_size)

    return dataset


def _build_keras_model() -> tf.keras.Model:
  """Creates a DNN Keras model for classifying penguin data.

   Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  inputs = [keras.layers.Input(shape=(1,), name=f) for f in _INPUT_KEYS]
  d = keras.layers.concatenate(inputs)
  for _ in range(2):
    d = keras.layers.Dense(8, activation='relu')(d)
  outputs = [keras.layers.Dense(1, name = f)(d) for f in _OUTPUT_KEYS]
  #outputs = keras.layers.Dense(2)(d)

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=keras.optimizers.Adam(1e-2),
      loss='mse'
      )

  model.summary(print_fn=logging.info)
  return model


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """

  # This schema is usually either an output of SchemaGen or a manually-curated
  # version provided by pipeline author. A schema can also derived from TFT
  # graph if a Transform component is used. In the case when either is missing,
  # `schema_from_feature_spec` could be used to generate schema from very simple
  # feature_spec, but the schema returned would be very primitive.
  schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)
  
  train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
        batch_size=_TRAIN_BATCH_SIZE
    )
  eval_dataset = _input_fn(
     fn_args.eval_files,
     fn_args.data_accessor,
     schema,
     batch_size=_EVAL_BATCH_SIZE
     )
  
  print(train_dataset)

  model = _build_keras_model()
  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  # The result of the training should be saved in `fn_args.serving_model_dir`
  # directory.
  model.save(fn_args.serving_model_dir, save_format='tf')