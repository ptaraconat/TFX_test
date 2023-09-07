import tfx.v1 as tfx 
import tensorflow as tf
print(tfx.__version__)
print(tf.__version__)
import os
from absl import logging
import tempfile
import pandas as pd 
import numpy as np 

def _create_pipeline(pipeline_name: str, 
                     pipeline_root: str, 
                     data_root: str,
                     module_file: str, 
                     serving_model_dir: str,
                     metadata_path: str) -> tfx.dsl.Pipeline:
  """Creates a three component penguin pipeline with TFX."""
  # Brings data into the pipeline.
  example_gen = tfx.components.CsvExampleGen(input_base=data_root)

  # Uses user-provided Python function that trains a model.
  trainer = tfx.components.Trainer(module_file=module_file,
                                   examples=example_gen.outputs['examples'],
                                   train_args=tfx.proto.TrainArgs(num_steps=100),
                                   eval_args=tfx.proto.EvalArgs(num_steps=5),
                                   )

  # Pushes the model to a filesystem destination.
  pusher = tfx.components.Pusher(model=trainer.outputs['model'],
                                 push_destination=tfx.proto.PushDestination(filesystem=tfx.proto.PushDestination.Filesystem(base_directory=serving_model_dir)))

  # Following three components will be included in the pipeline.
  components = [example_gen,
                trainer,
                pusher]

  return tfx.dsl.Pipeline(pipeline_name=pipeline_name,
                          pipeline_root=pipeline_root,
                          metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path),
                          components=components)

PIPELINE_NAME = "regression"
# Output directory to store artifacts generated from the pipeline.
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)
# Path to a SQLite DB file to use as an MLMD storage.
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')
# Output directory where created models from the pipeline will be exported.
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)
logging.set_verbosity(logging.INFO)  # Set default logging level.
_trainer_module_file = 'trainer.py'
DATA_ROOT = tempfile.mkdtemp(prefix='tfx-data')  # Create a temporary directory.

_data_filepath = os.path.join(DATA_ROOT, "data.csv")

x1 = np.linspace(0,1,1000)
x2 = np.linspace(0.,2,1000)
y1 = x1**2. + 4*x2
y2 = x1*x2
df = pd.DataFrame()
df['x1'] = x1
df['x2'] = x2
df['y1'] = y1
df['y2'] = y2
df.to_csv(_data_filepath,sep = ',', decimal = '.')
#have a look on downloaded data 
df_tmp = pd.read_csv(_data_filepath, sep = ',', decimal = '.')
print(df_tmp)

pipeline = _create_pipeline(pipeline_name=PIPELINE_NAME,
                            pipeline_root=PIPELINE_ROOT,
                            data_root=DATA_ROOT,
                            module_file=_trainer_module_file,
                            serving_model_dir=SERVING_MODEL_DIR,
                            metadata_path=METADATA_PATH)
tfx.orchestration.LocalDagRunner().run(pipeline)