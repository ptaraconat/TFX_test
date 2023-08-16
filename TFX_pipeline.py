import os
import tempfile
import urllib

import tensorflow as tf
import tensorflow_model_analysis as tfma

from tfx import v1 as tfx
#load_ext tfx.orchestration.experimental.interactive.notebook_extensions.skip

print(tfx.__version__)
print(tf.__version__)

# Chemin vers le répertoire de données
_data_root = tempfile.mkdtemp(prefix='tfx-data')
DATA_PATH = 'https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/chicago_taxi_pipeline/data/simple/data.csv'
_data_filepath = os.path.join(_data_root, "data.csv")
urllib.request.urlretrieve(DATA_PATH, _data_filepath)



example_gen = tfx.components.CsvExampleGen(input_base=_data_root)


# Créez une instance du Pipeline
pipeline = tfx.dsl.Pipeline(
    pipeline_name="my_tfx_pipeline",
    # ... autres spécifications de pipeline
    components=[example_gen]
)

# Exécutez le pipeline en utilisant LocalDagRunner
local_dag_runner = tfx.orchestration.LocalDagRunner()
local_dag_runner.run(pipeline)

for artifact in example_gen.outputs['examples'].get():
  print(artifact.split, artifact.uri)