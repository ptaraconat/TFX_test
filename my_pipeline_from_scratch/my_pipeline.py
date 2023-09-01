import tfx.v1 as tfx 
from tfx.types import artifact_utils
from tfx.types import standard_artifacts

import os as os 

def create_pipeline(pipeline_name, 
                    pipeline_root,
                    data_path,
                    enable_cache,
                    metadata_connection_config = None,
                    beam_pipeline_args = None) :
    components = []

    input = tfx.proto.Input(splits=[tfx.proto.Input.Split(name='train',pattern='train/*'),
                                    tfx.proto.Input.Split(name='eval',pattern='eval/*')
                                    ])
    example_gen = tfx.components.CsvExampleGen(input_base = data_path, input_config = input)
    components.append(example_gen)

    statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'],
                                                  exclude_splits=['eval'])
    components.append(statistics_gen)

    schema_gen = tfx.components.SchemaGen(statistics=statistics_gen.outputs["statistics"],
                                          infer_feature_shape=False,
                                          exclude_splits=['eval'])
    components.append(schema_gen)

    transform = tfx.components.Transform(examples=example_gen.outputs["examples"],
                                         schema=schema_gen.outputs["schema"],
                                         module_file="module.py")
    components.append(transform)

    #transformer = tfx.components.Transform(examples=example_gen.outputs['examples'],
    #                                       schema=schema_gen.outputs['schema'],
    #                                       module_file=os.path.abspath(_taxi_transform_module_file))

    pipeline = tfx.dsl.Pipeline(pipeline_name = pipeline_name,
                                pipeline_root = pipeline_root,
                                components = components,
                                enable_cache = enable_cache,
                                metadata_connection_config = metadata_connection_config,
                                beam_pipeline_args = beam_pipeline_args)
    return pipeline

PIPELINE_NAME = 'pipeline_test'
PIPELINE_ROOT = 'pipeline_outut'
DATA_PATH = 'datasets'
ENABLE_CACHE = True
METADATA_PATH = os.path.join('.', 'tfx_metadata', PIPELINE_NAME, 'metadata.db')

print(METADATA_PATH)

pipeline = create_pipeline(PIPELINE_NAME,
                           PIPELINE_ROOT,
                           DATA_PATH,
                           ENABLE_CACHE,
                           metadata_connection_config = tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH))

tfx.orchestration.LocalDagRunner().run(pipeline)

print(pipeline.components[0].outputs)

# Assuming you have a CsvExampleGen instance named csv_example_gen
#schema_artifact = csv_example_gen.outputs['output'].get()[0]

# Load the schema
#schema = artifact_utils.get_single_instance(schema_artifact).mlmd_artifact.schema