import tfx.v1 as tfx 

def create_pipeline(pipeline_name, 
                    pipeline_root, 
                    enable_cache, 
                    metadata_connection_config, 
                    beam_pipeline_args) : 
    components = []
    pipeline = tfx.dsl.Pipeline(pipeline_name = pipeline_name,
                                pipeline_root = pipeline_root,
                                components = components,
                                enable_cache = enable_cache,
                                metadata_connection_config = metadata_connection_config,
                                beam_pipeline_args = beam_pipeline_args)
    return pipeline