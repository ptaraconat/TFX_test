import tensorflow as tf
import tensorflow_transform as tft

# Define your preprocessing function
def preprocessing_fn(inputs):
    """Preprocess input features.

    Args:
        inputs: A dictionary of input feature tensors.

    Returns:
        A dictionary of transformed feature tensors.
    """
    # Select columns you want to center and scale
    columns_to_center_scale = ['culmen_length_mm',
                               'culmen_depth_mm',
                               'flipper_length_mm',
                               'body_mass_g']

    # Center and scale the selected columns
    centered_scaled_features = {}
    for column_name in columns_to_center_scale:
        input_column = inputs[column_name]
        print(input_column,tf.as_string(tf.shape(input_column)))
        # Convert SparseTensor to dense tensor
        input_column = tf.sparse.to_dense(input_column, default_value=0.0)
        mean = tft.mean(input_column)
        # Calculate standard deviation manually
        variance = tft.var(input_column)
        stddev = tf.sqrt(variance)
        #print(mean,stddev)
        centered_scaled_features[column_name] = (input_column - mean) / stddev

    # Combine the transformed features with the unchanged features
    # You can include other features as needed in this dictionary
    transformed_features = {
        ** inputs
        #**centered_scaled_features,
        #'species' : inputs['species']
    }

    return transformed_features