from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_data_validation as tfdv
print('TF version:', tf.__version__)
print('TFDV version:', tfdv.version.__version__)
import numpy as np 
import pandas as pd 
import os as os 


# Generate some data 
N = 10
x1 = np.random.normal(0,2,N)
x2 = 4*np.random.normal(1,2,N)
x3 = np.random.random_integers(2, size = N)
y = np.random.random_integers(4, size = N)
df = pd.DataFrame()
df['x1'] = x1
df['x2'] = x2
df['x3'] = x3
df['y'] = y 
# 
df_train, df_test = train_test_split(df,test_size= 0.4, random_state= 10)
df_train.to_csv('data_train.csv',sep = ',', decimal = '.', index = False)
df_test.to_csv('data_test.csv',sep = ',', decimal = '.', index = False )

# Get statistics 
train_stats = tfdv.generate_statistics_from_csv(data_location='data_train.csv')
#print(train_stats)
#tfdv.visualize_statistics(train_stats)
# Get schema from the data 
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema=schema)
tfdv.write_schema_text(schema,'schema_test.pbtxt')
del schema 


# 
eval_stats = tfdv.generate_statistics_from_csv(data_location = 'data_test.csv')
schema = tfdv.load_schema_text('schema_test.pbtxt')
anomalies = tfdv.validate_statistics(statistics = eval_stats, schema = schema)
tfdv.display_anomalies(anomalies)
print(anomalies)

