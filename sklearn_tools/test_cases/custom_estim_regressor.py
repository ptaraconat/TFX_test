from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer
import numpy as np
import matplotlib.pyplot as plt
import sys as sys 
sys.path.append('../')
from custom_estimators import * 

def create_pipeline():
    '''
    '''
    optimizer = tf.keras.optimizers.Adam()
    loss = 'mse'
    hidden_layers=[tf.keras.layers.Dense(units = 5, activation = 'relu'),
                   tf.keras.layers.Dense(units = 5, activation = 'relu')]
    tfestimator = TFEstimatorRegressor(optimizer,
                                       loss,
                                       hidden_layers= hidden_layers,
                                       epochs= 1000)
    scaler = StandardScaler()
    # pipeline = make_pipeline([scaler, tfestimator]) # Doesn't work with make_pipeline, don't know why ......
    pipeline = Pipeline([('standard_scaler', scaler),
                         ('TF_NN', tfestimator)])
    return pipeline

def main():
    # Define some data
    X = np.linspace(0,4,100)
    X = np.expand_dims(X, axis = -1)
    y = X**2.
    # Define model
    pipeline = create_pipeline()
    # fit model
    pipeline.fit(X,y)
    # test and display model results
    yhat = pipeline.predict(X)
    plt.plot(y,yhat,'ro')
    plt.plot([np.min(y),np.max(y)], [np.min(y),np.max(y)], 'k')
    plt.show()
    # Cross validation of model performances 
    cross_val_approach = ShuffleSplit(n_splits = 10, random_state= 0)
    scoring = {'r2' : make_scorer(r2_score),
                'mse' : make_scorer(mean_squared_error)} 
    res_dict = cross_validation(create_pipeline, X, y, cross_val_approach, scoring)
    print(res_dict)
    print('train mean r2 ::: ', np.mean(res_dict['train_r2']))
    print('train std r2 ::: ', np.std(res_dict['train_r2']))
    print('test mean r2 ::: ', np.mean(res_dict['test_r2']))
    print('test std r2 ::: ', np.std(res_dict['test_r2']))

if __name__ == '__main__':
    main()