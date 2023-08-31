from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, ShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

class TFModel(tf.keras.Model):
    '''
    TensorFlow custom model 
    '''
    def __init__(self, 
                 hidden_layers = [tf.keras.layers.Dense(units = 10, activation = 'relu'),
                                  tf.keras.layers.Dense(units = 10, activation = 'relu')],
                                  output_activation = 'linear'):
        super(TFModel, self).__init__()
        self.hidden_layers = hidden_layers
        self.output_activation = output_activation

    def call(self,input):
        x = input 
        for layer in self.hidden_layers : 
            x = layer(x)
        x = self.output_layer(x)
        return x 
    
    def set_output_dim(self, output_dim): 
        '''
        Define the output layer given the output dimension. 
        Will work only for flat output. 
        Should ba called before building model 
        '''
        self.output_layer = tf.keras.layers.Dense(units = output_dim, 
                                                  activation = self.output_activation)

class TFModelRegressor(TFModel): 

    def __init__(self,
                 hidden_layers = [tf.keras.layers.Dense(units = 10, activation = 'relu'),
                                  tf.keras.layers.Dense(units = 10, activation = 'relu')]): 
        super(TFModelRegressor, self).__init__(hidden_layers= hidden_layers)
        self.output_activation = 'linear'
        self.set_output_dim(1)

class TFEstimator(BaseEstimator):
    ''' TensorFlow interfacer
    '''
    def __init__(self,model,optimizer,loss,batch_size = 200, epochs = 100):

        self.tf_model = model 
        self.optimizer = optimizer 
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y) : 
        self.tf_model.set_output_dim(y.shape[-1])
        self.tf_model.build(X.shape)
        self.tf_model.compile(optimizer = self.optimizer,loss = self.loss)
        self.tf_model.fit(X, y, epochs = self.epochs, batch_size = self.batch_size)
        return self 
    
    def predict(self, X): 
        prediction = self.tf_model(X)
        return prediction.numpy()
    
    def score(self, X, y):
        return 0 
    
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"tf_model": self.tf_model, 
                "optimizer": self.optimizer,
                "loss": self.loss,
                "batch_size": self.batch_size,
                "epochs": self.epochs}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class TFEstimatorRegressor(TFEstimator): 

    def __init__(self,model,optimizer,loss,batch_size = 200, epochs = 100):
        super(TFEstimatorRegressor, self).__init__(model, 
                                                   optimizer, 
                                                   loss,
                                                   batch_size = batch_size, 
                                                   epochs = epochs)
        self.tf_model.output_layer = tf.keras.layers.Dense(units = 1, activation = 'linear')
    
    def fit(self, X, y, verbose = 0) : 
        self.tf_model.build(X.shape)
        self.tf_model.compile(optimizer = self.optimizer,loss = self.loss)
        self.tf_model.fit(X, y, epochs = self.epochs, batch_size = self.batch_size, verbose =verbose)
        return self 

def cross_validation(model_initializer, X, y, cross_val_approach, scoring): 

    res_dict = dict()
    for key in scoring.keys(): 
        res_dict['train_'+key] = []
        res_dict['test_'+key] = []

    for train_index, test_index in cross_val_approach.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = model_initializer()
        model.fit(X_train, y_train)
 
        for key, scorer in scoring.items(): 
            y_pred = model.predict(X_train)
            res_dict['train_'+key].append(scorer(y_pred,y_train))
            y_pred = model.predict(X_test)
            res_dict['test_'+key].append(scorer(y_pred,y_test))

        del model

    return res_dict

def create_pipeline():
    optimizer = tf.keras.optimizers.Adam()
    loss = 'mse'
    custom_model = TFModelRegressor()
    tfestimator = TFEstimatorRegressor(custom_model, optimizer, loss, epochs= 1000)
    scaler = StandardScaler()
    # pipeline = make_pipeline([scaler, tfestimator]) # Doesn't work with make_pipeline, don't know why ......
    pipeline = Pipeline([('standard_scaler', scaler),
                         ('TF_NN', tfestimator)])
    return pipeline

if __name__ == '__main__' : 
    optimizer = tf.keras.optimizers.Adam()
    loss = 'mse'
    custom_model = TFModelRegressor()
    tfestimator = TFEstimatorRegressor(custom_model, optimizer, loss, epochs= 1000)
    scaler = StandardScaler()

    # pipeline = make_pipeline([scaler, tfestimator]) # Doesn't work with make_pipeline, don't know why ......
    pipeline = Pipeline([('standard_scaler', scaler),
                         ('TF_NN', tfestimator)])
    

    X = np.linspace(0,4,100)
    X = np.expand_dims(X, axis = -1)
    y = X**2.

    #X = np.array([[1, 0.5],[2, 1.],[3, 1.5],[4, 5]])
    #y = np.array([1.,2.,3.,4.])
    #y = np.expand_dims(y,axis = -1)

    pipeline.fit(X,y)

    yhat = pipeline.predict(X)
    yhat2 = tfestimator.predict(scaler.transform(X))

    plt.plot(y,yhat,'ro')
    plt.plot(y,yhat,'b-^')
    plt.plot([np.min(y),np.max(y)], [np.min(y),np.max(y)], 'k')
    plt.show()

    cross_val_approach = ShuffleSplit(n_splits = 10, random_state= 0) 
    scoring = {'r2' : r2_score,
                'mse' : mean_squared_error} 

    res_dict = cross_validation(create_pipeline, X, y, cross_val_approach, scoring)

    print(np.mean(res_dict['train_r2']))
    print(np.mean(res_dict['test_r2']))








