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
                 hidden_layers = [tf.keras.layers.Dense(units = 10, 
                                                        activation = 'relu'),
                                  tf.keras.layers.Dense(units = 10, 
                                                        activation = 'relu')],
                                  output_activation = 'linear'):
        '''
        Arguments : 
        hidden_layers ::: list of keras.layers ::: Hidden layers of the NN 
        output_activation ::: str ::: activation used for the 
        prediction/last layer 
        Return : 
        None 
        '''
        super(TFModel, self).__init__()
        self.hidden_layers = hidden_layers
        self.output_activation = output_activation

    def call(self,input):
        '''
        Call method. Calculate the forward pass in the NN 
        Arguments : 
        input ::: array like object [batch_size, input_shape] ::: examples for 
        which the prediction will be made 
        Return : 
        x ::: array like object [batch_size, output_dim] ::: NN Prediction
        '''
        x = input 
        # loop over hidden_layers list 
        for layer in self.hidden_layers : 
            x = layer(x)
        x = self.output_layer(x)
        return x 
    
    def set_output_dim(self, output_dim): 
        '''
        Set model output layer given the output dimension. 
        Will work only for flat output. Should ba called before building model. 
        Arguments : 
        output_dim ::: int ::: Dimension of the NN prediction 
        Return : 
        None 
        '''
        self.output_layer = tf.keras.layers.Dense(units = output_dim, 
                                                  activation = self.output_activation)

class TFModelRegressor(TFModel): 
    '''
    Subclass of TFModel. 
    Dedicated to Regression models. 
    Output dim is only 1D, ATM. 
    '''
    def __init__(self,
                 hidden_layers = [tf.keras.layers.Dense(units = 10, 
                                                        activation = 'relu'),
                                  tf.keras.layers.Dense(units = 10, 
                                                        activation = 'relu')]): 
        '''
        Arguments : 
        hidden_layers ::: list of keras.layers ::: Hidden layers of the NN 
        Return : 
        None
        '''
        super(TFModelRegressor, self).__init__(hidden_layers= hidden_layers)
        self.output_activation = 'linear'
        self.set_output_dim(1)

class TFModelClassifier(TFModel): 
    '''
    Subclass of TFModel. 
    Dedicated to Binary Classification models, ATM. 
    '''
    def __init__(self, 
                 hidden_layers = [tf.keras.layers.Dense(units = 10, activation = 'relu'), 
                                  tf.keras.layers.Dense(units = 10, activation = 'relu')]):
        '''
        Arguments : 
        hidden_layers ::: list of keras.layers ::: Hidden layers of the NN 
        Return : 
        None
        '''
        super(TFModelClassifier, self).__init__(hidden_layers = hidden_layers)
        self.output_activation = 'sigmoid'
        self.set_output_dim(1)

class TFEstimator(BaseEstimator):
    ''' TensorFlow interfacer
    '''
    def __init__(self,model,optimizer,loss,batch_size = 200, epochs = 100):
        '''
        Arguments : 
        model ::: TFModel object ::: Neural Network model 
        optimizer ::: tf.keras.optimizer object::: optimizer used for updating NN 
        weights
        loss ::: tf.keras.loss object or str ::: loss function for computing model 
        errors during training
        batch_size ::: int ::: Sample size 
        epochs ::: int ::: number of time the optimization iterates over the entire 
        dataset
        '''

        self.tf_model = model 
        self.optimizer = optimizer 
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y) : 
        '''
        Fit estimator to data 
        Arguments : 
        X ::: array like object [batch_size, input_shape] ::: Input data 
        y ::: array like object [batch_size, output_dim] ::: Model targets 
        Return : 
        None  
        '''
        self.tf_model.set_output_dim(y.shape[-1])
        self.tf_model.build(X.shape)
        self.tf_model.compile(optimizer = self.optimizer,loss = self.loss)
        self.tf_model.fit(X, y, epochs = self.epochs, batch_size = self.batch_size)
        return self 
    
    def predict(self, X): 
        '''
        Arguments : 
        X ::: array like object [batch_size, input_shape] ::: Input data 
        Return : 
        Predictions ::: array like object [batch_size, output_dim] ::: Model 
        prediction
        '''
        prediction = self.tf_model(X)
        return prediction.numpy()
    
    def score(self, X, y):
        return 0 
    
    def get_params(self, deep=True):
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
        '''
        Arguments : 
        model ::: TFModel object ::: Neural Network model 
        optimizer ::: tf.keras.optimizer object::: optimizer used for updating NN 
        weights
        loss ::: tf.keras.loss object or str ::: loss function for computing model 
        errors during training
        batch_size ::: int ::: Sample size 
        epochs ::: int ::: number of time the optimization iterates over the entire 
        dataset
        '''
        super(TFEstimatorRegressor, self).__init__(model, 
                                                   optimizer, 
                                                   loss,
                                                   batch_size = batch_size, 
                                                   epochs = epochs)
        #self.tf_model.output_layer = tf.keras.layers.Dense(units = 1, activation = 'linear')
    
    def fit(self, X, y, verbose = 0) : 
        '''
        Fit estimator to data 
        Arguments : 
        X ::: array like object [batch_size, input_shape] ::: Input data 
        y ::: array like object [batch_size, output_dim] ::: Model targets 
        verbose ::: int ::: 0 or 1 
        Return : 
        None  
        '''
        self.tf_model.build(X.shape)
        self.tf_model.compile(optimizer = self.optimizer,loss = self.loss)
        self.tf_model.fit(X, y, epochs = self.epochs, batch_size = self.batch_size, 
                          verbose =verbose)
        return self 

class TFEstimatorClassifier(TFEstimator): 

    def __init__(self,model,optimizer,loss,batch_size = 200, epochs = 100):
        '''
        Arguments : 
        model ::: TFModel object ::: Neural Network model 
        optimizer ::: tf.keras.optimizer object::: optimizer used for updating NN 
        weights
        loss ::: tf.keras.loss object or str ::: loss function for computing model 
        errors during training
        batch_size ::: int ::: Sample size 
        epochs ::: int ::: number of time the optimization iterates over the entire 
        dataset
        '''
        super(TFEstimatorClassifier, self).__init__(model, 
                                                   optimizer, 
                                                   loss,
                                                   batch_size = batch_size, 
                                                   epochs = epochs)
        self.tf_model.output_layer = tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
    
    def fit(self, X, y, verbose = 0) : 
        '''
        Fit estimator to data 
        Arguments : 
        X ::: array like object [batch_size, input_shape] ::: Input data 
        y ::: array like object [batch_size, output_dim] ::: Model targets 
        verbose ::: int ::: 0 or 1 
        Return : 
        None  
        '''
        self.tf_model.build(X.shape)
        self.tf_model.compile(optimizer = self.optimizer,loss = self.loss)
        self.tf_model.fit(X, y, epochs = self.epochs, batch_size = self.batch_size, 
                          verbose =verbose)
        return self 
    
    def predict_proba(self, X): 
        '''
        Arguments : 
        X ::: array like object [batch_size, input_shape] ::: Input data 
        Return : 
        Predictions ::: array like object [batch_size, output_dim] ::: Model 
        prediction
        '''
        prediction = self.tf_model(X)
        return prediction.numpy()
    
    def predict(self, X, threshold = 0.5): 
        '''
        Arguments : 
        X ::: array like object [batch_size, input_shape] ::: Input data 
        threhold ::: float ::: included in [0, 1], probability threshold for 
        performing classification 
        Return : 
        Predictions ::: array like object [batch_size, output_dim] ::: Model 
        predictions
        '''
        prediction = self.predict_proba(X) > threshold
        return prediction

def cross_validation(model_initializer, X, y, cross_val_approach, scoring): 
    '''
    Cross validation function. Implemented because sklearn cross_validate 
    didn't work with the TFModel
    Arguments : 
    model_initializer ::: callable ::: Function that initialize the estimator 
    X ::: array like object [batch_size, input_shape] ::: Model input 
    y ::: array like object [batch_size, output_dim] ::: Model targets 
    cross_val_approach ::: sklearn splitter object ::: Approach used for 
    splitting data during cross validation 
    scoring ::: dictionnary ::: dictionary of the scorer 
    '''
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
