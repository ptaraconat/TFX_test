import sys as sys 
sys.path.append('../')
from custom_estimators import * 
from sklearn.datasets import load_breast_cancer
from scores_num import *

def create_pipeline():
    optimizer = tf.keras.optimizers.Adam()
    loss = 'mse'
    custom_model = TFModelClassifier(hidden_layers=[tf.keras.layers.Dense(units = 5, activation = 'relu'),
                                                    tf.keras.layers.Dense(units = 5, activation = 'relu')])
    tfestimator = TFEstimatorClassifier(custom_model, optimizer, loss, epochs= 1000)
    scaler = StandardScaler()
    # pipeline = make_pipeline([scaler, tfestimator]) # Doesn't work with make_pipeline, don't know why ......
    pipeline = Pipeline([('standard_scaler', scaler),
                         ('TF_NN', tfestimator)])
    return pipeline

def main(): 
    # Define some data 
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(X.shape)
    print(y.shape)
    # Define model
    pipeline = create_pipeline()
    # fit model
    pipeline.fit(X,y)
    # test and display model results 
    yhat = pipeline.predict(X)
    confusion_mat = confusion_matrix(y, yhat)
    print(confusion_mat)
    # Cross validation of model performances 
    cross_val_approach = ShuffleSplit(n_splits = 10, random_state= 0, test_size= 0.5) 
    scoring = {'accuracy' : accuracy_score,
                'recall' : recall_score} 
    res_dict = cross_validation(create_pipeline, X, y, cross_val_approach, scoring)
    print('train mean accuracy ::: ', np.mean(res_dict['train_accuracy']))
    print('train mean accuracy ::: ', np.std(res_dict['train_accuracy']))
    print('test mean accuracy ::: ', np.mean(res_dict['test_accuracy']))
    print('test mean accuracy ::: ', np.std(res_dict['test_accuracy']))

if __name__ == '__main__':
    main()