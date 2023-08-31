from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score,  make_scorer

def accuracy_scorer(y_true, y_proba, threshold = 0.5):
    y_hat = y_proba > threshold
    conf_matrix = confusion_matrix(y_true, y_hat)
    TN = conf_matrix[0, 0]
    TP = conf_matrix[1, 1]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    accuracy = (TN+TP)/(TN+TP+FN+FP)
    return accuracy

def specificity_scorer(y_true, y_proba, threshold = 0.5):
    y_hat = y_proba > threshold
    conf_matrix = confusion_matrix(y_true, y_hat)
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    specificity = TN/(TN+FP)
    return specificity

def recall_scorer(y_true, y_proba, threshold = 0.5):
    y_hat = y_proba > threshold
    conf_matrix = confusion_matrix(y_true, y_hat)
    TP = conf_matrix[1, 1]
    FN = conf_matrix[1, 0]
    recall = TP/(TP+FN)
    return recall
    
def precision_scorer(y_true, y_proba, threshold = 0.5):
    y_hat = y_proba > threshold
    conf_matrix = confusion_matrix(y_true, y_hat)
    TP = conf_matrix[1, 1]
    FP = conf_matrix[0, 1]
    precision = TP/(TP+FP)
    return precision   

def f1_scorer(y_true, y_proba, threshold = 0.5):
    recall = recall_scorer(y_true, y_proba, threshold = threshold)
    precision = precision_scorer(y_true, y_proba, threshold = threshold) 
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def get_scoring(threshold = 0.5):

    accuracy_scorer_partial = lambda y_true, y_proba : accuracy_scorer(y_true, y_proba, threshold = threshold)
    recall_scorer_partial = lambda y_true, y_proba : recall_scorer(y_true, y_proba, threshold = threshold)
    precision_scorer_partial = lambda y_true, y_proba : precision_scorer(y_true, y_proba, threshold = threshold)
    f1_scorer_partial = lambda y_true, y_proba : f1_scorer(y_true, y_proba, threshold = threshold)
    specificity_scorer_partial = lambda y_true, y_proba : specificity_scorer(y_true, y_proba, threshold = threshold)

    scoring = {'accuracy' : make_scorer(accuracy_scorer_partial, needs_proba = True),
                'roc_auc' : make_scorer(roc_auc_score),
                'recall' : make_scorer(recall_scorer_partial, needs_proba = True),
                'precision' : make_scorer(precision_scorer_partial, needs_proba = True),
                'f1_score' : make_scorer(f1_scorer_partial, needs_proba = True),
                'specificity' : make_scorer(specificity_scorer_partial, needs_proba = True)} 
    return scoring