import numpy as np

def confusion_matrix(actual, predictions):
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    else:
        c_matrix = np.zeros((2,2))
        for i in range(predictions.shape[0]):
            if float(predictions[i]) == 1:
                if predictions[i] == actual[i]:
                    c_matrix[1,1] += 1
                else: 
                    c_matrix[0,1] += 1
            else:
                if predictions[i] == actual[i]:
                    c_matrix[0,0] += 1
                else:
                    c_matrix[1,0] += 1
    return c_matrix



def accuracy(actual, predictions):
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    else:
        cm = confusion_matrix(actual,predictions)
    return (cm[0,0] + cm[1,1])/np.sum(cm)

def precision_and_recall(actual, predictions):
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    else:
        cm = confusion_matrix(actual, predictions)
        if (cm[1,1]+cm[0,1]) == 0:
            return 0, 0
        else: 
            precision = float(cm[1,1])/(cm[1,1]+cm[0,1])
        if (cm[1,1]+cm[1,0]) == 0:
            return 0, 0
        else:
            recall = float(cm[1,1])/(cm[1,1]+cm[1,0])
    return precision, recall

def f1_measure(actual, predictions):
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    else:
        precision, recall = precision_and_recall(actual,predictions)
        if recall == 0 or precision == 0 or ((1/recall)+(1/precision)) == 0:
            return 0
        else:
            return 2/((1/recall)+(1/precision))


