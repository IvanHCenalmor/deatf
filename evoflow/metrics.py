import numpy as np
from sklearn.metrics import accuracy_score


def mse(true, prediction):
    """
    :param true: True data, is the ground of truth in the evaluation.
    :param prediction: Predicted data, is the data that is evaluated.
    :return: The mean squared error calculated between the true and predicted data.
    """
    return np.sum((true-prediction)**2)/true.shape[0]

def accuracy_error(true, prediction):
    """
    :param true: True data, is the ground of truth in the evaluation.
    :param prediction: Predicted data, is the data that is evaluated.
    :return: The accuracy error calculated between the true and predicted data.
    """
    if len(true.shape) > 1:
        true = np.argmax(true, axis=1)
    if len(prediction.shape) > 1:
        prediction = np.argmax(prediction, axis=1)

    return 1-np.sum(true == prediction)/true.shape[0]

def balanced_accuracy(true, prediction):
    """
    :param true: True data, is the ground of truth in the evaluation.
    :param prediction: Predicted data, is the data that is evaluated.
    :return: The balaced accuracy error calculated between the true and predicted data.
    """
    # Number of classes (n), and the number of examples per class are computed.
    classes, count = np.unique(true, return_counts=True)
    # Weights for each class are computed. Summation of the weights of all examples belonging to a class must be 1/n.
    class_weights = [1/len(classes)/i for i in count]
    # Weights for each example are computed, depending on their class.
    example_weights = [class_weights[i] for i in true]
    # Accuracy is computed weighting each example according to the representation of the class it belongs to in the data.
    return accuracy_score(true, prediction, sample_weight=example_weights)
