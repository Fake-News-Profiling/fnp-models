import numpy as np


def _array_comp(y_true, y_true_val, y_pred, y_pred_val):
    return np.sum(np.logical_and(y_pred == y_true_val, y_true == y_pred_val))


def true_positives(y_true, y_pred):
    """ y_true == 1 and y_pred == 1 """
    return _array_comp(y_true, 1, y_pred, 1)


def false_negatives(y_true, y_pred):
    """ y_true == 1 and y_pred == 0 """
    return _array_comp(y_true, 1, y_pred, 0)


def true_negatives(y_true, y_pred):
    """ y_true == 0 and y_pred == 0 """
    return _array_comp(y_true, 0, y_pred, 0)


def false_positives(y_true, y_pred):
    """ y_true == 0 and y_pred == 1 """
    return _array_comp(y_true, 0, y_pred, 1)