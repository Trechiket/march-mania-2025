import numpy as np


def brier_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean((y_pred - y_true) ** 2)