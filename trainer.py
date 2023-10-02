import numpy as np


def backward_propagation(result, predictions, weights, bias):
    expected_matrix = np.eye(10)[result]
    print(cost_function(expected_matrix, predictions))


def cost_function(expected_matrix, predictions):
    return np.mean((expected_matrix - predictions) ** 2)

