import numpy as np

def get_gradients_previous_layer(weight, dz, previous_activation, coef):
    previous_dz = np.dot(weight.T, dz) * previous_activation * (1 - previous_activation)
    return previous_dz, coef * np.dot(previous_dz, previous_activation.T), coef * np.sum(previous_dz, axis=1,
                                                                                         keepdims=True)


def get_first_dz(y, sol):
    return sol - y


def get_gradients(x, y, activations, weights):
    coef = 1
    depth = len(activations)
    dz = get_first_dz(activations[-1], y)
    gradients_weight = []
    gradients_bias = []

    for i in range(depth - 1):
        dz, gradient_weight, gradient_bias = get_gradients_previous_layer(weights[-(i + 1)], dz, activations[-(i + 2)], coef)
        gradients_weight.append(gradient_weight)
        gradients_bias.append(gradient_bias)

    dz, gradient_weight, gradient_bias = get_gradients_previous_layer(weights[0], dz, x, coef)
    gradients_weight.append(gradient_weight)
    gradients_bias.append(gradient_bias)

    return gradients_weight, gradients_bias
