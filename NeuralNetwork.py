import numpy as np
from numpy import array
from numpy import float64 as float128


def write_matrix(matrix, closing_tag, file_obj):
    for row in matrix:
        file_obj.write(" ".join([f"{value:.10f}" for value in row]) + "\n")
    file_obj.write(closing_tag)


def save(weights, bias):
    with open("data.txt", "w") as save_file:
        for weight in weights:
            write_matrix(weight, "END_WEIGHT", save_file)
        save_file.write("END_LIST")
        for b in bias:
            write_matrix(b, "END_BIAS", save_file)


def load():
    weight_list = []
    bias_list = []

    with open("data.txt", "r") as file_data:
        file_data = file_data.read().split("END_LIST")
        weight_data = file_data[0]
        bias_data = file_data[1]

        for weight in weight_data.split("END_WEIGHT")[:-1]:
            weight_list.append(
                array([[float(value) for value in line.split()] for line in weight.strip().split("\n")],
                      dtype=float128))

        for bias in bias_data.split("END_BIAS")[:-1]:
            bias_list.append(
                array([[float(value) for value in line.split()] for line in bias.strip().split("\n")], dtype=float128))

    return weight_list, bias_list


def generate_random_neural_network(nb_inputs, nb_layer, nb_neurons, nb_outputs):
    weight_matrices = [np.random.randn(nb_neurons, nb_inputs)]
    for i in range(nb_layer - 1):
        weight_matrices.append(np.random.randn(nb_neurons, nb_neurons))
    weight_matrices.append(np.random.randn(nb_outputs, nb_neurons))

    bias_matrices = []
    for i in range(nb_layer):
        bias_matrices.append(np.random.randn(nb_neurons, 1))
    bias_matrices.append(np.random.randn(nb_outputs, 1))

    return weight_matrices, bias_matrices


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward_propagation(inputs, weights, bias):
    sol = inputs
    activations = []
    for i in range(len(weights)):
        sol = sigmoid(np.dot(weights[i], sol) + bias[i])
        activations.append(sol)
    return activations
