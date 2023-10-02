import numpy as np


def write_matrix(matrix, closing_tag, file_obj):
    for row in matrix:
        file_obj.write(" ".join([f"{value:.6f}" for value in row]) + "\n")
    file_obj.write(closing_tag)


def save(weights, bias, file_path):
    with open(file_path, "w") as save_file:
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
            weight_list.append([[float(value) for value in line.split()] for line in weight.strip().split("\n")])

        for bias in bias_data.split("END_BIAS")[:-1]:
            bias_list.append([[float(value) for value in line.split()] for line in bias.strip().split("\n")])

    return weight_list, bias_list


def generate_random_neural_network(nb_inputs, nb_layer, nb_neurons, nb_outputs):
    weight_matrices = [np.random.uniform(-1, 1, (nb_inputs, nb_neurons))]
    for i in range(nb_layer - 1):
        weight_matrices.append(np.random.uniform(-1, 1, (nb_neurons, nb_neurons)))

    weight_matrices.append(np.random.uniform(-1, 1, (nb_neurons, nb_outputs)))

    bias_matrices = []
    for i in range(nb_layer):
        bias_matrices.append(np.random.uniform(-1, 1, (1, nb_neurons)))
    bias_matrices.append(np.random.uniform(-1, 1, (1, nb_outputs)))
    return weight_matrices, bias_matrices


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_propagation(inputs, weights, bias):
    sol = inputs
    for i in range(len(weights)):
        sol = sigmoid(np.dot(sol, weights[i]) + bias[i])
    return sol
