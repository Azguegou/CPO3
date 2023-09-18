# Récupérer le txt avec la matrice de poids
# Récupérer l'input liste de matrices
# Renvoyer liste de matrices
import numpy as np


def write_matrix(matrix, end_base, file):
    for x in range(len(matrix)):
        line = matrix[x]
        for y in range(len(matrix[x])):
            file.write("%0.6f " % line[y])
        file.write("\n")
    file.write(end_base)


def save(weights, bias):
    save_file = open("data.txt", "w")

    for idMatrix in range(len(weights)):
        write_matrix(bias[idMatrix], "END_WEIGHT", save_file)

    save_file.write("END_LIST")

    for idMatrix in range(len(bias)):
        write_matrix(bias[idMatrix], "END_BIAS", save_file)

    save_file.close()


def load():
    weight_list = []
    bias_list = []

    file_data = open("data.txt", "r").read().split("END_LIST")
    weight_data = file_data[0]
    bias_data = file_data[1]

    for weight in weight_data.split("END_WEIGHT")[0:-1]:
        weight_wip = []
        for line in weight.split("\n")[0:-1]:
            line_wip = [float(value) for value in line.split(" ")[0:-1]]
            weight_wip.append(line_wip)
        weight_list.append(weight_wip)

    for bias in bias_data.split("END_BIAS")[0:-1]:
        bias_wip = []
        for line in bias.split("\n")[0:-1]:
            line_wip = [float(value) for value in line.split(" ")[0:-1]]
            bias_wip.append(line_wip)
        bias_list.append(bias_wip)

    return weight_list, bias_list


def generate_random_neural_network(nb_inputs, nb_layer, nb_neurons, nb_outputs):
    weight_matrices = [np.random.rand(nb_inputs, nb_neurons)]
    for i in range(nb_layer - 1):
        weight_matrices.append(np.random.rand(nb_neurons, nb_neurons))

    weight_matrices.append(np.random.rand(nb_neurons, nb_outputs))

    bias_matrices = []
    for i in range(nb_layer):
        bias_matrices.append(np.random.rand(nb_neurons, 1))
    bias_matrices.append(np.random.rand(nb_outputs, 1))

    return weight_matrices, bias_matrices


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_propagation(inputs, weights, bias):
    sol = inputs
    for i in range(len(weights)):
        sol = sigmoid(np.dot(sol, weights[i]) + bias[i])
    return sol
