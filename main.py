import os

import NeuralNetwork as neuralNetwork
import ImageToByteArray as imageGetter
import trainer as trainer
import JDD as jdd

nb_inputs = 1024
nb_layers = 3
nb_neurons = 1500
nb_outputs = 10

weights, bias = neuralNetwork.generate_random_neural_network(nb_inputs, nb_layers, nb_neurons, nb_outputs)
