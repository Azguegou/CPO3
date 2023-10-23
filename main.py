import os

import NeuralNetwork as neuralNetwork
import ImageToByteArray as imageGetter
import trainer as trainer

nb_inputs = 1024
nb_layers = 3
nb_neurons = 100
nb_outputs = 10

weights, bias = neuralNetwork.generate_random_neural_network(nb_inputs, nb_layers, nb_neurons, nb_outputs)
neuralNetwork.save(weights, bias, os.path.join(os.getcwd(), "data.txt"))
a, b = imageGetter.load_images()[0]
activations = neuralNetwork.forward_propagation(b, weights, bias)
print(trainer.get_gradients(b, a, activations, weights))
