import numpy as np

import ImageToByteArray as imageGetter
import NeuralNetwork

matrices = [
    np.random.rand(2, 3),
    np.random.rand(3, 3),
    np.random.rand(3, 3),
    np.random.rand(3, 3),
    np.random.rand(3, 3),
    np.random.rand(3, 1)
]

weights, bias = NeuralNetwork.generate_random_neural_network(1024, 5, 1500, 10)
a, b = imageGetter.load_images()[0]
lis = NeuralNetwork.forward_propagation(b, weights, bias)
print(lis)
