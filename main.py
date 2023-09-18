import numpy as np

import NeuralNetwork as neuralNetwork
import ImageToByteArray as imageGetter

matrices = [
    np.random.rand(2, 3),
    np.random.rand(3, 3),
    np.random.rand(3, 3),
    np.random.rand(3, 3),
    np.random.rand(3, 3),
    np.random.rand(3, 1)
]

weights, bias = neuralNetwork.generate_random_neural_network(2, 7, 3, 1)

neuralNetwork.save(weights, bias)

print(neuralNetwork.load())
print(imageGetter.read_image())

imageGetter.load_images()
