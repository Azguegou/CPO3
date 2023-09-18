import numpy as np
import NeuralNetwork
import ImageToByteArray as imageGetter

# matrices = [
#   np.random.rand(2, 3),
#   np.random.rand(3, 3),
#   np.random.rand(3, 3),
#   np.random.rand(3, 3),
#   np.random.rand(3, 3),
#   np.random.rand(3, 1)
# ]

weights, bias = NeuralNetwork.generate_random_neural_network(2, 7, 3, 1)
print((weights, bias))

NeuralNetwork.save(weights, bias)

print(NeuralNetwork.load())
print(imageGetter.read_image())

(imageGetter.load_images())
