import NeuralNetwork
import numpy as np

# matrices = [
#   np.random.rand(2, 3),
#   np.random.rand(3, 3),
#   np.random.rand(3, 3),
#   np.random.rand(3, 3),
#   np.random.rand(3, 3),
#   np.random.rand(3, 1)
# ]

weigths, bias = NeuralNetwork.generate_random_neural_network(2, 7, 3, 1)
print((weigths, bias))

NeuralNetwork.save(weigths, bias)

print(NeuralNetwork.load())
