import numpy as np
import NeuralNetwork as nN

matrices = [
    [[1, 2, 3],
     [1, 2, 2]],
    [[1, 2, 3],
     [1, 2, 2],
     [1, 3, 4]],
    [[6],
     [4],
     [1]]
]
print(matrices)
nN.save(matrices)
print(nN.load())

