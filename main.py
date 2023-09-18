import numpy as np
import NeuralNetwork as nN
import parse

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


str = "1.000 1.2000"
print(list(str))
print(str.split(" "))
print([float(i) for i in str.split(" ")])

