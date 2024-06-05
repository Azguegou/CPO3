from numpy import array
from numpy import float64 as float128
import ImageToByteArray as imageGetter
import NeuralNetwork
import NeuralNetwork as neuralNetwork
import trainer as trainer

nb_inputs = 1024
nb_layers = 10
nb_neurons = 1500
nb_outputs = 10
training_coef = 0.1
sample_size = 100
nb_test = 100

def adaptExpectedResult(n):
    result = [0 for i in range(10)]
    result[n] = 1
    return result


weights, bias = neuralNetwork.generate_random_neural_network(nb_inputs, nb_layers, nb_neurons, nb_outputs)
#weights, bias = neuralNetwork.load()
preFormattedInput = imageGetter.load_images()


inputMatrix = array([[preFormattedInput[j][1][i][0] for j in range(sample_size)]for i in range(nb_inputs)],dtype=float128)
adaptResult = [adaptExpectedResult(preFormattedInput[i][0]) for i in range(sample_size)]
resultMatrix = array([[adaptResult[j][i] for j in range(sample_size)]for i in range(nb_outputs)],dtype=float128)

for n in range(nb_test):
    print(str(n+1) + "/" + str(nb_test))

    activations = NeuralNetwork.forward_propagation(inputMatrix,weights,bias)
    weight_gradients, bias_gradients = trainer.get_gradients(array(inputMatrix, dtype=float128), resultMatrix, activations, weights)
    print(weight_gradients[0])
    for i in range(len(weight_gradients)):
        weights[i] = weights[i] - weight_gradients[len(weight_gradients)-i-1] * training_coef
        bias[i] = bias[i] - bias_gradients[len(bias_gradients)-i-1] * training_coef

    NeuralNetwork.save(weights, bias)
    print("saved")




