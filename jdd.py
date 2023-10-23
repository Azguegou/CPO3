from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import *
from tensorflow.keras.optimizers import *
from keras.models import *
from keras.layers import *
from matplotlib import pyplot

# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    pyplot.subplot(330 + 1)
    pyplot.imshow(trainX[100], cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    print(trainY[100])
    return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

def training_set():
    trainX, trainY, testX, testY = load_dataset()
    train_norm_X, test_norm_X = prep_pixels(trainX, testX)
    train_norm_Y, test_norm_Y = prep_pixels(trainY, testY)
    return train_norm_X, train_norm_Y, test_norm_X, test_norm_Y
