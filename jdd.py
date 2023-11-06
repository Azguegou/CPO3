from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import *


def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28))
    trainY = to_categorical(trainY)
    return trainX, trainY


def prep_pixels(train, test):
    train_norm = train.astype('float32')
    train_norm = train_norm / 255.0
    return train_norm


def training_set():
    trainX, trainY = load_dataset()
    train_norm_X = prep_pixels(trainX)
    train_norm_Y = prep_pixels(trainY)
    return train_norm_X, train_norm_Y
