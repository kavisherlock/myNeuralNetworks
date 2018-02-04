"""
A Theano-based program for training and running simple neural networks.
Based on Michael Neilson's online book "Neural Networks and Deep Learning"
http://neuralnetworksanddeeplearning.com/
"""

"""Libraries"""
# Standard library
import random

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import pool

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


"""Constants"""
GPU = True
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then modify "+\
        "theanoNetwork.py\nto set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify "+\
        "theanoNetwork.py to set\nthe GPU flag to True."


class Network(object):
    """
    Initializes the network
    Args:
        layers:
        miniBatchSize: size of the mini batch for SGD
    """
    def __init__(self, layers, miniBatchSize):
        self.layers = layers
        self.miniBatchSize = miniBatchSize
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")

        initLayer = self.layers[0]
        initLayer.set_inpt(self.x, self.x, self.miniBatchSize)

        for j in xrange(1, len(self.layers)):
            prevLayer, layer = self.layers[j - 1], self.layers[j]
            layer.set_inpt(
                prevLayer.output, prevLayer.outputDropout, self.miniBatchSize)

        self.output = self.layers[-1].output
        self.outputDropout = self.layers[-1].outputDropout
        self.testMiniBatchPredictions = None

    """
    Mini-Batch Stochastic Gradient Descent.
    Args:
        trainingData: list of tuples (x, y) representing the training inputs and the desired outputs
        epochs: number of n_epochs to run the algorithm
        miniBatchSize: size of the mini batch to use when sampling
        eta: the learning rate
        lmbda: the regularization parameter
        validationData: optional parameter; if provided, network will .
        testData: optional parameter; if provided, network will .
    """
    def SGD(self, trainingData, epochs, miniBatchSize, eta, validationData=None, testData=None, lmbda=0.0):
        trainingX, trainingY = trainingData
        validationX, validaitionY = validationData
        testX, testY = testData

        # compute number of mini-batches for training, validation and testing
        numTrainingBatches = size(trainingData) / miniBatchSize
        numValidationBatches = size(validationData) / miniBatchSize
        numTestBatches = size(testData) / miniBatchSize

        # define the (regularized) cost function, symbolic gradients, and updates
        l2NormSquared = sum([(layer.w ** 2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + 0.5 * lmbda * l2NormSquared / numTrainingBatches
        grads = T.grad(cost, self.params)
        updates = [(param, param - eta * grad) for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the accuracy in validation and test mini-batches.
        i = T.lscalar()  # mini-batch index
        trainMiniBatch = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x: trainingX[i*self.miniBatchSize: (i+1)*self.miniBatchSize],
                self.y: trainingY[i*self.miniBatchSize: (i+1)*self.miniBatchSize]
            })
        validateMiniBatchAccuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: validationX[i*self.miniBatchSize: (i+1)*self.miniBatchSize],
                self.y: validaitionY[i*self.miniBatchSize: (i+1)*self.miniBatchSize]
            })
        testMiniBatchAccuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: testX[i*self.miniBatchSize: (i+1)*self.miniBatchSize],
                self.y: testY[i*self.miniBatchSize: (i+1)*self.miniBatchSize]
            })
        self.testMiniBatchPredictions = theano.function(
            [i], self.layers[-1].yOut,
            givens={
                self.x: testX[i*self.miniBatchSize: (i+1)*self.miniBatchSize]
            })

        # Do the actual training
        bestValidationAccuracy = 0.0
        bestIteration = 0
        testAccuracy = 0
        for epoch in xrange(epochs):
            for minibatchIndex in xrange(numTrainingBatches):
                iteration = numTrainingBatches * epoch + minibatchIndex
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = trainMiniBatch(minibatchIndex)
                if (iteration+1) % numTrainingBatches == 0:
                    validationAccuracy = np.mean([validateMiniBatchAccuracy(j) for j in xrange(numValidationBatches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(epoch, validationAccuracy))
                    if validationAccuracy >= bestValidationAccuracy:
                        print("This is the best validation accuracy to date.")
                        bestValidationAccuracy = validationAccuracy
                        bestIteration = iteration
                        if testData:
                            testAccuracy = np.mean([testMiniBatchAccuracy(j) for j in xrange(numTestBatches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(testAccuracy))

        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            bestValidationAccuracy, bestIteration))
        print("Corresponding test accuracy of {0:.2%}".format(testAccuracy))


"""Layers"""
class ConvPoolLayer(object):
    """
    Initializes the layer
    Args:
        filterShape: tuple of length 4, whose entries are the number of filters, the number of input
                     feature maps, the filter height, and the filter width
        imageShape: tuple of length 4, whose entries are the mini-batch size, the number of input
                    feature maps, the image height, and the image width.
        poolSize: tuple of length 2, whose entries are the y and x pooling sizes.
        activationFn: activation function. By default, it is the sigmoid function
    """
    def __init__(self, filterShape, imageShape, poolSize=(2, 2), activationFn=sigmoid):
        self.filterShape = filterShape
        self.imageShape = imageShape
        self.poolSize = poolSize
        self.activationFn = activationFn
        self.inpt=None
        self.output=None
        self.outputDropout=None

        # initialize weights and biases
        nOut = (filterShape[0] * np.prod(filterShape[2:]) / np.prod(poolSize))
        self.w = theano.shared(
            np.asarray(np.random.normal(loc=0, scale=np.sqrt(1.0 / nOut), size=filterShape), dtype=theano.config.floatX),
            borrow=True
        )
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0, scale=1.0, size=(filterShape[0],)), dtype=theano.config.floatX),
            borrow=True
        )
        self.params = [self.w, self.b]

    """
    Set the input to the layer, and to compute the corresponding output
    Args:
        inpt: the input to the later
    """
    def set_inpt(self, inpt):
        self.inpt = inpt.reshape(self.imageShape)

        convOut = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filterShape,
            image_shape=self.imageShape)
        pooledOut = pool.pool_2d(input=convOut, ds=self.poolSize, ignore_border=True)

        self.output = self.activationFn(
            pooledOut + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.outputDropout = self.output  # no dropout in the convolutional layers


class FullyConnectedLayer(object):
    """
    Initializes the layer
    Args:
        nIn:
        nOut:
        activationFn: activation function. By default, it is the sigmoid function
        pDropout:
    """
    def __init__(self, nIn, nOut, activationFn=sigmoid, pDropout=0.0):
        self.nIn = nIn
        self.nOut = nOut
        self.activationFn = activationFn
        self.pDropout = pDropout
        self.inpt = None
        self.output = None
        self.yOut = None
        self.inptDropout = None
        self.outputDropout = None

        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=np.sqrt(1.0 / nOut), size=(nIn, nOut)), dtype=theano.config.floatX),
            name='w',
            borrow=True
        )
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(nOut,)), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )
        self.params = [self.w, self.b]

    """
    Set the input to the layer, and to compute the corresponding output
    Args:
        inpt: the input to the later
        inptDropout:
        miniBatchSize: mini-batch size
    """
    def set_inpt(self, inpt, inptDropout, miniBatchSize):
        self.inpt = inpt.reshape((miniBatchSize, self.nIn))
        self.output = self.activationFn(
            (1-self.pDropout)*T.dot(self.inpt, self.w) + self.b)
        self.yOut = T.argmax(self.output, axis=1)
        self.inptDropout = dropoutLayer(inptDropout.reshape((miniBatchSize, self.nIn)), self.pDropout)
        self.outputDropout = self.activationFn(T.dot(self.inptDropout, self.w) + self.b)

    """
    Return the accuracy for the mini-batch.
    Args:
        y: actual output
    """
    def accuracy(self, y):
        return T.mean(T.eq(y, self.yOut))


class SoftmaxLayer(object):
    """
    Initializes the layer
    Args:
        nIn:
        nOut:
        pDropout:
    """
    def __init__(self, nIn, nOut, pDropout=0.0):
        self.nIn = nIn
        self.nOut = nOut
        self.pDropout = pDropout
        self.inpt = None
        self.output = None
        self.yOut = None
        self.inptDropout = None
        self.outputDropout = None

        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((nIn, nOut), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((nOut,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    """
    Set the input to the layer, and to compute the corresponding output
    Args:
        inpt: the input to the later
        inptDropout:
        miniBatchSize: mini-batch size
    """
    def set_inpt(self, inpt, inptDropout, miniBatchSize):
        self.inpt = inpt.reshape((miniBatchSize, self.nIn))
        self.output = softmax((1 - self.pDropout) * T.dot(self.inpt, self.w) + self.b)
        self.yOut = T.argmax(self.output, axis=1)
        self.inptDropout = dropoutLayer(inptDropout.reshape((miniBatchSize, self.nIn)), self.pDropout)
        self.outputDropout = softmax(T.dot(self.inptDropout, self.w) + self.b)

    """
    Return the log-likelihood cost.
    Args:
       net:
    """
    def cost(self, net):
        return -T.mean(T.log(self.outputDropout)[T.arange(net.y.shape[0]), net.y])

    """
    Return the accuracy for the mini-batch.
    Args:
       y: actual output
    """
    def accuracy(self, y):
        return T.mean(T.eq(y, self.yOut))


"""Miscellaneous Functions"""


def size(data):
    return data[0].get_value(borrow=True).shape[0]


def dropoutLayer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
