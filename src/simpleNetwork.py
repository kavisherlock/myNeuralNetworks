"""
A program for training and running simple neural networks.
Based on Michael Neilson's online book "Neural Networks and Deep Learning"
http://neuralnetworksanddeeplearning.com/
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoidPrime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return a-y


class Network(object):
    """
    Initializes the network
    Args:
        dimensions: example: [2, 3, 1] is a three-layer network, with the first layer containing
                    2 neurons, the second layer 3 neurons, and the third layer 1 neuron.
        cost: cost function being used for the network. Can be quadratic, or cross-entropy
    """
    def __init__(self, dimensions, cost=CrossEntropyCost):
        self.numLayers = len(dimensions)
        self.sizes = dimensions
        # Randomly initializing using a Gaussian distribution with mean 0, and variance 1
        self.biases = [np.random.randn(y, 1) for y in dimensions[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(dimensions[:-1], dimensions[1:])]
        self.cost = cost

    """
    Gives output of the network by feed-forwarding the input
    Args:
        a: (n, 1) Numpy ndarray representing the input, where n is the no. of inputs
    Returns: output of the network
    """
    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    """
    Mini-Batch Stochastic Gradient Descent.
    Args:
        trainingData: list of tuples (x, y) representing the training inputs and the desired outputs
        nEpochs: number of n_epochs to run the algorithm
        miniBatchSize: size of the mini batch to use when sampling
        eta: the learning rate
        lmbda: the regularization parameter
        validationData: optional parameter; if provided, network will be evaluated against the test data 
                   after each epoch and partial progress printed out.
        track*: monitor the cost and accuracy on either the evaluation data or the training data, 
                 by setting the appropriate flags.
    Returns: if the track_* flags are set, returns a tuple containing four lists: the (per-epoch) costs 
             on the evaluation data, the accuracies on the evaluation data, the costs on the training
             data, and the accuracies on the training data respectively
    """
    def SGD(self, trainingData, nEpochs, miniBatchSize, eta, lmbda=0.0, validationData=None,
            trackValCost=False, trackValAccuracy=False, trackTrainCost=False, trackTrainAccuracy=False):
        nEval = 0
        if validationData:
            nEval = len(validationData)

        evaluationCost, evaluationAccuracy = [], []
        trainingCost, trainingAccuracy = [], []

        n = len(trainingData)

        for j in xrange(nEpochs):
            random.shuffle(trainingData)
            miniBatches = [trainingData[k:k + miniBatchSize] for k in xrange(0, n, miniBatchSize)]

            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta, lmbda, n)

            print "Epoch %s training complete" % j
            if trackTrainCost:
                cost = self.totalCost(trainingData, lmbda)
                trainingCost.append(cost)
                print "Cost on training data: {}".format(cost)

            if trackTrainAccuracy:
                accuracy = self.accuracy(trainingData, convert=True)
                trainingAccuracy.append(accuracy)
                print "Accuracy on training data: {} / {}".format(accuracy, n)

            if trackValCost:
                cost = self.totalCost(validationData, lmbda, convert=True)
                evaluationCost.append(cost)
                print "Cost on evaluation data: {}".format(cost)

            if trackValAccuracy:
                accuracy = self.accuracy(validationData)
                evaluationAccuracy.append(accuracy)
                print "Accuracy on evaluation data: {} / {}".format(self.accuracy(validationData), nEval)
            print

        return evaluationCost, evaluationAccuracy, trainingCost, trainingAccuracy

    """
    Update the network's weights and biases by applying gradient descent using backpropagation.
    Args:
        miniBatch: list of tuples (x, y) on which gradient descent if performed
        eta: the learning rate
        lmbda: the regularization parameter
        n: number of training examples
    """
    def updateMiniBatch(self, miniBatch, eta, lmbda, n):
        # Initialize gradients
        gradB = [np.zeros(b.shape) for b in self.biases]
        gradW = [np.zeros(w.shape) for w in self.weights]

        # Calculate gradients using backpropagation
        for x, y in miniBatch:
            deltaGradB, deltaGradW = self.backprop(x, y)
            gradB = [nb + dnb for nb, dnb in zip(gradB, deltaGradB)]
            gradW = [nw + dnw for nw, dnw in zip(gradW, deltaGradW)]

        # Update the weights and biases
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(miniBatch)) * nw
                        for w, nw in zip(self.weights, gradW)]
        self.biases = [b - (eta / len(miniBatch)) * nb
                       for b, nb in zip(self.biases, gradB)]

    """
    Backpropagation
    Args:
        x: input to the network
        y: desired output
    Returns: a tuple gradB, gradW representing the gradient for the cost function C_x.
    """
    def backprop(self, x, y):
        gradB = [np.zeros(b.shape) for b in self.biases]
        gradW = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        gradB[-1] = delta
        gradW[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.numLayers):
            z = zs[-l]
            sp = sigmoidPrime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            gradB[-l] = delta
            gradW[-l] = np.dot(delta, activations[-l - 1].transpose())

        return gradB, gradW

    """
    Evaluates the network on the test data
    Args:
        valData: evaluation data to evaluate on
    Returns: the number of test inputs for which the neural network outputs the correct result
    """
    def evaluate(self, valData):
        testResults = [(np.argmax(self.feedforward(x)), y) for (x, y) in valData]

        return sum(int(x == y) for (x, y) in testResults)

    """
    Calculates the accuracy of the neural network
    Args:
        data: data to check accuracy on
        convert: flag to be set true for training data
    Returns: the number of test inputs for which the neural network outputs the correct result
    """
    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    """
    Calculates the total cost for the data set
    Args:
        data: data to check accuracy on
        lmbda: the regularization parameter
        convert: flag to be set true for evaluation data
    Returns: the total cost for the data set
    """
    def totalCost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorizedResult(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost


def vectorizedResult(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    # The sigmoid function.
    return 1.0 / (1.0 + np.exp(-z))


def sigmoidPrime(z):
    # The derivative of the sigmoid function.
    return sigmoid(z) * (1 - sigmoid(z))
