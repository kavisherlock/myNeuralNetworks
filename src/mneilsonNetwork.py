#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class Network(object):
    """
    Initializes the network
    Args:
        dimensions: example: [2, 3, 1] is a three-layer network, with the first layer containing
                    2 neurons, the second layer 3 neurons, and the third layer 1 neuron.
    """
    def __init__(self, dimensions):
        self.num_layers = len(dimensions)
        self.sizes = dimensions
        # Randomly initializing using a Gaussian distribution with mean 0, and variance 1
        self.biases = [np.random.randn(y, 1) for y in dimensions[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(dimensions[:-1], dimensions[1:])]

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
        training_data: list of tuples (x, y) representing the training inputs and the desired outputs
        n_epochs: number of n_epochs to run the algorithm
        mini_batch_size: size of the mini batch to use when sampling
        eta: the learning rate
        test_data: optional parameter; if provided, network will be evaluated against the test data 
                   after each epoch and partial progress printed out.
    """
    def SGD(self, training_data, n_epochs, mini_batch_size, eta, test_data=None):
        n_test = 0
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in xrange(n_epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j + 1, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    """
    Update the network's weights and biases by applying gradient descent using backpropagation.
    Args:
        mini_batch: list of tuples (x, y) on which gradient descent if performed
        eta: the learning rate
    """
    def update_mini_batch(self, mini_batch, eta):
        # Initialize gradients
        gradB = [np.zeros(b.shape) for b in self.biases]
        gradW = [np.zeros(w.shape) for w in self.weights]

        # Calculate gradients using backpropagation
        for x, y in mini_batch:
            delta_gradB, delta_gradW = self.backprop(x, y)
            gradB = [nb + dnb for nb, dnb in zip(gradB, delta_gradB)]
            gradW = [nw + dnw for nw, dnw in zip(gradW, delta_gradW)]

        # Update the weights and biases
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, gradW)]
        self.biases = [b - (eta / len(mini_batch)) * nb
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
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        gradB[-1] = delta
        gradW[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            gradB[-l] = delta
            gradW[-l] = np.dot(delta, activations[-l - 1].transpose())

        return gradB, gradW

    """
    Evaluates the network on the test data
    Args:
        test_data: test data to evaluate on
    Returns: the number of test inputs for which the neural network outputs the correct result
    """
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
    # The sigmoid function.
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    # The derivative of the sigmoid function.
    return sigmoid(z) * (1 - sigmoid(z))
