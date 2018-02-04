import mnist_data_loader
# import simpleNetwork
#
# training_data, validation_data, test_data = mnist_data_loader.load_data_wrapper()
#
# net = simpleNetwork.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 0.5, 100, validation_data, True, True, True, True)

import theanoNetwork
from theanoNetwork import Network
from theanoNetwork import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = mnist_data_loader.load_data_shared()
mini_batch_size = 10
net = Network([FullyConnectedLayer(nIn=784, nOut=100),
               SoftmaxLayer(nIn=100, nOut=10)],
              mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
