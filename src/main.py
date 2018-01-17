import mnist_data_loader
import mneilsonNetwork

training_data, validation_data, test_data = mnist_data_loader.load_data_wrapper()

net = mneilsonNetwork.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)