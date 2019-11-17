import mnist_loader
import network
import numpy as np


def get_data(filename, num_labels, delimiter=','):
    train_data = []
    with open(filename) as fin:
        for line in fin:
            line_arr = np.array(list(map(float, line.strip().split(delimiter))))
            x, y = line_arr[:-num_labels], line_arr[-num_labels:]
            x = x.reshape(len(x), 1)
            y = y.reshape(len(y), 1)
            train_data.append((x, y))
    return train_data


def train_images():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, epochs=3, mini_batch_size=5, eta=3.0)


def train_2dim():
    training_data = get_data("data/data2.txt", num_labels=1)
    net = network.Network([2, 5, 1])
    num_epochs, batch_size, learn_rate = 90, 10, 3.0
    net.SGD(training_data, num_epochs, batch_size, learn_rate)  # works
    print("output at 0, 0:" + str(net.feedforward(np.array([0., 0.]).reshape(2, 1))))
    print("output at 10, 10:" + str(net.feedforward(np.array([10., 10.]).reshape(2, 1))))
    print("num_epochs, batch_size, learn_rate = {0}, {1}, {2}".format(num_epochs, batch_size, learn_rate))
    for b, w in zip(net.biases, net.weights):
        print(np.concatenate((b, w), 1))


if __name__ == "__main__":
    train_2dim()
