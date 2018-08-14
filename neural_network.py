import os
import numpy as np


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.lr = learning_rate

    # train the neural network
    def train(self):
        pass

    # query the neural network
    def query(self):
        pass


if __name__ == '__main__':
    learning_rate = 0.03
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    # link weight matrices , with and who
    self.wih = (np.random.rand(self.hnodes, self.inodes) - .5)
    self.who = (np.random.rand(self.onodes, self.hnodes) - .5)

   	self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
   	self.who = np.random.normal(0.0, pow(self.onodes, -0.5),(self.onodes, self.hnodes))
   	
    np.random.rand(3, 3) - 0.5

    # n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # learning rate is 0.3
