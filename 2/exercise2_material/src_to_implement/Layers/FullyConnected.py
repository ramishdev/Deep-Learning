import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self._optimizer = None
        self._gradient_weights = None
        self.input_tensor = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.vstack([self.weights, bias])

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        input_with_bias = np.hstack([input_tensor, np.ones((batch_size, 1))])
        return np.dot(input_with_bias, self.weights)

    def backward(self, error_tensor):
        batch_size = self.input_tensor.shape[0]
        input_with_bias = np.hstack([self.input_tensor, np.ones((batch_size, 1))])
        self._gradient_weights = np.dot(input_with_bias.T, error_tensor)
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        return np.dot(error_tensor, self.weights[:-1, :].T)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights
