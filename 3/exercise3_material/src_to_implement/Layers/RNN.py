import numpy as np
import copy
from Layers import Base
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH

class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self._memorize = False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc_hidden = FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_output = FullyConnected(hidden_size, output_size)
        self._gradient_weights = np.zeros((self.hidden_size + self.input_size + 1, self.hidden_size))
        self.weights = []
        self.tanh_activation = TanH()
        self.hidden_state = None
        self.previous_hidden_state = None
        self.batch_size = None
        self.optimizer = None
        self.hidden_memory = []

    def forward(self, input_tensor):
        self.batch_size = input_tensor.shape[0]
        if self._memorize and self.hidden_state is not None:
            self.hidden_state[0] = self.previous_hidden_state
        else:
            self.hidden_state = np.zeros((self.batch_size + 1, self.hidden_size))
        output_tensor = np.zeros((self.batch_size, self.output_size))

        for t in range(self.batch_size):
            combined_input = np.hstack([self.hidden_state[t][np.newaxis, :], input_tensor[t][np.newaxis, :]])
            self.hidden_memory.append(combined_input)
            hidden_input = self.fc_hidden.forward(combined_input)
            self.hidden_state[t + 1] = self.tanh_activation.forward(hidden_input)
            output_tensor[t] = self.fc_output.forward(self.hidden_state[t + 1][np.newaxis, :])
        
        self.previous_hidden_state = self.hidden_state[-1]
        self.input_tensor = input_tensor

        return output_tensor

    def backward(self, error_tensor):
        output_error = np.zeros((self.batch_size, self.input_size))
        grad_tanh = 1 - self.hidden_state[1:] ** 2
        hidden_error = np.zeros((1, self.hidden_size))

        for t in reversed(range(self.batch_size)):
            output_gradient = self.fc_output.backward(error_tensor[t][np.newaxis, :])
            grad_hidden = hidden_error + output_gradient
            grad_hidden_state = grad_tanh[t] * grad_hidden
            combined_gradient = self.fc_hidden.backward(grad_hidden_state)
            hidden_error = combined_gradient[:, :self.hidden_size]
            input_gradient = combined_gradient[:, self.hidden_size:]
            output_error[t] = input_gradient

        if self.optimizer is not None:
            self.fc_output.weights = self.optimizer.calculate_update(self.fc_output.weights, self.fc_output.gradient_weights)
            self.fc_hidden.weights = self.optimizer.calculate_update(self.fc_hidden.weights, self.fc_hidden.gradient_weights)

        return output_error

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_output.initialize(weights_initializer, bias_initializer)
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self._weights = self.fc_hidden.weights

    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
