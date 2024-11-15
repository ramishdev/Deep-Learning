import numpy as np
from Layers import Base, Helpers
import copy
class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True
        self.epsilon = 1e-9
        self.initialize()
        self.moving_mean = None
        self.moving_variance = None
        self.momentum = 0.9
        self._optimizer = None

    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.gamma = np.ones(self.channels)
        self.beta = np.zeros(self.channels)

    def reformat(self, tensor):
        if tensor.ndim == 4:
            self.reformat_shape = tensor.shape
            B, H, M, N = tensor.shape
            tensor = tensor.reshape(B, H, M * N).transpose(0, 2, 1).reshape(B * M * N, H)
            return tensor
        else:
            B, H, M, N = self.reformat_shape
            tensor = tensor.reshape(B, M * N, H).transpose(0, 2, 1).reshape(B, H, M, N)
            return tensor

    
    def forward(self, input_tensor):
        isCheckTrue = input_tensor.ndim == 4
        if isCheckTrue:
            input_tensor = self.reformat(input_tensor)
        self.input_tensor = input_tensor

        if self.testing_phase:
            self.mean = self.moving_mean
            self.variance = self.moving_variance
        else:
            self.mean = np.mean(self.input_tensor, axis=0)
            self.variance = np.var(self.input_tensor, axis=0)
            if self.moving_mean is None:
                self.moving_mean = self.mean
                self.moving_variance = self.variance
            else:
                self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.mean
                self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * self.variance

        self.normalized_input = (self.input_tensor - self.mean) / np.sqrt(self.variance + self.epsilon)
        output = self.gamma * self.normalized_input + self.beta
        if isCheckTrue:
            output = self.reformat(output)
        return output

    def backward(self, error_tensor):
        isCheckTrue = error_tensor.ndim == 4
        if isCheckTrue:
            error_tensor = self.reformat(error_tensor)
        
        gradient_beta = np.sum(error_tensor, axis=0)
        gradient_gamma = np.sum(error_tensor * self.normalized_input, axis=0)
        grad_input = Helpers.compute_bn_gradients(error_tensor, self.input_tensor,  self.gamma, self.mean, self.variance)
        if self._optimizer is not None:
            self._optimizer.weight.calculate_update(self.gamma, gradient_gamma)
            self._optimizer.bias.calculate_update(self.beta, gradient_beta)
        if isCheckTrue:
            grad_input = self.reformat(grad_input)
        self.gradient_weights = gradient_gamma
        self.gradient_bias = gradient_beta
        return grad_input
    
    @property
    def weights(self):
        return self.gamma

    @weights.setter
    def weights(self, gamma):
        self.gamma = gamma

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, beta):
        self.beta = beta

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weight = optimizer
        self._optimizer.bias = optimizer