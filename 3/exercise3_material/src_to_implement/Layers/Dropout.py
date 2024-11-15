import numpy as np
from Layers import Base

class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        if not self.testing_phase:
            self.mask = (np.random.rand(*input_tensor.shape) < self.probability) / self.probability
            return input_tensor * self.mask
        else:
            return input_tensor


    def backward(self, grad_output):
        return grad_output * self.mask