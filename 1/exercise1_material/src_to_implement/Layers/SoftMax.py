import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        shift_input = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_shift_input = np.exp(shift_input)
        softmax_output = exp_shift_input / np.sum(exp_shift_input, axis=1, keepdims=True)
        self.output_tensor = softmax_output
        return softmax_output

    def backward(self, error_tensor):
        return self.output_tensor * (error_tensor - np.sum(error_tensor * self.output_tensor, axis=1, keepdims=True))
