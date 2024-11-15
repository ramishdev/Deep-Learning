import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.max_positions = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channels, height, width = input_tensor.shape
        pool_height, pool_width = self.pooling_shape
        stride_height, stride_width = self.stride_shape
        
        out_height = (height - pool_height) // stride_height + 1
        out_width = (width - pool_width) // stride_width + 1
        
        output_tensor = np.zeros((batch_size, channels, out_height, out_width))
        self.max_positions = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * stride_height
                        h_end = h_start + pool_height
                        w_start = j * stride_width
                        w_end = w_start + pool_width
                        
                        pool_region = input_tensor[b, c, h_start:h_end, w_start:w_end]
                        max_value = np.max(pool_region)
                        output_tensor[b, c, i, j] = max_value
                        
                        max_position = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                        self.max_positions[b, c, i, j] = (h_start + max_position[0], w_start + max_position[1])
        
        return output_tensor

    def backward(self, error_tensor):
        batch_size, channels, out_height, out_width = error_tensor.shape
        grad_input = np.zeros_like(self.input_tensor)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h, w = self.max_positions[b, c, i, j]
                        grad_input[b, c, h, w] += error_tensor[b, c, i, j]
        
        return grad_input
