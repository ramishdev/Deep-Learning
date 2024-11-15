import numpy as np
from scipy import signal
from Layers.Base import BaseLayer
from scipy.signal import correlate2d, convolve2d
import copy

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        if type(stride_shape) == int:
            stride_shape = (stride_shape,stride_shape)
        elif len(stride_shape) == 1:
            stride_shape = (stride_shape[0], stride_shape[0])
        self.stride_shape = stride_shape
        self.is_conv2d = (len(convolution_shape) == 3)
        self.weights = np.random.uniform(0,1,(num_kernels, *convolution_shape))
        if self.is_conv2d:
            self.convolution_shape = convolution_shape
        else:
            self.convolution_shape = (*convolution_shape, 1)
            self.weights = self.weights[:, :, :, np.newaxis]
        self.num_kernels = num_kernels
        self._optimizer = None
        self.bias = np.random.uniform(0,1,(num_kernels,))
        self.gradient_weights = None
        self.gradient_bias = None

    def forward(self, input_tensor):
        def pad_input(input_tensor, conv_shape):
            padded_image = np.zeros(
                (input_tensor.shape[0], input_tensor.shape[1],
                input_tensor.shape[2] + conv_shape[1] - 1,
                input_tensor.shape[3] + conv_shape[2] - 1)
            )
            is_shape1_even = int(conv_shape[1] % 2 == 0)
            is_shape2_even = int(conv_shape[2] % 2 == 0)

            if conv_shape[1] // 2 == 0 and conv_shape[2] // 2 == 0:
                return input_tensor
            else:
                padded_image[:, :, (conv_shape[1] // 2):-(conv_shape[1] // 2) + is_shape1_even, (conv_shape[2] // 2):-(conv_shape[2] // 2) + is_shape2_even] = input_tensor
                return padded_image

        def compute_output_dimensions(padded_shape, conv_shape, stride_shape):
            h = np.ceil((padded_shape[2] - conv_shape[1] + 1) / stride_shape[0])
            w = np.ceil((padded_shape[3] - conv_shape[2] + 1) / stride_shape[1])
            return int(h), int(w)

        def convolve_and_add_bias(input_slice, kernel, bias):
            return np.sum(input_slice * kernel) + bias

        self.input_tensor = input_tensor
        if input_tensor.ndim == 3:
            input_tensor = input_tensor[:, :, :, np.newaxis]

        padded_image = pad_input(input_tensor, self.convolution_shape)
        input_tensor = padded_image
        
        h, w = compute_output_dimensions(padded_image.shape, self.convolution_shape, self.stride_shape)

        output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, h, w))
        self.output_shape = output_tensor.shape

        for b in range(input_tensor.shape[0]):
            for k in range(self.num_kernels):
                for i in range(h):
                    for j in range(w):
                        h_start = i * self.stride_shape[0]
                        w_start = j * self.stride_shape[1]
                        h_end = h_start + self.convolution_shape[1]
                        w_end = w_start + self.convolution_shape[2]
                        
                        if h_end <= input_tensor.shape[2] and w_end <= input_tensor.shape[3]:
                            input_slice = input_tensor[b, :, h_start:h_end, w_start:w_end]
                            output_tensor[b, k, i, j] = convolve_and_add_bias(input_slice, self.weights[k, :, :, :], self.bias[k])
                        else:
                            output_tensor[b, k, i, j] = 0

        if not self.is_conv2d:
            output_tensor = output_tensor.squeeze(axis=3)

        return output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weights = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)
        
    def backward(self, error_tensor):
        def reshape_error_tensor(error_tensor):
            return error_tensor.reshape(self.output_shape)

        def prepare_for_upsampling():
            up_error_T = np.zeros((self.input_tensor.shape[0], self.num_kernels, *self.input_tensor.shape[2:]))
            return up_error_T

        def upsample_error_tensor(error_tensor, up_error_T):
            for batch in range(up_error_T.shape[0]):
                for kernel in range(up_error_T.shape[1]):
                    for h in range(self.error_T.shape[2]):
                        for w in range(self.error_T.shape[3]):
                            h_start = h * self.stride_shape[0]
                            w_start = w * self.stride_shape[1]
                            up_error_T[batch, kernel, h_start, w_start] = error_tensor[batch, kernel, h, w]
            return up_error_T

        def initialize_gradients():
            gradient_bias = np.zeros(self.num_kernels)
            gradient_weights = np.zeros(self.weights.shape)
            return gradient_bias, gradient_weights

        def calculate_gradients(upsampled_delta):
            delta_input = np.zeros(self.input_tensor.shape)
            padded_input = np.zeros((*self.input_tensor.shape[:2], self.input_tensor.shape[2] + self.convolution_shape[1] - 1,
                                    self.input_tensor.shape[3] + self.convolution_shape[2] - 1))

            padding_height = int(np.floor(self.convolution_shape[2] / 2))
            padding_width = int(np.floor(self.convolution_shape[1] / 2))

            for batch in range(upsampled_delta.shape[0]):
                for kernel in range(upsampled_delta.shape[1]):
                    self.gradient_bias[kernel] += np.sum(error_tensor[batch, kernel, :])

                    for channel in range(self.input_tensor.shape[1]):
                        delta_input[batch, channel, :] += convolve2d(upsampled_delta[batch, kernel, :], self.weights[kernel, channel, :], 'same')

                for channel in range(self.input_tensor.shape[1]):
                    for h in range(padded_input.shape[2]):
                        for w in range(padded_input.shape[3]):
                            if (h > padding_width - 1) and (h < self.input_tensor.shape[2] + padding_width):
                                if (w > padding_height - 1) and (w < self.input_tensor.shape[3] + padding_height):
                                    padded_input[batch, channel, h, w] = self.input_tensor[batch, channel, h - padding_width, w - padding_height]

                for kernel in range(self.num_kernels):
                    for channel in range(self.input_tensor.shape[1]):
                        self.gradient_weights[kernel, channel, :] += correlate2d(padded_input[batch, channel, :], upsampled_delta[batch, kernel, :], 'valid')

            return delta_input, padded_input

        def update_parameters():
            if self._optimizer is not None:
                self.weights = self._optimizer.weights.calculate_update(self.weights, self.gradient_weights)
                self.bias = self._optimizer.bias.calculate_update(self.bias, self.gradient_bias)

        self.error_T = reshape_error_tensor(error_tensor)
        if not self.is_conv2d:
            self.input_tensor = self.input_tensor[:, :, :, np.newaxis]

        self.up_error_T = prepare_for_upsampling()
        self.up_error_T = upsample_error_tensor(self.error_T, self.up_error_T)

        self.gradient_bias, self.gradient_weights = initialize_gradients()
        return_tensor, self.de_padded = calculate_gradients(self.up_error_T)

        update_parameters()

        if not self.is_conv2d:
            return_tensor = return_tensor.squeeze(axis=3)

        return return_tensor

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)