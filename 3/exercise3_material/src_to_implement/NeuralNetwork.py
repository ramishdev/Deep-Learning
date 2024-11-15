import copy
import pickle

class NeuralNetwork:
    def __init__(self, optimizer,weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    @property
    def phase(self):
        return all(layer.testing_phase for layer in self.layers)
 
    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        reg_loss = 0
        for layer in self.layers:
            layer.testing_phase = False
            input_tensor = layer.forward(input_tensor)
            if self.optimizer.regularizer is not None:
                reg_loss += self.optimizer.regularizer.norm(layer.weights)
        return self.loss_layer.forward(input_tensor, label_tensor) + reg_loss
 
    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = 'train'
        for _ in range(iterations):
            loss_value = self.forward()
            self.loss.append(loss_value)
            self.backward()

    def test(self, input_tensor):
        self.phase = 'test'
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)
        return input_tensor