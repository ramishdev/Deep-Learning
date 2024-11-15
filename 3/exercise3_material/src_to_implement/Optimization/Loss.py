import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.epsilon = np.finfo(float).eps
    
    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        # Compute the cross entropy loss
        loss = -np.sum(label_tensor * np.log(prediction_tensor + self.epsilon))
        return loss
    
    def backward(self, label_tensor):
        # Compute the gradient of the loss with respect to the predictions
        gradient = -label_tensor / (self.prediction_tensor + self.epsilon)
        return gradient
