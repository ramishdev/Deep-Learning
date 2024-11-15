import numpy as np

class Optimizer(object):
    def __init__(self):
        self.regularizer = None
    
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        prev = weight_tensor if not isinstance(weight_tensor, np.ndarray) else weight_tensor.copy()
        weight_tensor -= self.learning_rate * gradient_tensor
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(prev)
        return weight_tensor
class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)
        prev = weight_tensor if not isinstance(weight_tensor, np.ndarray) else weight_tensor.copy()
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(prev)
        return weight_tensor + self.velocity


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-8
        self.m_t = None
        self.v_t = None
        self.t = 0  

    def calculate_update(self, weight_tensor, gradient_tensor):
        prev = weight_tensor if not isinstance(weight_tensor, np.ndarray) else weight_tensor.copy()
        if self.m_t is None:
            self.m_t = np.zeros_like(weight_tensor)
            self.v_t = np.zeros_like(weight_tensor)

        self.t += 1 

        self.m_t = self.mu * self.m_t + (1 - self.mu) * gradient_tensor
        self.v_t = self.rho * self.v_t + (1 - self.rho) * (gradient_tensor ** 2)

        m_hat = self.m_t / (1 - self.mu ** self.t)
        v_hat = self.v_t / (1 - self.rho ** self.t)
        if self.regularizer is not None:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(prev)
        return weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
