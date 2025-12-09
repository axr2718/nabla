import numpy as np

class Optimizer:
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'weight_grad'):
                layer.weight_grad = None
                layer.bias_grad = None

class SGD(Optimizer):
    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                layer.weight = layer.weight - self.learning_rate * layer.weight_grad

                layer.bias = layer.bias - self.learning_rate * layer.bias_grad

class Adam(Optimizer):
    # Default parameters used by most libraries and papers
    def __init__(self, layers, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(layers, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m_w, self.v_w = {}, {}
        self.m_b, self.v_b = {}, {}

    def step(self):
        self.t += 1
        
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                if layer not in self.m_w:
                    self.m_w[layer] = np.zeros_like(layer.weight)
                    self.v_w[layer] = np.zeros_like(layer.weight)
                    self.m_b[layer] = np.zeros_like(layer.bias)
                    self.v_b[layer] = np.zeros_like(layer.bias)

                grad_w = layer.weight_grad
                grad_b = layer.bias_grad


                self.m_w[layer] = self.beta1 * self.m_w[layer] + (1 - self.beta1) * grad_w
                self.v_w[layer] = self.beta2 * self.v_w[layer] + (1 - self.beta2) * (grad_w ** 2)

                m_hat_w = self.m_w[layer] / (1 - self.beta1 ** self.t)
                v_hat_w = self.v_w[layer] / (1 - self.beta2 ** self.t)

                layer.weight = layer.weight - self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.eps)

                self.m_b[layer] = self.beta1 * self.m_b[layer] + (1 - self.beta1) * grad_b
                self.v_b[layer] = self.beta2 * self.v_b[layer] + (1 - self.beta2) * (grad_b ** 2)

                m_hat_b = self.m_b[layer] / (1 - self.beta1 ** self.t)
                v_hat_b = self.v_b[layer] / (1 - self.beta2 ** self.t)

                layer.bias = layer.bias - self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.eps)