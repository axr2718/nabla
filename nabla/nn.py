import numpy as np

class Module:
    def __init__(self):
        self.input = None
        self.output = None
        self.training = True

    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

class Linear(Module):
    def __init__(self, in_features, out_features):
        # For He initialization
        std = np.sqrt(2.0 / in_features)
        self.weight = np.random.randn(in_features, out_features) * std
        self.bias = np.zeros((1, out_features))

        self.weight_grad = None
        self.bias_grad = None

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weight) + self.bias

        return self.output
    
    def backward(self, grad):
        self.weight_grad = np.dot(self.input.T, grad)

        self.bias_grad = np.sum(grad, axis=0, keepdims=True)

        input_grad = np.dot(grad, self.weight.T)

        return input_grad

########################################################################
########################################################################
#                           ACTIVATIONS
########################################################################
########################################################################
class ReLU(Module):
    def forward(self, input):
        self.input = input

        self.output = np.maximum(0, self.input)
        
        return self.output
    
    def backward(self, grad):
        return grad * (self.input > 0).astype(np.float64)

# Decided to remove to make report and experiments shorter
# class Softplus(Module):
#     def __init__(self, beta=1.0):
#         super().__init__()
#         self.beta = beta

#     def forward(self, input):
#         self.input = input

#         self.output = (1 / self.beta) * np.log(1 + np.exp(self.beta * self.input))

#         return self.output
    
#     def backward(self, grad):
#         return grad * (1 / (1 + np.exp(-self.beta * self.input)))
    
class SiLU(Module):
    def forward(self, input):
        self.input = input
        sigmoid = 1 / (1 + np.exp(-self.input))

        self.output = self.input * sigmoid

        return self.output
    
    def backward(self, grad):
        sigmoid = 1 / (1 + np.exp(-self.input))
        return grad * (sigmoid + self.output * (1 - sigmoid))
    
class Sigmoid(Module):
    def forward(self, input):
        self.input = input

        self.output = 1 / (1 + np.exp(-self.input))

        return self.output
    
    def backward(self, grad):
        return grad * (self.output * (1.0 - self.output))
    
class Tanh(Module):
    def forward(self, input):
        self.input = input

        self.output = np.tanh(self.input)

        return self.output
    
    def backward(self, grad):
        return grad * (1 - self.output**2)
    
########################################################################
########################################################################
#                           CRITERION
########################################################################
########################################################################
class MSELoss(Module):
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target

        self.output = np.mean((prediction - target)**2)

        return self.output
    
    def backward(self):
        return 2 * (self.prediction - self.target) / self.prediction.size
    

class CrossEntropyLoss(Module):
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        batch_size = prediction.shape[0]

        exps = np.exp(prediction - np.max(prediction, axis=1, keepdims=True))
        self.probabilities = exps / np.sum(exps, axis=1, keepdims=True)

        self.output = -np.log(self.probabilities[np.arange(batch_size), target])
        self.output = np.sum(self.output) / batch_size

        return self.output
    
    def backward(self):
        batch_size = self.prediction.shape[0]

        grad = self.probabilities.copy()
        grad[np.arange(batch_size), self.target] -= 1

        return grad / batch_size
    
########################################################################
########################################################################
#                           DROPOUT
########################################################################
########################################################################
class Dropout(Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob
        self.mask = None

    def forward(self, input):
        self.input = input

        if not self.training:
            return input
        
        self.mask = np.random.binomial(1, 1 - self.prob, size=input.shape)

        self.output = (self.input * self.mask) / (1 - self.prob)

        return self.output
    
    def backward(self, grad):
        if not self.training:
            return grad
        
        return (grad * self.mask) / (1 - self.prob)

########################################################################
########################################################################
#                           NORMALIZATION
########################################################################
########################################################################
# TODO: BatchNorm
# TODO: RMSNorm

