import numpy as np


########################## MOUDLE ########################## 

class Module():

    def __call__(self, *args):
        self.out = self.forward(*args)
        return self.out

    def forward(self):
        raise NotImplementedError()

    def backward(self):
        return NotImplementedError()


########################## LINEAR ########################## 

class Linear(Module):
    
    def __init__(self, n_in, n_out, trainable=True):
        self.trainable = trainable
        self.weights = np.random.randn(n_in, n_out) 
        self.bias = np.zeros(n_out)
    
    def forward(self, x):
        self.inp = x
        return x @ self.weights + self.bias
    
    def backward(self, out_grad):
        self.weights_grad = self.inp.T @ out_grad
        self.bias_grad = out_grad.sum(axis=0)
        return out_grad @ self.weights.T

    def update(self, lr):
        self.weights -= lr * self.weights_grad
        self.bias -= lr * self.bias_grad


########################## ACTIVATIONS ########################## 

class Relu(Module):
    
    def forward(self, x):
        self.inp = x
        return np.maximum(self.inp, 0)
    
    def backward(self, out_grad):
        return (self.inp > 0).astype(float) * out_grad


class Sigmoid(Module):

    def forward(self, x):
        self.inp = x
        return 1 / (1 + np.exp(-self.inp))

    def backward(self, out_grad):
        return self.out * (1 - self.out) * out_grad


class Tanh(Module):

    def forward(self, x):
        self.inp = x
        return (np.exp(self.inp) - np.exp(-self.inp)) / (np.exp(self.inp) + np.exp(-self.inp))

    def backward(self, out_grad):
        return (1 - self.out ** 2) * out_grad


########################## LOSS ########################## 

class MSE(Module): 
    
    def forward(self, pred, targ):
        self.pred = pred
        self.targ = targ
        return ((self.pred - self.targ)**2).mean()
    
    def backward(self):
        return 2 * (self.pred - self.targ) / self.targ.shape[0]


class Cross_Entropy(Module): 

    def forward(self, pred, targ):
        pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
        self.probs = pred / np.sum(pred, axis=1, keepdims=True)
        self.targ = targ
        return -np.log(self.probs[range(self.targ.shape[0]), self.targ].mean())

    def backward(self):
        self.probs[range(self.targ.shape[0]), self.targ] -= 1
        return self.probs / self.targ.shape[0]