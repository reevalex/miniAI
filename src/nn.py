from typing import Any
import numpy as np


########################## MOUDLE ########################## 

class Module():
    def __init__(self):
        self._modules = []

    def __setattr__(self, k, v):
        if isinstance(v, str): 
            self._modules.append(v)
        super().__setattr__(k, v)

    def __repr__(self):
        repr_str = [f'{self.__class__.__name__}(']
        repr_str.extend([f'  ({i}): {v}' for i, v in enumerate(self._modules)])
        return '\n'.join(repr_str) + '\n)'

    def __call__(self, *args):
        self._out = self.forward(*args)
        return self._out
    
    def add_module(self, v):
        self._modules.append(v)

    def forward(self):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()
    
    def parameters(self):
        return []
    

class Sequential(Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = [*layers]
        for l in self.layers:
            self.add_module(l)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


########################## LINEAR ########################## 

class Linear(Module):
    
    def __init__(self, n_inp, n_out, bias=True):
        super().__init__()
        self.weights = np.random.randn(n_inp, n_out) * np.sqrt(2 / n_inp)
        self.bias = np.zeros(n_out) if bias else None

    def __repr__(self):
        return f'{self.__class__.__name__}(in_features={self.weights.shape[0]}, out_features={self.weights.shape[1]}, bias={self.bias is not None})'

    def forward(self, x):
        self._inp = x
        return self._inp @ self.weights + self.bias
    
    def backward(self, out_grad):
        self._weights_grad = self._inp.T @ out_grad
        self._bias_grad = out_grad.sum(axis=0)
        return out_grad @ self.weights.T

    def update(self, lr):
        self.weights -= lr * self._weights_grad
        self.bias -= lr * self._bias_grad


########################## ACTIVATIONS ########################## 

class ReLU(Module):
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def forward(self, x):
        self._inp = x
        return np.maximum(self._inp, 0)
    
    def backward(self, out_grad):
        return (self._inp > 0).astype(float) * out_grad


class Sigmoid(Module):

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def forward(self, x):
        self._inp = x
        return 1 / (1 + np.exp(-self._inp))

    def backward(self, out_grad):
        return self._out * (1 - self._out) * out_grad


class Tanh(Module):

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def forward(self, x):
        self._inp = x
        return (np.exp(self._inp) - np.exp(-self._inp)) / (np.exp(self._inp) + np.exp(-self._inp))

    def backward(self, out_grad):
        return (1 - self._out ** 2) * out_grad


########################## LOSS ########################## 

class MSE(Module): 
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def forward(self, preds, targs):
        self._preds = preds
        self._targs = targs
        return ((self._preds - self._targs)**2).mean()
    
    def backward(self):
        return 2 * (self._preds - self._targs) / self._targs.shape[0]


class Cross_Entropy(Module): 
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def forward(self, preds, targs):
        self._targs = targs
        preds_max = np.max(preds, axis=1, keepdims=True)         
        self._log_probs = preds_max + np.log(np.exp(preds - preds_max).sum(axis=1, keepdims=True))
        self._log_probs = preds - self._log_probs
        return -self._log_probs[range(self._targs.shape[0]), self._targs].mean()

    def backward(self):
        probs = np.exp(self._log_probs)
        probs[range(self._targs.shape[0]), self._targs] -= 1
        return probs / self._targs.shape[0] 
