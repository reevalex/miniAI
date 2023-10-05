import os
import argparse
import numpy as np

from tqdm import trange
from src import nn


class Model(): 
    
    def __init__(self, n_inp, n_hidden, n_out):
        self.layers = [nn.Linear(n_inp, n_hidden),    nn.Tanh(), 
                       nn.Linear(n_hidden, n_hidden), nn.Tanh(),
                       nn.Linear(n_hidden, n_out)]
        self.loss_fn = nn.MSE()
    
    def forward(self, x, targ):
        for layer in self.layers:
            x = layer(x)

        loss = self.loss_fn(x, targ)
        pred = x
        return loss, pred
    
    def backward(self):
        out_grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            out_grad = layer.backward(out_grad)

    def step(self, lr):
        for layer in self.layers:
            if hasattr(layer, 'trainable'):
                if layer.trainable:
                    layer.update(lr)


# -----------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a simple numeric neural network')
    parser.add_argument('--lr', '-l', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--epochs', '-o', type=int, default=100, help='Epochs')
    args = parser.parse_args()
    print(vars(args))

    x = np.linspace(-10, 10, 400).reshape(-1, 1)
    y = np.sin(x)
    model = Model(n_inp=1, n_hidden=16, n_out=1)

    for epoch in trange(args.epochs):
        loss, pred = model.forward(x, y)
        model.backward()
        model.step(args.lr)
        
    print(f'Final Loss: {loss:.4f}')

    
