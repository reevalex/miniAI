import argparse
import numpy as np

from sklearn import datasets
from utils import helpers
from tqdm import trange
from src import nn


class Model(nn.Module):

    def __init__(self, n_inp, n_hidden, n_out):
        super().__init__()
        self.layers = [nn.Linear(n_inp, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_out)]
        self.loss = nn.Cross_Entropy()

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def backward(self):
        out_grad = self.loss.backward()
        for l in reversed(self.layers):
            out_grad = l.backward(out_grad)

    def step(self, lr):
        for l in self.layers:
            if hasattr(l, 'weights'):
                l.update(lr)


# -----------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a simple numeric neural network')
    parser.add_argument('--lr', '-l', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--epochs', '-o', type=int, default=100, help='Epochs')
    args = parser.parse_args()
    print(vars(args))

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    n, m = x.shape
    c = y.max() + 1
    bs = 10
    model = Model(n_inp=m, n_hidden=32, n_out=c)

    for epoch in trange(args.epochs):
        for i in range(0, n, bs):
            s = slice(i, min(n, i+bs))
            xb, yb = x[s], y[s]
            preds = model(xb)
            loss = model.loss(preds, yb)
            model.backward()
            model.step(args.lr)
        helpers.report(loss, preds, yb)

    
