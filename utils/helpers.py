import numpy as np

def accuracy(preds, yb):
    return (preds.argmax(axis=1) == yb).mean()


def report(loss, preds, yb):
    print(f'{loss:.2f}, {accuracy(preds, yb):.2f}')