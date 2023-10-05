import requests
import gzip
import os
import numpy as np

def fetch_mnist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 
                  't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    for file_name in file_names:
        r = requests.get(url_base + file_name, allow_redirects=True)
        with open(os.path.join(folder_path, file_name), 'wb') as f:
            f.write(r.content)

    with gzip.open(os.path.join(folder_path, 'train-images-idx3-ubyte.gz'), 'rb') as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(os.path.join(folder_path, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(os.path.join(folder_path, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(os.path.join(folder_path, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return (train_images, train_labels), (test_images, test_labels)
