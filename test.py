import numpy as np
import torch
from sklearn.model_selection import train_test_split

class Kfold:
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def split(self, X):
        n = len(X)
        idx = np.arange(n)
        np.random.shuffle(idx)
        for i in range(self.n_splits):
            test = idx[i * n // self.n_splits: (i + 1) * n // self.n_splits]
            train = np.setdiff1d(idx, test)
            yield train, test

# Create data
X = np.arange(10)
y = np.concatenate((np.ones(5), np.zeros(5)))
print(X, y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

kf = Kfold(n_splits=5)
for train_index, test_index in kf.split(X_tensor):
    X_train_tensor, X_test_tensor = X_tensor[train_index], X_tensor[test_index]
    y_train_tensor, y_test_tensor = y_tensor[train_index], y_tensor[test_index]
    
    print(X_train_tensor, X_test_tensor)
    print(y_train_tensor, y_test_tensor)
    