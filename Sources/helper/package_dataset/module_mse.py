import numpy as np
import torch


# Определяем функцию потерь
def mse_loss(f, w, x_train, y_train):
    y_pred = f(x_train, w)
    mse = np.mean((y_train - y_pred) ** 2)
    return mse


def mse_loss_torch(dataset_X, dataset_Y, w, f):
    y_pred = f(dataset_X, w)
    mse = torch.mean((dataset_Y - y_pred) ** 2)
    return mse


def mse_loss_grad(f, w, x_train, y_train, batch_size, delta=1e-6):
    # Choose n random data points from the training set without replacement
    indices = np.random.choice(x_train.shape[0], batch_size, replace=False)
    X_batch = x_train[indices, :]
    y_batch = y_train[indices]
    n_features = len(w)
    grad = np.zeros(n_features, dtype=float)

    for i in range(n_features):
        w[i] += delta
        f_plus = mse_loss(f, w, X_batch, y_batch)
        w[i] -= delta
        w[i] -= delta
        f_minus = mse_loss(f, w, X_batch, y_batch)
        w[i] += delta
        grad[i] = np.divide(f_plus - f_minus, 2 * delta)

    return grad


def constant_lr_scheduling(epoch, initial_lr):
    return initial_lr
