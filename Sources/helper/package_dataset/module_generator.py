import torch
import numpy as np
import random


class Generator:
    @staticmethod
    def generate_2d_fast(f, dots_count, dist, density, variance, weights):
        X = np.linspace(-dist, dist, density)
        Y = np.array([f(x, weights) for x in X])

        Dataset_X = np.random.rand(dots_count, 1) * 2 * dist - dist
        Dataset_Y = np.array([(f(x, weights) + random.uniform(-1, 1) * variance) for x in Dataset_X])

        return X, Y, Dataset_X, Dataset_Y

    @staticmethod
    def generate_2d(X, Y, dots_count, radius):
        dataset = []

        x_min = min(X) - radius
        y_min = min(Y) - radius
        x_max = max(X) + radius
        y_max = max(Y) + radius

        method = 'cpu'
        if torch.cuda:
            if torch.cuda.is_available():
                method = 'cuda'

        # print(method)
        device = torch.device(method)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

        while len(dataset) < dots_count:
            x_rand = torch.empty(1).uniform_(x_min, x_max).to(device)
            y_rand = torch.empty(1).uniform_(y_min, y_max).to(device)

            within_radius = (x_rand - X_tensor) ** 2 + (y_rand - Y_tensor) ** 2 <= radius ** 2
            if torch.any(within_radius):
                dataset.append([x_rand.item(), y_rand.item()])

        return np.array(dataset)
    