import math
from helper.package_dataset.module_mse import *


# ====================================================<< Adam >>====================================================== #

def train_our_adam(
    f,
    x_train,
    y_train,
    eps_minimum,
    x0,
    batch_size=None,
    lr_scheduling_func=constant_lr_scheduling,
    initial_lr=1.0,
    num_iterations=10000,
    apply_min=True,
    apply_value=True,
    eps_Adam=1e-8,
):
    B1 = 0.9
    B2 = 0.99

    x = np.copy(x0)
    points = [x.copy()]
    value = 0.0
    G = eps_Adam
    moment = 0.0

    if batch_size is None:
        batch_size = len(x_train) // 2

    if apply_value:
        value = mse_loss(f, x, x_train, y_train)

    for i in range(1, num_iterations):
        if apply_value:
            if apply_min and value < eps_minimum:
                break
        else:
            if apply_min and mse_loss(f, x, x_train, y_train) < eps_minimum:
                break

        grad_x = mse_loss_grad(f, x, x_train, y_train, batch_size)
        moment = moment*B1 + (1 - B1)*grad_x
        G = G*B2 + (1 - B2) * (grad_x.dot(grad_x))

        moment_more = moment / (1 - B1 ** i)
        G_more = G / (1 - B2 ** i)
        new_x = x - lr_scheduling_func(i, initial_lr) * moment_more / (math.sqrt(G_more + eps_Adam))

        if apply_value:
            new_value = mse_loss(f, new_x, x_train, y_train)
            if new_value < value:
                x = new_x
                value = new_value
        else:
            x = new_x

        points.append(x.copy())

    return points

# ====================================================<< Adam >>====================================================== #