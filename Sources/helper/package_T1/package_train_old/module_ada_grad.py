import math
from helper.package_dataset.module_mse import *


# ==================================================<< Ada_grad >>==================================================== #

def train_our_ada_grad(
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
        eps_ada_grad=1e-6
):
    x = np.copy(x0)
    points = [x.copy()]
    value = 0.0
    G = eps_ada_grad
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
        G = G + grad_x.dot(grad_x)
        new_x = x - grad_x * lr_scheduling_func(i, initial_lr) / (math.sqrt(G + eps_ada_grad))

        if apply_value:
            new_value = mse_loss(f, new_x, x_train, y_train)
            if new_value < value:
                x = new_x
                value = new_value
        else:
            x = new_x

        points.append(x.copy())

    return points

# ==================================================<< Ada_grad >>==================================================== #
