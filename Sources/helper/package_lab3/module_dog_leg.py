import math
import copy
import numpy as np


def func(f, X, Y, w, j):
    x = X[j]
    value = Y[j]

    return f(x, w) - value


def derivative(f, X, Y, w, i, j, delta=1e-6):
    w1 = np.copy(w)
    w2 = np.copy(w)

    w1[i] -= delta
    w2[i] += delta

    f_plus = func(f, X, Y, w1, j)
    f_minus = func(f, X, Y, w2, j)

    return (f_minus - f_plus) / (2 * delta)


def jacobian(f, X, Y, w, delta):
    row_count = len(X)
    col_count = len(w)

    jac = np.zeros((row_count, col_count))

    for i in range(row_count):
        for j in range(col_count):
            jac[i][j] = derivative(f, X, Y, w, j, i, delta)

    return jac


def dog_leg(f, X, Y, initial_params, max_iter=100, epsilon=2e-2, delta=1e-6, radius=1.5):
    e1 = epsilon  # 1e-12
    e2 = epsilon  # 1e-12
    e3 = epsilon  # 1e-12

    current_params = np.copy(initial_params)

    r = f(X, current_params) - Y
    jac = jacobian(f, X, Y, current_params, delta)
    gradient = jac.T @ r

    if np.linalg.norm(r) <= e3 or np.linalg.norm(gradient) <= e1:
        return current_params, -1

    for iteration in range(max_iter):
        jac = jacobian(f, X, Y, current_params, delta)
        r = f(X, current_params) - Y

        # Gradient computing
        gradient = jac.T @ r

        # Check convergience
        if np.linalg.norm(gradient) <= e1:
            return current_params, iteration + 1
        elif np.linalg.norm(r) <= e3:
            return current_params, iteration + 1

        alpha = np.linalg.norm(gradient) ** 2 / np.linalg.norm(jac @ gradient) ** 2
        stepest_descent = -alpha * gradient
        gauss_newton = -1 * np.linalg.inv(jac.T @ jac) @ jac.T @ r

        beta = 0.0
        dog_leg = np.zeros(len(current_params))

        # Step
        if np.linalg.norm(gauss_newton) <= radius:
            # print('Now GAUSS_NEWTON')
            dog_leg = np.copy(gauss_newton)
        elif alpha * np.linalg.norm(stepest_descent) >= radius:
            dog_leg = (radius / np.linalg.norm(stepest_descent)) * stepest_descent
        else:
            a = alpha * stepest_descent
            b = np.copy(gauss_newton)
            c = a.T @ (b - a)

            if c <= 0:
                beta = (math.sqrt(math.fabs(
                    c * c + np.linalg.norm(b - a, 2) * (radius * radius - np.linalg.norm(a) ** 2))) - c) / (np.linalg.norm(
                    b - a, 2) + e3)
            else:
                beta = (radius * radius - np.linalg.norm(a, 2)) / (
                        math.sqrt(
                            c * c + np.linalg.norm(b - a, 2) * max(0, radius * radius - np.linalg.norm(a, 2))) - c + e3)
            dog_leg = alpha * stepest_descent + (gauss_newton - alpha * stepest_descent) * beta

        if np.linalg.norm(dog_leg) <= e2 * (np.linalg.norm(current_params) + e2):
            return current_params, iteration + 1

        new_params = current_params + dog_leg

        obj_new = f(X, new_params) - Y

        # Recalc roi for radius
        denum = 0.0    
        if np.linalg.norm(gauss_newton) <= radius:
            denum = np.linalg.norm(r, 2) / 2
        elif alpha * np.linalg.norm(stepest_descent) >= radius:
            denum = radius * (2 * alpha * np.linalg.norm(gradient) - radius) / (2.0 * alpha)
        else:
            a = stepest_descent * alpha
            b = copy.copy(gauss_newton)
            c = a.T @ (b - a)

            if c <= 0:
                beta = (math.sqrt(math.fabs(
                    c * c + np.linalg.norm(b - a, 2) * (radius ** 2 - np.linalg.norm(a, 2)))) - c) / (np.linalg.norm(
                    b - a, 2) + e3)
            else:
                beta = (radius * radius - np.linalg.norm(a, 2)) / (
                        math.sqrt(
                            c * c + np.linalg.norm(b - a) ** 2 * max(0, radius * radius - np.linalg.norm(a, 2))) - c + e3)

            denum = alpha * (1 - beta) * (1 - beta) * (np.linalg.norm(gradient) ** 2) / 2.0 + beta * (
                    2.0 - beta) * np.linalg.norm(r, 2) / 2

        roi = (np.linalg.norm(r, 2) / 2 - np.linalg.norm(obj_new, 2) / 2) / (denum + e3)

        # Changing radius
        if roi > 0:
            current_params = np.copy(new_params)
        if roi > 0.75:
            radius = max(radius, 3.0 * np.linalg.norm(dog_leg))
        elif roi < 0.25:
            radius /= 2.0

            if radius <= e2 * (np.linalg.norm(current_params) + e2):
                return current_params, iteration + 1

    return current_params, max_iter + 1
