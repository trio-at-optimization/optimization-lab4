# Для расчетов
import math

import numpy as np

# ==================================================================================================================== #


def custom_gradient_descent_with_lr_scheduling_and_moment(
        f
        , gradient, x0
        , lr_scheduling_func
        , initial_lr=1.0
        , num_iterations=1000
        , eps=1e-6
        , minimum=0.0
        , apply_min=False
        , apply_value=True
):
    """
    Функция вычисления градиентного спуска с заданной функцией поиска коэффициента обучения

    Аргументы:
    f -- функция
    x0 -- начальная точка
    -----------------------------------------------------------------------------------------
    lr_search_func -- функция поиска оптимального коэффициента обучения (learning rate)
        Аргументы:
        f -- функция
        gradient -- функция градиента
        a -- левая граница интервала
        b -- правая граница интервала
        eps -- точность поиска

        Возвращает:
        x -- точка минимума функции
    -----------------------------------------------------------------------------------------
    eps -- точность поиска
    num_iterations -- количество итераций
    step_size -- размер шага

    Возвращает:
    points -- массив оптимальных на каждом шаге точек
    """

    x = np.copy(x0)
    moment = 0.0
    points = [x.copy()]
    value = 0.0
    if apply_value:
        value = f(x)
    for i in range(1, num_iterations):
        if apply_value:
            if apply_min and abs(value - minimum) < eps:
                break
        else:
            if apply_min and abs(f(x) - minimum) < eps:
                break

        grad_x = gradient(x)
        current_lr = lr_scheduling_func(i, initial_lr)
        moment = moment*0.9 - current_lr * grad_x
        new_x = x + moment

        if apply_value:
            new_value = f(new_x)
            if new_value < value:
                x = new_x
                value = new_value
        else:
            x = new_x

        points.append(x.copy())

    return points

# ==================================================================================================================== #


def custom_gradient_descent_with_lr_scheduling_and_nesterov_moment(
        f
        , gradient, x0
        , lr_scheduling_func
        , initial_lr=1.0
        , num_iterations=1000
        , eps=1e-6
        , minimum=0.0
        , apply_min=False
        , apply_value=True
):
    """
    Функция вычисления градиентного спуска с заданной функцией поиска коэффициента обучения

    Аргументы:
    f -- функция
    x0 -- начальная точка
    -----------------------------------------------------------------------------------------
    lr_search_func -- функция поиска оптимального коэффициента обучения (learning rate)
        Аргументы:
        f -- функция
        gradient -- функция градиента
        a -- левая граница интервала
        b -- правая граница интервала
        eps -- точность поиска

        Возвращает:
        x -- точка минимума функции
    -----------------------------------------------------------------------------------------
    eps -- точность поиска
    num_iterations -- количество итераций
    step_size -- размер шага

    Возвращает:
    points -- массив оптимальных на каждом шаге точек
    """

    x = np.copy(x0)
    moment = 0.0
    points = [x.copy()]
    value = 0.0
    if apply_value:
        value = f(x)
    for i in range(1, num_iterations):
        if apply_value:
            if apply_min and abs(value - minimum) < eps:
                break
        else:
            if apply_min and abs(f(x) - minimum) < eps:
                break

        grad_x = gradient(x + moment*0.9)
        current_lr = lr_scheduling_func(i, initial_lr)
        moment = moment*0.9 - current_lr * grad_x
        new_x = x + moment

        if apply_value:
            new_value = f(new_x)
            if new_value < value:
                x = new_x
                value = new_value
        else:
            x = new_x

        points.append(x.copy())

    return points

# ==================================================================================================================== #

def custom_gradient_descent_with_lr_scheduling_and_adagrad(
        f
        , gradient, x0
        , lr_scheduling_func
        , initial_lr=1.0
        , num_iterations=1000
        , eps=1e-6
        , eps_adagrad = 1e-6
        , minimum=0.0
        , apply_min=False
        , apply_value=True
):
    """
    Функция вычисления градиентного спуска с заданной функцией поиска коэффициента обучения

    Аргументы:
    f -- функция
    x0 -- начальная точка
    -----------------------------------------------------------------------------------------
    lr_search_func -- функция поиска оптимального коэффициента обучения (learning rate)
        Аргументы:
        f -- функция
        gradient -- функция градиента
        a -- левая граница интервала
        b -- правая граница интервала
        eps -- точность поиска

        Возвращает:
        x -- точка минимума функции
    -----------------------------------------------------------------------------------------
    eps -- точность поиска
    num_iterations -- количество итераций
    step_size -- размер шага

    Возвращает:
    points -- массив оптимальных на каждом шаге точек
    """

    x = np.copy(x0)
    points = [x.copy()]
    value = 0.0
    G = eps_adagrad
    if apply_value:
        value = f(x)
    for i in range(1, num_iterations):
        if apply_value:
            if apply_min and abs(value - minimum) < eps:
                break
        else:
            if apply_min and abs(f(x) - minimum) < eps:
                break

        grad_x = gradient(x)
        G = G + grad_x.dot(grad_x)
        new_x = x - grad_x * lr_scheduling_func(i, initial_lr) / (math.sqrt(G + eps_adagrad))

        if apply_value:
            new_value = f(new_x)
            if new_value < value:
                x = new_x
                value = new_value
        else:
            x = new_x
        points.append(x.copy())

    return points

# ==================================================================================================================== #


def custom_gradient_descent_with_lr_scheduling_and_RMSProp (
        f
        , gradient, x0
        , lr_scheduling_func
        , initial_lr=1.0
        , num_iterations=1000
        , eps=1e-6
        , eps_RMSProp = 1e-8
        , minimum=0.0
        , apply_min=False
        , apply_value=True
):
    """
    Функция вычисления градиентного спуска с заданной функцией поиска коэффициента обучения

    Аргументы:
    f -- функция
    x0 -- начальная точка
    -----------------------------------------------------------------------------------------
    lr_search_func -- функция поиска оптимального коэффициента обучения (learning rate)
        Аргументы:
        f -- функция
        gradient -- функция градиента
        a -- левая граница интервала
        b -- правая граница интервала
        eps -- точность поиска

        Возвращает:
        x -- точка минимума функции
    -----------------------------------------------------------------------------------------
    eps -- точность поиска
    num_iterations -- количество итераций
    step_size -- размер шага

    Возвращает:
    points -- массив оптимальных на каждом шаге точек
    """

    x = np.copy(x0)
    points = [x.copy()]
    value = 0.0
    G = eps_RMSProp
    if apply_value:
        value = f(x)
    for i in range(1, num_iterations):
        if apply_value:
            if apply_min and abs(value - minimum) < eps:
                break
        else:
            if apply_min and abs(f(x) - minimum) < eps:
                break

        grad_x = gradient(x)
        G = G*0.9 + (1 - 0.9) * (grad_x.dot(grad_x))
        new_x = x - grad_x * lr_scheduling_func(i, initial_lr) / (math.sqrt(G + eps_RMSProp))

        if apply_value:
            new_value = f(new_x)
            if new_value < value:
                x = new_x
                value = new_value
        else:
            x = new_x
        points.append(x.copy())

    return points

# ==================================================================================================================== #


def custom_gradient_descent_with_lr_scheduling_and_Adam (
        f
        , gradient, x0
        , lr_scheduling_func
        , initial_lr=1.0
        , num_iterations=1000
        , eps=1e-6
        , eps_Adam = 1e-8
        , minimum=0.0
        , apply_min=False
        , apply_value=True
):
    """
    Функция вычисления градиентного спуска с заданной функцией поиска коэффициента обучения

    Аргументы:
    f -- функция
    x0 -- начальная точка
    -----------------------------------------------------------------------------------------
    lr_search_func -- функция поиска оптимального коэффициента обучения (learning rate)
        Аргументы:
        f -- функция
        gradient -- функция градиента
        a -- левая граница интервала
        b -- правая граница интервала
        eps -- точность поиска

        Возвращает:
        x -- точка минимума функции
    -----------------------------------------------------------------------------------------
    eps -- точность поиска
    num_iterations -- количество итераций
    step_size -- размер шага

    Возвращает:
    points -- массив оптимальных на каждом шаге точек
    """
    B1 = 0.9
    B2 = 0.999

    x = np.copy(x0)
    points = [x.copy()]
    value = 0.0
    G = eps_Adam
    moment = 0.0
    if apply_value:
        value = f(x)
    for i in range(1, num_iterations):
        if apply_value:
            if apply_min and abs(value - minimum) < eps:
                break
        else:
            if apply_min and abs(f(x) - minimum) < eps:
                break

        grad_x = gradient(x)
        moment = moment*B1 + (1 - B1)*grad_x
        G = G*B2 + (1 - B2) * (grad_x.dot(grad_x))
        # new_x = x - lr_scheduling_func(i, initial_lr) / (math.sqrt(G + eps_Adam)) * moment

        moment_more = moment / (1 - B1 ** i)
        G_more = G / (1 - B2 ** i)
        new_x = x - lr_scheduling_func(i, initial_lr) * (moment_more) / (math.sqrt(G_more + eps_Adam))

        if apply_value:
            new_value = f(new_x)
            if new_value < value:
                x = new_x
                value = new_value
        else:
            x = new_x
        points.append(x.copy())

    return points

# ==================================================================================================================== #
