import numpy as np
import math
import types
from ..package_lab3 import gauss_newton_fast
from ..package_lab3 import dog_leg
from ..package_lab3 import adam
from ..package_lab3 import lbfgs

from .module_mse import mse_loss_norm

import scipy
from numdifftools import Jacobian, Hessian

def scipy_minimize(method, manual_save_history=False):
    def minimize_func(f_real, X, Y, x0, epsilon, max_iter):
        def f(w):
            return mse_loss_norm(X, Y, w, f_real)

        def fun_der(x):
            return Jacobian(lambda xx: f(xx))(x).ravel()

        def fun_hess(x):
            return Hessian(lambda xx: f(xx))(x)

        def get_scipy_history():
            if not manual_save_history:
                return scipy.optimize.minimize(f, x0, method=method, options={'maxiter' : max_iter})['allvecs']

            x_list = []
            callback = (lambda x: x_list.append(x))
            if method == 'dogleg':
                scipy.optimize.minimize(f, x0, method=method,
                         jac=fun_der, hess=fun_hess, callback=callback)
            else:
                scipy.optimize.minimize(f, x0, method=method, options={'maxiter': max_iter}, callback=callback)

            return x_list

        points = get_scipy_history()
        return points[-1], len(points)

    return minimize_func


def get_func_method(argument):
    switch_dict = {
        'gauss-newton': gauss_newton_fast,
        'dog-leg': dog_leg,
        'adam': adam,
        'l-bfgs': lbfgs,
        'scipy-bfgs': scipy_minimize('BFGS', True),
        'scipy-l-bfgs': scipy_minimize('L-BFGS-B', True),
        'scipy-dog-leg': scipy_minimize('dogleg', True),
    }

    result = switch_dict.get(argument)
    if result is None:
        raise ValueError(f'Unknown method - \"{argument}\"')

    return result


def get_func_research(f_label):
    compiled_function = compile(f_label, "<string>", "exec")
    exec(compiled_function)

    # Создаем объект функции из скомпилированного кода
    return types.FunctionType(compiled_function.co_consts[0], globals())
