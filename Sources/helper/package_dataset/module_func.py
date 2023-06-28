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


def scipy_minimize(method, manual_save_history=False, is_least_squares=False):
    def minimize_func(f_real, X, Y, x0, epsilon, max_iter):
        def f(w):
            return mse_loss_norm(X, Y, w, f_real)

        def fun_der(x):
            return Jacobian(lambda xx: f(xx))(x).ravel()

        def fun_hess(x):
            return Hessian(lambda xx: f(xx))(x)

        if not manual_save_history:
            return scipy.optimize.minimize(f, x0, method=method, options={'maxiter' : max_iter})['allvecs']

        def f_with_count(w):
            f_with_count.num_iterations += 1
            return f(w)

        f_with_count.num_iterations = 0

        if is_least_squares is True:
            res = scipy.optimize.least_squares(f_with_count, x0, method=method)
        else:
            if method == 'dogleg':
                res = scipy.optimize.minimize(f_with_count, x0, method=method,
                         jac=fun_der, hess=fun_hess)
            else:
                # bounds = [(-50, 50), (-50, 50)]
                res = scipy.optimize.minimize(f_with_count, x0, method=method, options={'maxiter': max_iter})

        res_w = np.array(res.x, dtype=float)

        return res_w, f_with_count.num_iterations

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
        'scipy-least_squares-dog-box': scipy_minimize('dogbox', True, True),
        'scipy-least_squares-trf': scipy_minimize('trf', True, True),
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
