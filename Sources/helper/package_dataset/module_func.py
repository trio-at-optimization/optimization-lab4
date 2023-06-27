import types
import numpy as np
import torch


def get_func_research(f_label):
    compiled_function = compile(f_label, "<string>", "exec")
    exec(compiled_function)

    # Создаем объект функции из скомпилированного кода
    return types.FunctionType(compiled_function.co_consts[0], globals())
