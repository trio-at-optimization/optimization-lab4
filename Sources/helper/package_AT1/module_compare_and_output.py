import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


class file_info_3d:
    def __init__(self, X=None, Y=None, f=None, x0=None):
        self.X = X
        self.Y = Y
        self.Z = np.vectorize(lambda x, y: f(np.array([x, y])))(X, Y)
        self.f = f
        self.x0 = x0


def print_lines_grad(file_info_3d, result, label, nth=1, title='Спуск на графике функции', filename='',
                     filename_extension='.png', dpi=512):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 8))

    list_result_nth = result[0::nth]
    
    if not np.array_equal(list_result_nth[-1], result[-1]):
        list_result_nth = np.vstack([list_result_nth, result[-1]])

    # levels = np.unique(sorted([file_info_3d.f(p) for p in list_result_nth]))
    # levels=(levels if len(levels) > 1 else None)
    cf = ax.contourf(file_info_3d.X, file_info_3d.Y, file_info_3d.Z, antialiased=True, linewidths=1.7)
    fig.colorbar(cf, ax=ax)

    x = list_result_nth[:, 0]
    y = list_result_nth[:, 1]
    ax.plot(x, y, marker='.', markersize=10, markerfacecolor='black', color='coral', label=label, linewidth=1.8)
    print(
        f'{label:15} ==> '
        f'{file_info_3d.f(result[-1]):10f} in [{result[-1][0]:10f}, {result[-1][1]:10f}] in {len(result) - 1} steps.')

    # Добавление заголовка и подписей осей
    if title != '':
        plt.title(title)

    # Добавление легенды
    if len(label) > 0:
        plt.legend(loc='upper left')

    if filename != '':
        plt.savefig(filename + '_lines' + filename_extension, dpi=dpi, bbox_inches=0, transparent=True)

    plt.show()


def minimize_and_output(
        func
        , initial_x
        , x_lin
        , y_lin
        , output_label
        , method_label
        , message
        , constraints=None
        , bounds=None
        , options=None
        , manual_history=False
        , nth=1
    ):
    def get_points_hostory():
        if not manual_history:
            return minimize(func, initial_x, method=method_label, options=options)['allvecs']

        points = []
        
        def callback(x, _=None):
            points.append(x)

        minimize(func, initial_x, method=method_label, constraints=constraints, bounds=bounds, callback=callback)

        return points

    points = get_points_hostory()

    X, Y = np.meshgrid(x_lin, y_lin)
    f_info = file_info_3d(X, Y, func, initial_x)

    print(message)
    print_lines_grad(f_info, np.array(points), output_label, nth=nth)
