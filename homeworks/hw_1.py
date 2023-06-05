import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def f(x0, y0):
    x, y = sym.symbols('x y')

    symbolic_value = 2 * (x + 2) ** 4 + (y - 6) ** 4

    numerical_value_ = sym.lambdify([x, y], symbolic_value, 'numpy')
    numerical_value = numerical_value_(x0, y0)

    return numerical_value, symbolic_value


def F(x, y, lim_x, lim_y, r_k=0.01):
    return f(x, y)[0] + r_k * \
        (1 / (x - lim_x[0]) ** 2 + 1 / (y - lim_y[0]) ** 2 + 1 / (x - lim_x[1]) ** 2 + 1 / (y - lim_y[1]) ** 2)


def df(lim_x, lim_y):
    x, y, r_k = sym.symbols('x y r_k')
    df_dx_ = sym.diff(
        f(x, y)[1] + r_k *
        (1 / (x - lim_x[0]) ** 2 + 1 / (y - lim_y[0]) ** 2 + 1 / (x - lim_x[1]) ** 2 + 1 / (y - lim_y[1]) ** 2), x)
    df_dy_ = sym.diff(
        f(x, y)[1] + r_k *
        (1 / (x - lim_x[0]) ** 2 + 1 / (y - lim_y[0]) ** 2 + 1 / (x - lim_x[1]) ** 2 + 1 / (y - lim_y[1]) ** 2), y)
    # print(f'Частные производные:\n'
    #       f'df_dx = {df_dx_}\n'
    #       f'df_dy = {df_dy_}\n')

    return df_dx_, df_dy_


def subs_diff(x0, y0, a, df_dx_, df_dy_):
    x, y, r_k = sym.symbols('x y r_k')

    df_dx = float(df_dx_.subs([(x, x0), (r_k, a)]).evalf())
    df_dy = float(df_dy_.subs([(y, y0), (r_k, a)]).evalf())

    return df_dx, df_dy


def calc_fun(alpha_0, x_0, y_0, dx, dy, lim_x, lim_y, r_k):
    x_0 = x_0 - alpha_0 * dx
    y_0 = y_0 - alpha_0 * dy
    val_fun = F(x_0, y_0, lim_x, lim_y, r_k)

    return val_fun


def golden_section(alpha, x_0, y_0, dx, dy, lim_x, lim_y, r_k):
    # minimum = optimaze.golden(f,)
    lmb = (np.sqrt(5) + 1) / 2  # золотое сечение
    delta = 0.0001  # требуемая точность
    a = 0
    b = alpha

    # ax_alpha = np.linspace(a, b)
    # ax_fun = [calc_fun(a, x_0, y_0, dx, dy, lim_x, lim_y, r_k) for a in ax_alpha]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.plot(ax_alpha, ax_fun)
    # ax.set_xlabel('Alphas')
    # ax.set_ylabel('Fun')

    k = 1

    while abs(b - a) > delta:
        alpha_1 = b - (b - a) / lmb
        alpha_2 = a + (b - a) / lmb

        i = calc_fun(alpha_1, x_0, y_0, dx, dy, lim_x, lim_y, r_k)
        j = calc_fun(alpha_2, x_0, y_0, dx, dy, lim_x, lim_y, r_k)

        if calc_fun(alpha_1, x_0, y_0, dx, dy, lim_x, lim_y, r_k) > \
                calc_fun(alpha_2, x_0, y_0, dx, dy, lim_x, lim_y, r_k):
            a = alpha_1
            z = a
        else:
            b = alpha_2
            z = b

    #     ax.scatter(z, calc_fun(z, x_0, y_0, dx, dy, lim_x, lim_y, r_k))
    #     ax.text(z, calc_fun(z, x_0, y_0, dx, dy, lim_x, lim_y, r_k) + 1, f'$a({k})$', fontsize=8)
    #     k += 1
    #
    # plt.show()

    new_alpha = (a + b) / 2

    return new_alpha


def gradient_descent(r_k, x_0, y_0, alpha_0, lim_x, lim_y):
    df_dx, df_dy = df(lim_x, lim_y)
    x_0_list, y_0_list, func_list = [x_0], [y_0], [f(x_0, y_0)[0]]
    grad_1 = 100
    iterations = 1

    while grad_1 > 0.001:
        alpha = alpha_0

        # рассчитаем значение частных производных
        dx, dy = subs_diff(x_0, y_0, r_k, df_dx, df_dy)

        grad = np.gradient([dx, dy])
        grad_1 = np.sqrt(grad[0] ** 2 + grad[1] ** 2)

        alpha = golden_section(alpha, x_0, y_0, dx, dy, lim_x, lim_y, r_k)

        # будем обновлять веса в направлении,
        # обратном направлению градиента, умноженному на шаг сходимости
        x_0 = x_0 - alpha * dx
        y_0 = y_0 - alpha * dy

        # будем добавлять текущие веса в соответствующие списки
        x_0_list.append(x_0)
        y_0_list.append(y_0)

        # и рассчитывать и добавлять в список текущий уровень ошибки
        func_list.append(f(x_0, y_0)[0])
        iterations += 1

    print(f'Ответ:\n'
          f' x = {x_0}\n'
          f' y = {y_0}\n'
          f' f(x,y) = {f(x_0, y_0)[0]}\n'
          f' |grad| = {grad_1}\n'
          f' iterations = {iterations}')

    return x_0_list, y_0_list, func_list


def draw_graph_2(x_list, y_list, func_list, lim_x, lim_y):
    fig = plt.figure(figsize=(14, 12))

    x = np.linspace(lim_x[0], lim_x[1], 1000)
    y = np.linspace(lim_y[0], lim_y[1], 1000)

    x, y = np.meshgrid(x, y)

    fun = f(x, y)[0]

    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(x, y, fun, alpha=0.5, cmap='viridis')

    ax.text(x_list[0], y_list[0], func_list[0], 'A', size=16)
    ax.text(x_list[-1], y_list[-1], func_list[-1], 'B', size=16)

    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_zlabel('f(x, y)', fontsize=15)

    # выведем путь алгоритма оптимизации
    ax.plot(x_list, y_list, func_list, '.-', c='red')

    plt.show()


def draw_interactive_graph(x_list, y_list, func_list, lim_x, lim_y):
    x = np.outer(np.linspace(lim_x[0], lim_x[1], 1000), np.ones(1000))
    y = np.outer(np.linspace(lim_y[0], lim_y[1], 1000), np.ones(1000)).T
    fun = f(x, y)[0]

    surf = go.Surface(
        x=x,
        y=y,
        z=fun,
        opacity=0.7,
    )

    trace = go.Scatter3d(
        x=x_list,
        y=y_list,
        z=func_list,
        mode='lines+markers',
        marker=dict(
            color=np.linspace(0, 50, len(x_list)),  # set color to an array/list of desired values
            colorscale='Viridis',
            size=5, )

    )

    layout = go.Layout(title='3D Scatter plot')
    fig = go.Figure(data=[trace, surf], layout=layout)
    fig.show()


if __name__ == '__main__':
    r_k = 0.01
    x_0 = 1
    y_0 = 0.5
    alpha_0 = 0.0004
    lim_x = [0, 3]
    lim_y = [0, 2]
    x_list, y_list, func_list = gradient_descent(r_k, x_0, y_0, alpha_0, lim_x, lim_y)

    # draw_graph_2(x_list, y_list, func_list, lim_x, lim_y)

    draw_interactive_graph(x_list, y_list, func_list, lim_x, lim_y)
#
