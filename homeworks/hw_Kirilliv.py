import numpy as np
import sympy as sym
import matplotlib.pyplot as plt


def f(x, y):
    return (x - 1) ** 4 + (y + 5) ** 3


def F(x, y, r_k=0.01):
    return (x - 1) ** 4 + (y + 5) ** 3 + r_k * (1 / x ** 2 + 1 / y ** 2 + 1 / (x - 2) ** 2 + 1 / (y - 2) ** 2)


def df():
    x, y, r_k = sym.symbols('x y r_k')
    df_dx = sym.diff(
        (x - 1) ** 4 + (y + 5) ** 3 + r_k * (1 / x ** 2 + 1 / y ** 2 + 1 / (x - 2) ** 2 + 1 / (y - 2) ** 2), x)
    df_dy = sym.diff(
        (x - 1) ** 4 + (y + 5) ** 3 + r_k * (1 / x ** 2 + 1 / y ** 2 + 1 / (x - 2) ** 2 + 1 / (y - 2) ** 2), y)
    print(f'Частные производные:\n'
          f'df_dx = {df_dx}\n'
          f'df_dy = {df_dy}\n')


def calc_fun(alpha_0, x_0, y_0, dx, dy, r_k):
    x_0 = x_0 - alpha_0 * dx
    y_0 = y_0 - alpha_0 * dy
    val_fun = F(r_k, x_0, y_0)

    return val_fun


def golden_section(alpha, x_0, y_0, dx, dy, r_k):
    # minimum = optimaze.golden(f,)
    lmb = (np.sqrt(5) + 1) / 2  # золотое сечение
    delta = 0.0001  # требуемая точность
    a = 0
    b = alpha

    while abs(b - a) > delta:
        alpha_1 = b - (b - a) / lmb
        alpha_2 = a + (b - a) / lmb

        if calc_fun(alpha_1, x_0, y_0, dx, dy, r_k) > calc_fun(alpha_2, x_0, y_0, dx, dy, r_k):
            a = alpha_1
        else:
            b = alpha_2

    new_alpha = (a + b) / 2

    return new_alpha


def gradient_descent(r_k, x_0, y_0, alpha_0):
    df()
    df_dx = lambda x: r_k * (-2 / (x - 2) ** 3 - 2 / x ** 3) + 4 * (x - 1) ** 3
    df_dy = lambda y: r_k * (-2 / (y - 2) ** 3 - 2 / y ** 3) + 3 * (y + 5) ** 2
    x_0_list, y_0_list, func_list = [x_0], [y_0], [f(x_0, y_0)]
    grad_1 = 100
    iterations = 1

    while grad_1 > 0.001:
        alpha = alpha_0

        # рассчитаем значение частных производных
        dx = df_dx(x_0)
        dy = df_dy(y_0)

        grad = np.gradient([dx, dy])
        grad_1 = np.sqrt(grad[0] ** 2 + grad[1] ** 2)

        alpha = golden_section(alpha, x_0, y_0, dx, dy, r_k)
        alpha += alpha

        # будем обновлять веса в направлении,
        # обратном направлению градиента, умноженному на шаг сходимости
        x_0 = x_0 - alpha * dx
        y_0 = y_0 - alpha * dy

        # будем добавлять текущие веса в соответствующие списки
        x_0_list.append(x_0)
        y_0_list.append(y_0)

        # и рассчитывать и добавлять в список текущий уровень ошибки
        func_list.append(f(x_0, y_0))
        iterations += 1

    print(f'Ответ:\n'
          f' x = {x_0}\n'
          f' y = {y_0}\n'
          f' f(x,y) = {f(x_0, y_0)}\n'
          f' |grad| = {grad_1}\n'
          f' iterations = {iterations}')

    return x_0_list, y_0_list, func_list


def draw_graph(x_list, y_list, func_list):
    fig = plt.figure(figsize=(14, 12))

    x = np.linspace(0, 2, 1000)
    y = np.linspace(0, 2, 1000)

    x, y = np.meshgrid(x, y)

    fun = f(x, y)

    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(x, y, fun, alpha=0.5, cmap='viridis')

    ax.text(x_list[0], y_list[0], func_list[0], 'A', size=25)
    ax.text(x_list[-1], y_list[-1], func_list[-1], 'B', size=25)

    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_zlabel('f(x, y)', fontsize=15)

    # выведем путь алгоритма оптимизации
    ax.plot(x_list, y_list, func_list, '.-', c='red')

    plt.show()


if __name__ == '__main__':
    r_k = 0.01
    x_0 = 1.6
    y_0 = 1.6
    alpha_0 = 0.0001
    x_list, y_list, func_list = gradient_descent(r_k, x_0, y_0, alpha_0)

    print(f(0, 0))

    draw_graph(x_list, y_list, func_list)
