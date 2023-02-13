import math
import time
import copy
import numpy as np


class Interval:
    def __init__(self, x_start, x_end):
        self.x_start = x_start
        self.x_end = x_end

    def center(self):
        return (self.x_end + self.x_start) / 2

    def length(self):
        return abs(self.x_end - self.x_start)


def f(point):
    return 80 * pow((pow(point[0], 2) - point[1]), 2) + 3 * pow((point[0] - 1), 2) + 110


def f1(x):
    return 80 * pow((pow(x, 2) - x), 2) + 3 * pow((x - 1), 2) + 110


def method_svann(x_start, step):
    k = 0
    x_values = [x_start]
    fun_result_without_step_size = f1(x_start - step)
    fun_result_on_start = f1(x_start)
    fun_result_with_step_size = f1(x_start + step)
    interval = Interval(x_start - step, x_start)
    if fun_result_without_step_size >= fun_result_on_start and fun_result_on_start <= fun_result_with_step_size:
        return interval
    elif fun_result_without_step_size <= fun_result_on_start and fun_result_on_start >= fun_result_with_step_size:
        raise Exception("Интервал не найден, плохое значение: (%f)" % (str(x_start)))
    else:
        delta = float(0.0)
        k += 1
        if fun_result_without_step_size >= fun_result_on_start >= fun_result_with_step_size:
            delta = step
            interval.x_start = x_values[0]
            x_values.insert(k, x_start + step)
        elif fun_result_without_step_size <= fun_result_on_start <= fun_result_with_step_size:
            delta = -step
            interval.x_end = x_values[0]
            x_values.insert(k, x_start - step)
        while True:
            x_values.insert(k + 1, (x_values[k] + pow(2.0, k) * delta))
            if f(x_values[k + 1]) >= f(x_values[k]):
                if delta > 0:
                    interval.x_end = x_values[k + 1]
                elif delta < 0:
                    interval.x_start = x_values[k + 1]
            else:
                if delta > 0:
                    interval.x_start = x_values[k]
                elif delta < 0:
                    interval.x_end = x_values[k]
            if f(x_values[k + 1]) >= f(x_values[k]):
                break
            k += 1
    return interval


def bisection_method(epsilon, interval):
    k = 0
    x_middle = interval.center()
    while True:
        x_left_middle = interval.x_start + interval.length() / 4
        x_right_middle = interval.x_end - interval.length() / 4
        if f1(x_left_middle) < f1(x_middle):
            interval.x_end = x_middle
            x_middle = x_left_middle
        elif f1(x_right_middle) < f1(x_middle):
            interval.x_start = x_middle
            x_middle = x_right_middle
        else:
            interval.x_start = x_left_middle
            interval.x_end = x_right_middle
        k += 1
        if not interval.length() > epsilon:
            break
    return x_middle


def golden_section_method(epsilon, interval):
    k = 0
    phi = (1 + math.sqrt(5.0)) / 2
    while interval.length() > epsilon:
        z = (interval.x_end - (interval.x_end - interval.x_start) / phi)
        y = (interval.x_start + (interval.x_end - interval.x_start) / phi)
        if f1(y) <= f1(z):
            interval.x_start = z
        else:
            interval.x_end = y
        k += 1
    return interval.center()


def fibonacci_method(epsilon, interval):
    k = 0
    n = 3
    fib_arr = [1.0, 1.0, 2.0, 3.0]
    f1 = 2.0
    f2 = 3.0
    while fib_arr[len(fib_arr) - 1] < interval.length() / epsilon:
        fib_arr.append(f1 + f2)
        f1 = f2
        f2 = fib_arr[len(fib_arr) - 1]
        n = n + 1
    for i in range(1, n - 3):
        y = (interval.x_start + fib_arr[n - i - 1] / fib_arr[n - i + 1] * (interval.x_end - interval.x_start))
        z = (interval.x_start + fib_arr[n - i] / fib_arr[n - i + 1] * (interval.x_end - interval.x_start))
        if f1(y) <= f1(z):
            interval.x_end = z
        else:
            interval.x_start = y
        k += 1
    return interval.center()


def best_near_by(point, method):
    c = copy.deepcopy(point)
    for i in range(0, 2):
        interval = method_svann(1, 0.001)

        if method == 0:
            c[i] = bisection_method(0.01, interval)
        elif method == 1:
            c[i] = golden_section_method(0.01, interval)
        elif method == 2:
            c[i] = fibonacci_method(0.01, interval)
    return c


def hook_djeevs(dim, start_point, rho, eps):
    tic = time.perf_counter()
    new_x = start_point.copy()
    x_before = start_point.copy()
    delta = np.zeros(dim)

    for i in range(0, dim):
        if start_point[i] == 0.0:
            delta[i] = rho
        else:
            delta[i] = rho * abs(start_point[i])

    step_length = rho
    k = 0
    f_before = f(new_x)
    while k < maxIterations and eps < step_length:
        k = k + 1
        for i in range(0, dim):
            new_x[i] = x_before[i]

        new_x = best_near_by(new_x, 0)
        new_f = f(new_x)
        keep = True
        while new_f < f_before and keep:
            for i in range(0, dim):
                if new_x[i] <= x_before[i]:
                    delta[i] = - abs(delta[i])
                else:
                    delta[i] = abs(delta[i])
                tmp = x_before[i]
                x_before[i] = new_x[i]
                new_x[i] = new_x[i] + new_x[i] - tmp

            f_before = new_f
            new_x = best_near_by(new_x, 0)
            new_f = f(new_x)

            if f_before <= new_f:
                break
            keep = False

            for i in range(0, dim):
                if 0.5 * abs(delta[i]) < abs(new_x[i] - x_before[i]):
                    keep = True
                    break

        if eps <= step_length and f_before <= new_f:
            step_length = step_length * rho
            for i in range(0, dim):
                delta[i] = delta[i] * rho

    end_point = x_before.copy()
    toc = time.perf_counter()
    print("Метод Хука — Дживса")
    print("Количество итераций: " + str(k))
    print(f"Вычисление заняло {toc - tic:0.10f} секунд")
    print("f([" + str(end_point[0]) + ", " + str(end_point[1]) + "]) = {" + str(f(end_point)) + "}")
    return end_point


def nelder_mead(x_start, no_inc_thr=10e-6, no_inc_break=10, max_iter=0, alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    tic = time.perf_counter()
    dim = len(x_start)
    best_previous = f(x_start)
    no_inc = 0
    res = [[x_start, best_previous]]
    n = len(x_start)
    for i in range(dim):
        x = copy.copy(x_start)
        for j in range(dim):
            if i == j:
                x[j] += (n * (math.sqrt(n + 1) + n - 1)) / (n * math.sqrt(2))
            else:
                x[j] += (n * (math.sqrt(n + 1) - 1)) / (n * math.sqrt(2))
        score = f(x)
        res.append([x, score])

    k = 0
    while True:
        res.sort(key=lambda elem: elem[1])
        best = res[0][1]

        if max_iter and k >= max_iter:
            break
        k += 1

        if best < best_previous - no_inc_thr:
            no_inc = 0
            best_previous = best
        else:
            no_inc += 1

        if no_inc >= no_inc_break:
            break

        ##### ЦЕНТРОИД #####
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res) - 1)

        ##### ОТРАЖЕНИЕ #####
        xr = x0 + alpha * (np.array(x0) - res[-1][0])
        r_score = f(xr)
        if res[0][1] <= r_score < res[-2][1]:
            del res[-1]
            res.append([xr, r_score])
            continue

        ##### РАСШИРЕНИЕ #####
        if r_score < res[0][1]:
            xe = x0 + gamma * (np.array(x0) - res[-1][0])
            e_score = f(xe)
            if e_score < r_score:
                del res[-1]
                res.append([xe, e_score])
                continue
            else:
                del res[-1]
                res.append([xr, r_score])
                continue

        ##### СОКРАЩЕНИЕ ######
        xc = x0 + rho * (np.array(x0) - res[-1][0])
        c_score = f(xc)
        if c_score < res[-1][1]:
            del res[-1]
            res.append([xc, c_score])
            continue

        ##### ПОНИЖЕНИЕ #####
        x1 = res[0][0]
        n_res = []
        for tup in res:
            red_x = []
            for i in range(len(tup[0])):
                red_x.append(x1[i] + sigma * (tup[0][i] - x1[i]))
            score = f(red_x)
            n_res.append([red_x, score])
        res = n_res

    toc = time.perf_counter()
    print("Метод Нелдера — Мида")
    print(f"Количество итераций: " + str(k))
    print(f"Вычисление заняло {toc - tic:0.10f} секунд")
    print("f([" + str(res[0][0]) + "]) = {" + str(res[0][1]) + "}")
    return res[0][0]


if __name__ == '__main__':
    maxIterations = 100

    x0 = [0.5, 0.5]
    print(x0)

    epsilon = 0.0001

    hook_djeevs(2, x0, 0.5, epsilon)
    print()
    nelder_mead(x0)


