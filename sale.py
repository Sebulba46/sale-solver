import numpy as np
import time
from copy import deepcopy


def gauss(a, b):
    start_time = time.time()

    mat_a = deepcopy(a)
    mat_b = deepcopy(b)

    n = len(mat_a)
    x = np.zeros(n)  # матрица numpy для ответа
    k = 0
    for _ in range(n):
        max_val = abs(mat_a[k][k])  # поиск ведущего элемента в строках от k до n с максимальным значением и этого значения
        index = k
        for i in range(k + 1, n):  # выбор главного элемента по столбцу
            if abs(mat_a[i][k]) > max_val:
                max_val = abs(mat_a[i][k])
                index = i
        if max_val == 0:  # если максимальный элемент равен нулю, то решить методом гаусса не получится
            print(f"Решение получить невозможно из-за нулевого столбца {index} матрицы A")
            return 0

        mat_a[[k, index]] = mat_a[[index, k]]  # перестановка строки с максимальным значением наверх
        mat_b[index], mat_b[k] = mat_b[k], mat_b[index]  # перестановка y с максимальным значением наверх

        for i in range(k, n):  # цикл деления и вычитания строк после найденного максимального значения
            temp = mat_a[i][k]  # макс значение
            if abs(temp) == 0:  # если нулевой коэффициент, то пропускаем
                continue
            mat_a[i] = mat_a[i] / temp  # деление строки на max значение
            mat_b[i] = mat_b[i] / temp  # деление y на max значение
            if i == k:  # не вычитаем уравнение само из себя
                continue
            mat_a[i] = mat_a[i] - mat_a[k]  # производим вычитание матриц numpy
            mat_b[i] = mat_b[i] - mat_b[k]

        k += 1
    for k in reversed(range(n)):  # обратный ход
        x[k] = mat_b[k]
        for i in range(k):
            mat_b[i] = mat_b[i] - mat_a[i][k] * x[k]
    return x, str(round(time.time() - start_time, 5)) + 's'


def gauss_rectangle(a, b):
    start_time = time.time()

    n = len(a)
    a = np.column_stack([a, b])  # объединение входов в один
    x = np.zeros(n)  # матрица ответов
    k = 0
    for _ in range(n):
        max_val = abs(a[k][k])  # поиск ведущего элемента в строках от k до n с максимальным значением и этого значения
        index = k
        for i in range(k + 1, n):  # выбор главного элемента по столбцу
            if abs(a[i][k]) > max_val:  # выбор по модулю
                max_val = abs(a[i][k])
                index = i
        if max_val == 0:  # если максимальный элемент равен нулю, то решить методом гаусса не получится
            print(f"Решение получить невозможно из-за нулевого столбца {index} матрицы A")
            return 0

        a[[k, index]] = a[[index, k]]  # перестановка строки с максимальным значением наверх

        a[k] = a[k] / a[k][k]  # деление уравнения на коэффициент x[k, k]
        a_copy = a.copy()  # запоминаем состояние системы уравнений

        for i in range(n - 1 - k):
            if k > i:
                a[k][i] = 0  # все коэффициенты ниже ведущего элемента по столбцу обращаем в ноль
        for i in range(k + 1, n):
            for j in range(k, n + 1):
                a[i][j] = a_copy[i][j] - a_copy[i][k] * a_copy[k][
                    j]  # формула прямоугольника из книги Киреева и Пантелеева

        k += 1
    b = a[:, -1]  # матрица y

    for k in reversed(range(n)):  # обратный ход
        x[k] = b[k]
        for i in range(k):
            b[i] = b[i] - a[i][k] * x[k]
    return x, str(round(time.time() - start_time, 5)) + 's'


def lu_solve(a_lu, b_lu):
    start_time = time.time()

    def decompose_to_LU(a):
        lu_matrix = np.matrix(np.zeros(a.shape))
        for k in range(a.shape[0]):
            for j in range(k, a.shape[0]):
                lu_matrix[k, j] = a[k, j] - lu_matrix[k, :k] * lu_matrix[:k, j]
            for i in range(k + 1, a.shape[0]):
                lu_matrix[i, k] = (a[i, k] - lu_matrix[i, :k] * lu_matrix[:k, k]) / lu_matrix[k, k]
        return lu_matrix

    def get_LU(lu):
        L = lu.copy()
        for i in range(L.shape[0]):
            L[i, i] = 1
            L[i, i + 1:] = 0
        U = lu.copy()
        for i in range(1, U.shape[0]):
            U[i, :i] = 0
        return np.matrix(L), U

    def solve(a, b, rev=False):
        x = np.zeros(b.shape[0])
        for i in range(b.shape[0])[::-1] if rev else range(b.shape[0]):
            x[i] = (b[i] - np.sum(x * a[i].T)) / a[i, i]
        return x

    L_lu, U_lu = get_LU(decompose_to_LU(a_lu))
    return solve(U_lu, solve(L_lu, b_lu), rev=True), str(round(time.time() - start_time, 5)) + 's'


def gs_LU(lu):
    L = lu.copy()
    for i in range(L.shape[0]):
        L[i, i + 1:] = 0
    U = lu.copy()
    for i in range(0, U.shape[0]):
        U[i, :i+1] = 0
    return np.matrix(L), U


def gauss_seidel(a, b, tolerance, max_iterations):
    start_time = time.time()

    x = np.zeros_like(b, dtype=np.double)

    n = len(a)

    L_lu, U_lu = get_LU(a)
    L_lu_neg = np.linalg.matrix_power(L_lu, -1)

    for i in range(max_iterations):

        x_new = np.dot(np.dot(-L_lu_neg, U_lu), x) + np.dot(L_lu_neg, b)
        x_new = np.array(x_new).reshape(n,)

        if np.allclose(x, x_new, rtol=tolerance):
            return x, str(round(time.time() - start_time, 5)) + 's'

        x = x_new

    return x, str(round(time.time() - start_time, 5)) + 's'


def iteration(a, b, tolerance, max_iterations):
    start_time = time.time()
    x = np.zeros_like(b, dtype=np.double)
    T = a - np.diag(np.diagonal(a))

    for k in range(max_iterations):
        x_old = x.copy()
        x[:] = (b - np.dot(T, x)) / np.diagonal(a)
        if np.linalg.norm(x - x_old, ord=np.inf) <= tolerance:
            break

    return x, str(round(time.time() - start_time, 5)) + 's'
