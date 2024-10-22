from prettytable import PrettyTable
from threading import Thread
from time import time

import numpy as np
import matplotlib.pyplot as plt


def create_matrices(n):
    """
    Возвращает 2 матрицы, заполненные единицами
    :param n: Размер матриц (int)
    :return: 2 матрицы заданного размера
    """
    m1, m2 = np.ones((n, n)), np.ones((n, n))
    return m1, m2


def rows_multiplier(s_row, e_row, matrix1, matrix2, data):
    """
    Вычисляет элементы заданных строк произведения двух матриц
    :param s_row: Первая строка
    :param e_row: Последняя строка
    :param matrix1: Матрица A
    :param matrix2: Матрица B
    :param data: Результирующая матрица
    """
    m_size = len(matrix1)
    for i in range(s_row, e_row):
        for j in range(m_size):
            for k in range(m_size):
                data[i][j] += matrix1[i][k] * matrix2[k][j]


def multithreading_matrices_multiplier(matrix1, matrix2, n_threads):
    """
    Возвращает A x B. Вычисление производятся при заданном числе потоков
    :param matrix1: Матрица A
    :param matrix2: Матрица B
    :param n_threads: Число потоков (от 1 до размера матриц)
    :return: Произведение матрицы A на B
    """
    res_matrix = np.zeros((matrix1.shape[0], matrix2.shape[1]))
    threads = []
    step = size // n_threads
    chunks = np.arange(0, size + 1, step)
    chunks[len(chunks) - 1] = size

    for i in range(n_threads):
        thread = Thread(target=rows_multiplier,
                        args=(chunks[i], chunks[i + 1], matrix1, matrix2, res_matrix))
        thread.start()
        threads.append(thread)

    for process in threads:
        process.join()

    return res_matrix


def draw_plots(time_results, matrix_sizes, n_processes_list):
    """
    Строит графики зависимости времени вычисления от размера матриц для различного количества потоков
    :param time_results: Словарь, содержащий список времени вычисления произведения в зависимости от размера матриц,
                         для каждого количества потоков
    :param matrix_sizes: Список размеров матриц, с которыми проводилось исследование
    :param n_processes_list: Список количества потоков, с которыми проводилось исследование
    """
    for n_processes in n_processes_list:
        plt.plot(matrix_sizes, time_results[n_processes])
        plt.legend(n_processes_list)
        plt.ylabel("Elapsed time, s")
        plt.xlabel("Matrices size")
    plt.show()


def print_table(time_results, matrix_sizes, n_processes_list):
    """
    Строит таблицу зависимости времени вычисления от размера матриц для различного количества потоков
    :param time_results: Словарь, содержащий список времени вычисления произведения в зависимости от размера матриц,
                         для каждого количества потоков
    :param matrix_sizes: Список размеров матриц, с которыми проводилось исследование
    :param n_processes_list: Список количества потоков, с которыми проводилось исследование
    """
    table = PrettyTable()
    table.add_column("Size", matrix_sizes)
    for n_processes in n_processes_list:
        table.add_column(str(n_processes), time_results[n_processes])
    print(table)


if __name__ == '__main__':
    # Инциализация структур данных
    sizes = np.arange(100, 200 + 1, 100)
    threads = [1, 8, 16]
    res = {n_threads: [0.0 for i in range(len(sizes))] for n_threads in threads}
    counter = -1

    # Определение зависимости
    for size in sizes:
        counter += 1
        A, B = create_matrices(size)
        for n_threads in threads:
            # Вычисление произведения и фиксирование времени выполнения
            start = time()
            mult_res = multithreading_matrices_multiplier(A, B, n_threads)
            duration = time() - start
            res[n_threads][counter] = np.round(duration, 2)

    print_table(res, sizes, threads)
    draw_plots(res, sizes, threads)
