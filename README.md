# OpSys CourseWork: Matrix Multiplication

<!-- Meta & Language -->
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?logo=ubuntu&logoColor=white)](#)

<!-- Data Science & Math -->
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](#)

<!-- Infrastructure & Tools -->
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?logo=python&logoColor=white)](#)
[![PrettyTable](https://img.shields.io/badge/PrettyTable-2.2.0-3776AB?logo=python&logoColor=white)](#)

**Исследование многопоточности и многопроцессорности Python в ОС Linux** на примере операции умножения матриц

Парная курсовая работа по дисциплине «Операционные системы», сравнивающая производительность последовательных, многопоточных и многопроцессорных вычислений в Python.

## О проекте

Проект реализует алгоритм умножения квадратных матриц размерности от 100 до 1000. Для исследования параллельных вычислений реализованы два подхода: разделение задачи на потоки (`threading`) и на независимые процессы с использованием разделяемой памяти (`multiprocessing.SharedMemory`).

Участники:
  - [KnightKotOR](https://github.com/KnightKotOR)
  - [yrmint](https://github.com/yrmint)

### Ключевые возможности

- **Многопоточная реализация:** Разделение строк результирующей матрицы между потоками (1, 8 и 16 потоков) с помощью библиотеки `threading`.
- **Многопроцессорная реализация:** Создание независимых процессов (1, 8 и 16 процессов) с помощью `multiprocessing` и синхронизация результатов через `SharedMemory`.
- **Бенчмаркинг:** Автоматизированный замер времени выполнения для матриц разного размера.
- **Визуализация:** Вывод результатов в виде наглядных таблиц (`PrettyTable`) и графиков зависимости времени от размерности (`Matplotlib`).

### Результаты

- **Многопоточность:** Время выполнения не сократилось по сравнению с однопоточным режимом. Это связано с наличием GIL (Global Interpreter Lock) в CPython, который запрещает параллельное выполнение байт-кода Python в потоках, нагружающих CPU.
- **Многопроцессорность:** Получено ожидаемое ускорение вычислений. Для малых матриц однопроцессорный режим быстрее из-за накладных расходов на создание процессов, но при увеличении размера матрицы многопроцессорный подход показывает значительный выигрыш. Увеличение числа процессов сверх количества логических ядер ЦП не дает существенного прироста производительности.

Результат программы multiprocess в графическом виде для 8-ядерного Ryzen 7700(слева) и 4-ядерного Intel Core i5-1135G7(справа):

<p align="center">
  <img width="360" height="270" alt="image" src="https://github.com/user-attachments/assets/295ea915-7354-4fdb-aa64-7e115137b3eb" />
  <img width="360" height="270" alt="image" src="https://github.com/user-attachments/assets/c90e7b09-66dc-4545-bad3-3cd24c9b64e8" />
</p>
  
## Технологический Стек

- **Язык:** Python 3.10
- **ОС:** Ubuntu 22.04.3
- **Вычисления и данные:** NumPy
- **Визуализация и отчетность:** Matplotlib, PrettyTable

## Установка и запуск

### Предварительные требования
- [Python](https://www.python.org/downloads/) версии 3.10 или выше

### 1. Клонирование репозитория

```commandline
git clone https://github.com/KnightKotOR/OpSys_CourseWork.git
```

### 2. Установка зависимостей

Создайте виртуальное окружение и установите необходимые библиотеки:

```commandline
python -m venv venv
source venv/bin/activate  # для Windows: venv\Scripts\activate
```

```commandline
pip install numpy matplotlib prettytable
```

### 3. Запуск скриптов

Для запуска исследования многопоточной реализации (демонстрация влияния GIL):
```commandline
python multithread.py
```

Для запуска исследования многопроцессорной реализации (реальное ускорение):
```commandline
python multiprocess.py
```

После запуска скриптов в консоли отобразятся таблицы с временем выполнения, а во всплывающих окнах будут построены графики зависимостей.
