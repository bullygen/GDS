"""Низкоуровневая сеть: ограничения, шаги динамики, ферзи и сохранение состояния."""
from pathlib import Path  # Используем переносимое представление пути к выходному файлу.
import gds  # Все дальнейшие действия сети исполняются внутри C++.

network = gds.GDSNetwork(2, [2, 2], seed=7)  # Создаём две переменные с двумя значениями каждая.
network.add_forbidden_pair(0, 0, 1, 0)  # Запрещаем одновременный выбор нулевого значения обеих переменных.
network.add_weighted_pair(0, 1, 1, 1, 0.2)  # Мягко поощряем пару единичных значений.
network.set_unary_bias(0, 1, 0.1)  # Добавляем индивидуальное предпочтение одного нейрона.
network.finalize_weights()  # Перестраиваем списки связей после добавления пар.
answer = network.run(1000)  # Выполняем не более тысячи стохастических шагов.
assert answer.satisfied  # Проверяем охранные условия конечной CSP, а не физику внешнего аппарата.
print("Назначение:", answer.assignment)  # Выводим индекс выбранного значения каждой переменной.
print("Охранные нейроны:", network.get_guard_state())  # В допустимом состоянии охранные нейроны выключены.
directory = Path("outputs")  # Все результаты помещаем в исключённый из Git каталог.
directory.mkdir(exist_ok=True)  # Создаём каталог, если это первый запуск примера.
filename = str(directory / "network.gdsw")  # Выбираем имя текстового файла состояния GDSW2.
network.save_weights(filename)  # Сохраняем также состояние генератора случайных чисел.
restored = gds.GDSNetwork.load_weights(filename)  # Загружаем сеть без повторного описания ограничений.
assert restored.step().neuron_index == network.step().neuron_index  # Проверяем одинаковое продолжение динамики.
queens = gds.GDSNetwork.make_n_queens(8)  # Строим классическую задачу восьми ферзей на C++.
assert queens.run(100000).satisfied  # Проверяем решение этой независимой задачи ограничений.
coloring = gds.GDSNetwork.make_graph_3_coloring(3, [(0, 1), (1, 2), (2, 0)])  # Строим раскраску треугольника.
assert coloring.run(1000).satisfied  # Три вершины должны получить три разных цвета.
