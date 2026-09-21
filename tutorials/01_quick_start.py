"""Первое расписание: создание модели, поиск и независимая проверка результата."""
import gds  # Импортируем тонкий интерфейс библиотеки C++.

problem = gds.make_parallel_jobs(2, 8, [2, 3, 2])  # Задаём два аппарата, восемь слотов и длительности трёх работ.
compiled = problem.compile()  # C++ строит размещения и ограничения совместимости.
options = gds.SolverOptions()  # Получаем воспроизводимые стандартные параметры поиска.
options.seed = 42  # Фиксируем зерно генератора случайных чисел.
schedule = gds.solve_gds(compiled, options)  # Запускаем сеть и независимую проверку полного расписания.
assert schedule.feasible, schedule.stop_reason  # Останавливаем пример, если допустимое расписание не найдено.
print(schedule.stop_reason)  # Выводим понятную причину завершения поиска.
for placement in schedule.placements:  # Просматриваем выбранное размещение каждой работы.
    print(placement.action, placement.resource, placement.start, placement.end)  # Границы заданы в слотах, правая граница не входит.
assert compiled.evaluate(schedule.assignment).feasible  # Повторно проверяем именно выбранные индексы размещений.
