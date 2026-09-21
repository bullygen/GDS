"""Первое расписание: создание модели, поиск и независимая проверка результата."""
from pathlib import Path  # Задаём отдельный каталог рисунков данного примера.
import matplotlib  # Подключаем средство построения графиков.
matplotlib.use("Agg")  # Сохраняем рисунки без необходимости графического рабочего стола.
import matplotlib.pyplot as plt  # Сохраняем и закрываем созданные рисунки.
import gds_plot  # Отделяем визуализацию от вычислительного ядра C++.
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

output = Path("outputs") / Path(__file__).stem  # Разделяем результаты разных туториалов по каталогам.
output.mkdir(parents=True, exist_ok=True)  # Создаём каталог для двух рисунков.
axes = gds_plot.gantt(schedule, problem, show_labels=False)  # Отключаем текст внутри временных полос отдельным параметром.
axes.figure.tight_layout()  # Размещаем подписи осей и названия ресурсов.
axes.figure.savefig(output / "schedule.png", dpi=150)  # Сохраняем диаграмму расписания.
plt.close(axes.figure)  # Освобождаем память после сохранения расписания.
resource_axes = gds_plot.resource_traces(schedule, slot_seconds=problem.slot_seconds, problem=problem)  # Рисуем все ресурсы и величины друг под другом.
resource_axes[0].figure.tight_layout()  # Разделяем подписи соседних ресурсных графиков.
resource_axes[0].figure.savefig(output / "resources.png", dpi=150)  # Сохраняем общий рисунок с отдельными графиками.
plt.close(resource_axes[0].figure)  # Закрываем рисунок ресурсных рядов.
