"""Стандартные постановки: цепочки операций JSSP и направленный цикл коммивояжёра."""
from pathlib import Path  # Задаём отдельный каталог рисунков данного примера.
import matplotlib  # Подключаем средство построения графиков.
matplotlib.use("Agg")  # Сохраняем рисунки без необходимости графического рабочего стола.
import matplotlib.pyplot as plt  # Сохраняем и закрываем созданные рисунки.
import gds_plot  # Отделяем визуализацию от вычислительного ядра C++.
import gds  # Построение обеих задач целиком реализовано на C++.

jobs = [  # Каждая вложенная последовательность задаёт операции одной работы.
    [gds.Operation(0, 2), gds.Operation(1, 2)],  # Первая работа проходит сначала станок ноль, затем станок один.
    [gds.Operation(1, 2), gds.Operation(0, 1)],  # Вторая работа проходит станки в противоположном порядке.
]
compiled = gds.make_jssp(2, 8, jobs).compile()  # Строим условия занятости станков и предшествования операций.
schedule = gds.solve_gds(compiled)  # Ищем размещения операций без разрыва внутри операции.
assert schedule.feasible  # Принимаем расписание только после полной проверки.
print("Размещения JSSP:", [(p.action, p.resource, p.start, p.end) for p in schedule.placements])  # Выводим четыре операции.
costs = [[0, 1, 9], [9, 0, 1], [1, 9, 0]]  # Стоимость дуги зависит от её направления.
options = gds.SolverOptions()  # Создаём отдельные параметры для исследования маршрутов.
options.iterations = 1000  # Ограничиваем число шагов одного запуска.
options.restarts = 3  # Повторяем запуск с тремя последовательными зёрнами.
tour = gds.solve_tsp(costs, options)  # Строим и исследуем сеть по координатам позиция—город.
assert tour.feasible  # Проверяем существование найденного замкнутого цикла.
assert gds.tour_cost(tour.tour, costs) == tour.cost  # Стоимость включает возвращение из последнего города в первый.
print("Цикл:", tour.tour, "стоимость:", tour.cost)  # Печатаем лучший из исследованных допустимых маршрутов.

problem = compiled.problem  # Для рисунка JSSP берём описания станков из скомпилированной задачи.

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
