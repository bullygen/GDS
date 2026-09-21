"""Отдельный пакет графиков: расписание и временные ряды ресурсов."""
from pathlib import Path  # Результаты сохраняем в отдельный локальный каталог.
import matplotlib  # Выбираем режим без графического рабочего стола.
matplotlib.use("Agg")  # Пример одинаково исполняется локально и в автоматических проверках.
import matplotlib.pyplot as plt  # Получаем операции сохранения и закрытия рисунков.
import gds  # Используем C++-библиотеку для поиска.
import gds_plot  # Используем независимый Python-пакет только для рисования.

problem = gds.make_parallel_jobs(2, 8, [2, 3, 2])  # Создаём небольшую задачу двух аппаратов.
schedule = gds.solve_gds(problem.compile())  # Находим и проверяем расписание.
assert schedule.feasible  # Подтверждаем результат до построения отчётного рисунка.

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

options = gds.SolverOptions()  # Ограничиваем размер популяции для учебной иллюстрации.
options.population, options.generations = 4, 2  # Четырёх членов достаточно для демонстрации GDE3.
population = gds.gde3(problem.compile(), options)  # Получаем множество планов с отдельными значениями критериев.
axes = gds_plot.pareto(population, x=1, y=2)  # Показываем проекцию по простою и пиковому моменту.
axes.figure.tight_layout()  # Размещаем подписи критериев в пределах рисунка.
axes.figure.savefig(output / "pareto.png", dpi=150)  # Сохраняем третий самостоятельный график.
plt.close(axes.figure)  # Освобождаем последний графический объект.
print("Рисунки сохранены в", output)  # Сообщаем, где находятся результаты примера.
