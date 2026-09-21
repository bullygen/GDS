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
output = Path("outputs")  # Выбираем каталог, исключённый из публикации.
output.mkdir(exist_ok=True)  # Создаём его при необходимости.
axes = gds_plot.gantt(schedule, problem)  # Строим полосы действий и свободных интервалов.
axes.figure.tight_layout()  # Размещаем русские подписи внутри рисунка.
axes.figure.savefig(output / "schedule.png", dpi=150)  # Сохраняем самостоятельное изображение.
plt.close(axes.figure)  # Освобождаем ресурсы графического объекта.
axes, momentum_axes = gds_plot.resource_traces(schedule)  # Разделяем память и момент по разным вертикальным осям.
axes.figure.tight_layout()  # Подгоняем расположение двух подписей величин.
axes.figure.savefig(output / "resources.png", dpi=150)  # Сохраняем временные ряды.
plt.close(axes.figure)  # Завершаем работу с графиком.
options = gds.SolverOptions()  # Ограничиваем размер популяции для учебной иллюстрации.
options.population, options.generations = 4, 2  # Четырёх членов достаточно для демонстрации GDE3.
population = gds.gde3(problem.compile(), options)  # Получаем множество планов с отдельными значениями критериев.
axes = gds_plot.pareto(population, x=1, y=2)  # Показываем проекцию по простою и пиковому моменту.
axes.figure.tight_layout()  # Размещаем подписи критериев в пределах рисунка.
axes.figure.savefig(output / "pareto.png", dpi=150)  # Сохраняем третий самостоятельный график.
plt.close(axes.figure)  # Освобождаем последний графический объект.
print("Рисунки сохранены в", output)  # Сообщаем, где находятся результаты примера.
