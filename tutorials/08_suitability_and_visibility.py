"""Функции пригодности, угловая видимость и непрерывное время до квантования."""
from pathlib import Path  # Задаём отдельный каталог рисунков данного примера.
import matplotlib  # Подключаем средство построения графиков.
matplotlib.use("Agg")  # Сохраняем рисунки без необходимости графического рабочего стола.
import matplotlib.pyplot as plt  # Сохраняем и закрываем созданные рисунки.
import gds_plot  # Отделяем визуализацию от вычислительного ядра C++.
import gds  # Геометрия, квантование и обработка пригодности выполняются в C++.

pointing = gds.radec_to_unit(0, 0)  # Переводим прямое восхождение и склонение в единичный вектор.
body = [[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]  # Используем учебные положения мешающего тела по слотам.
mask = gds.angular_visibility(pointing, body, [], 45, at_least=True)  # Требуем угловое удаление не менее 45 градусов.
problem = gds.Problem(5)  # Создаём сетку, согласованную с длиной таблицы положений.
satellite = gds.Resource()  # Описываем единственный аппарат.
satellite.name, satellite.sequence = "Телескоп", True  # Включаем последовательную линию телескопа.
problem.add_resource(satellite)  # Сохраняем ресурс в постановке.
mode = gds.Mode()  # Создаём способ исполнения наблюдения.
mode.duration = 2  # Наблюдение занимает два последовательных слота.
observation = gds.Action()  # Создаём заявку на одно наблюдение.
observation.name, observation.modes = "Цель", [mode]  # Задаём имя и единственный способ исполнения.
observation.suitability = mask  # Нулевое значение запрещает любой интервал, содержащий этот слот.
problem.add_action(observation)  # Передаём полную заявку в C++.
compiled = problem.compile()  # Проверяем пригодность на всём интервале каждого размещения.
assert [p.start for p in compiled.domains[0]] == [0, 3]  # Два допустимых начала не пересекают запрещённый третий слот.
sample = gds.sample_slot(lambda t: 1 + t, 0, 1)  # Исследуем учебную непрерывную функцию в узлах одного слота.
print(sample.minimum, sample.maximum, sample.mean)  # Для линейной функции среднее равно 1,5.
print(gds.log_suitability([0.5, 0.8]))  # Логарифмическое объединение сохраняет произведение мягких предпочтений.
window = gds.quantize_interval(0.2, 4.8, 1, "inside")  # Вписываем сеточный интервал внутрь непрерывного окна.
print(window.begin, window.end)  # Получаем полуоткрытый интервал от первого до четвёртого слота.

schedule = gds.greedy(compiled)  # Выбираем размещение с учётом вычисленной видимости.
assert schedule.feasible  # Проверяем расписание перед визуализацией.

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
