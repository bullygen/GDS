"""Группировка аппаратов с общим наземным каналом и альтернативами исполнения."""
from pathlib import Path  # Результаты сохраняем в отдельный локальный каталог.
import matplotlib  # Выбираем режим без графического рабочего стола.
matplotlib.use("Agg")  # Пример одинаково исполняется локально и в автоматических проверках.
import matplotlib.pyplot as plt  # Получаем операции сохранения и закрытия рисунков.
import gds  # Используем C++-библиотеку для поиска.
import gds_plot  # Используем независимый Python-пакет только для рисования.

problem = gds.Problem(12, 10.0)  # Планируем двенадцать слотов по десять секунд.
for name in ["Спутник А", "Спутник Б"]:  # Создаём независимые последовательные линии аппаратов.
    satellite = gds.Resource()  # Получаем структуру описания ресурса.
    satellite.name = name  # Присваиваем название для отчётов и графика.
    satellite.sequence = True  # На аппарате одновременно выполняется не более одного действия.
    problem.add_resource(satellite)  # Индексы аппаратов будут равны нулю и единице.
channel = gds.Resource()  # Создаём общий возобновляемый ресурс.
channel.name = "Наземный канал"  # Обозначаем физический смысл общей ёмкости.
channel_id = problem.add_resource(channel)  # Канал получает индекс два и единичную ёмкость.
for index in range(4):  # Каждая из четырёх заявок может исполняться любым аппаратом.
    observation = gds.Action()  # Создаём новую заявку.
    observation.name = f"Наблюдение {index}"  # Сохраняем имя в описании задачи.
    observation.windows = [gds.Window(0, 6), gds.Window(8, 12)]  # Требуем целиком попасть в одно окно видимости.
    modes = []  # Подготавливаем список способов исполнения; список затем присваивается целиком.
    for satellite_id in [0, 1]:  # Перебираем два допустимых аппарата.
        mode = gds.Mode()  # Создаём один способ исполнения на выбранном аппарате.
        mode.resource = satellite_id  # Указываем основной ресурс действия.
        mode.duration = 20.0  # Длительность задаём в секундах, то есть в двух слотах.
        mode.demands = [gds.Demand(channel_id, 1)]  # На всё время наблюдения резервируем общий канал.
        modes.append(mode)  # Добавляем альтернативу в локальный список.
    observation.modes = modes  # Передаём в C++ полное множество альтернатив.
    problem.add_action(observation)  # Добавляем заявку с единственным выбором среди всех альтернатив.
compiled = problem.compile()  # C++ строит нейроны время—действие—аппарат и общие ограничения канала.
schedule = gds.solve_gds(compiled)  # Ищем совместимое расписание всей группировки.
assert schedule.feasible  # Проверка охватывает оба аппарата и наземный канал.
assert max(schedule.traces[channel_id].load) <= 1  # Независимо смотрим фактическую загрузку общего ресурса.
print([(p.action, p.resource, p.start) for p in schedule.placements])  # Печатаем выбранные аппарат и начальный слот.

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
