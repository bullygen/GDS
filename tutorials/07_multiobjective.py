"""Многокритериальный поиск: скрещивание расписаний GDS, GDE3 и граница Парето."""
from pathlib import Path  # Задаём отдельный каталог рисунков данного примера.
import matplotlib  # Подключаем средство построения графиков.
matplotlib.use("Agg")  # Сохраняем рисунки без необходимости графического рабочего стола.
import matplotlib.pyplot as plt  # Сохраняем и закрываем созданные рисунки.
import gds_plot  # Отделяем визуализацию от вычислительного ядра C++.
import gds  # Генетические операторы и ранжирование реализованы на C++.

problem = gds.Problem(5)  # Создаём заведомо тесный горизонт для необязательных наблюдений.
satellite = gds.Resource()  # Описываем один аппарат.
satellite.name, satellite.sequence = "Аппарат", True  # Разрешаем только последовательную работу.
satellite.momentum_limit = 10  # Вводим физический предел момента.
problem.add_resource(satellite)  # Регистрируем ресурс.
for index, momentum in enumerate([3, -2, 4]):  # Разные заявки создают разные компромиссы.
    mode = gds.Mode()  # Создаём способ исполнения длительностью две секунды.
    mode.duration, mode.momentum = 2, [momentum, 0, 0]  # Задаём время и знак приращения момента.
    observation = gds.Action()  # Создаём наблюдение, которое разрешено пропустить.
    observation.name, observation.modes = f"Цель {index}", [mode]  # Назначаем имя и способ исполнения.
    observation.optional, observation.priority = True, index + 1  # Пропуск даёт потерю научного приоритета.
    problem.add_action(observation)  # Добавляем заявку с отдельным значением пропуска.
compiled = problem.compile()  # Строим конечное множество всех размещений и пропусков.
options = gds.SolverOptions()  # Задаём одинаковый бюджет двум эволюционным методам.
options.population, options.generations = 8, 5  # Используем восемь расписаний и пять поколений.
options.iterations, options.repair_steps = 1000, 50  # Ограничиваем внутренние запуски сети и исправления.
options.momentum_balancing = True  # Включаем эвристику исходного жадного члена популяции GDE3.
for solver in [gds.genetic, gds.gde3]:  # Сравниваем два разных эволюционных оператора.
    result = solver(compiled, options)  # Каждый метод возвращает итоговую популяцию и её фронт.
    print(solver.__name__)  # Указываем название применённого метода.
    for schedule in result.pareto_front:  # Просматриваем приближение границы Парето.
        print(schedule.feasible, schedule.objectives, schedule.assignment)  # Не смешиваем допустимость с качеством критериев.
        assert not any(gds.dominates(other, schedule) for other in result.population)  # Проверяем недоминируемость внутри популяции.

    candidates = [member for member in result.pareto_front if member.feasible]  # Оставляем подтверждённые планы текущего метода.
    assert candidates, "Метод не нашёл допустимого расписания."  # Не выдаём нарушенный план за итоговое расписание.
    schedule = min(candidates, key=lambda member: member.objectives)  # Для рисунка выбираем один воспроизводимый компромисс.
    output = Path("outputs") / Path(__file__).stem / solver.__name__  # Разделяем результаты разных туториалов по каталогам.
    output.mkdir(parents=True, exist_ok=True)  # Создаём каталог для двух рисунков.
    axes = gds_plot.gantt(schedule, problem, show_labels=False)  # Отключаем текст внутри временных полос отдельным параметром.
    axes.figure.tight_layout()  # Размещаем подписи осей и названия ресурсов.
    axes.figure.savefig(output / "schedule.png", dpi=150)  # Сохраняем диаграмму расписания.
    plt.close(axes.figure)  # Освобождаем память после сохранения расписания.
    resource_axes = gds_plot.resource_traces(schedule, slot_seconds=problem.slot_seconds, problem=problem)  # Рисуем все ресурсы и величины друг под другом.
    resource_axes[0].figure.tight_layout()  # Разделяем подписи соседних ресурсных графиков.
    resource_axes[0].figure.savefig(output / "resources.png", dpi=150)  # Сохраняем общий рисунок с отдельными графиками.
    plt.close(resource_axes[0].figure)  # Закрываем рисунок ресурсных рядов.
