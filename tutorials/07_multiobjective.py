"""Многокритериальный поиск: скрещивание расписаний GDS, GDE3 и граница Парето."""
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
