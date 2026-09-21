"""Память, векторный момент, энергия, квота и исправление исходного конфликтного плана."""
import gds  # Все проверки ресурсов и перестановки выполняет C++.

problem = gds.Problem(8)  # Задаём восемь секундных слотов.
satellite = gds.Resource()  # Создаём ресурс аппарата с накопительными величинами.
satellite.name, satellite.sequence = "Научный аппарат", True  # Включаем последовательное исполнение.
satellite.memory_capacity = 8  # Память измеряется в согласованных единицах входных данных.
satellite.momentum_limit = 5  # Ограничиваем евклидову норму вектора момента.
satellite.initial_level, satellite.minimum_level, satellite.maximum_level = 10, 0, 10  # Используем накопительный уровень как энергию.
problem.add_resource(satellite)  # Аппарат получает индекс ноль.
for index, delta in enumerate([[3, 0, 0], [-2, 0, 0], [1, 0, 0]]):  # Задаём приращения момента трёх наблюдений.
    mode = gds.Mode()  # Описываем исполнение длительностью один слот.
    mode.momentum, mode.memory = delta, 2  # В конце действия добавляем вектор момента и два блока данных.
    mode.flows = [gds.Flow(0, rate=-1)]  # В течение действия расходуем одну единицу энергии в секунду.
    observation = gds.Action()  # Создаём заявку научной группы.
    observation.name, observation.group = f"Наблюдение {index}", "Наука"  # Группа позволит задать квоту.
    observation.modes = [mode]  # Передаём способ исполнения целиком.
    problem.add_action(observation)  # Сохраняем заявку в постановке.
problem.add_quota(gds.Quota("Наука", 3))  # Требуем выполнить все три научные заявки.
compiled = problem.compile()  # Строим размещения с проверкой диапазонов входных величин.
initial = [0, 0, 0]  # Намеренно ставим три наблюдения в один начальный слот.
assert not compiled.evaluate(initial).feasible  # Независимая проверка обнаруживает конфликт аппарата.
options = gds.SolverOptions()  # Получаем пределы локального поиска.
options.momentum_balancing = True  # Для жадного старта можно сначала выбирать уменьшающие момент действия.
schedule = gds.min_conflicts(compiled, initial, options)  # Переносим действия из наиболее конфликтной области.
assert schedule.feasible  # Проверяем аппарат, память, энергию, момент и квоту одновременно.
print("Память:", schedule.traces[0].memory)  # Выводим память после каждого слота.
print("Энергия:", schedule.traces[0].level)  # Выводим оставшийся накопительный ресурс.
print("Момент:", schedule.traces[0].momentum)  # Выводим полный вектор, сохраняя знаки компонент.
print("Уменьшающие момент:", gds.find_reducers([[3, 0, 0], [-2, 0, 0], [1, 0, 0]])[0])  # Применяем критерий из статьи 2007 года.
