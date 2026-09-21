"""Парная пригодность, предшествование и непрерывный блок научных действий."""
import math  # Нужна только демонстрационная формула внешнего предпочтения.
import gds  # Компиляция функции в веса и запреты остаётся в C++.


def pair_preference(first, second):
    """Запрещает обратный порядок и поощряет небольшой допустимый разрыв."""
    gap = second.start - first.end  # Вычисляем разрыв в целых слотах между действиями.
    return 0.0 if gap < 0 else math.exp(-0.1 * gap)  # Нуль создаст жёсткий запрет, положительное значение — мягкий вес.


problem = gds.Problem(8)  # Создаём восемь секундных слотов.
satellite = gds.Resource()  # Описываем одну последовательную линию.
satellite.name, satellite.sequence = "Аппарат", True  # Задаём имя и единичную последовательную занятость.
problem.add_resource(satellite)  # Регистрируем аппарат под индексом ноль.
for name, group in [("Калибровка", "Подготовка"), ("Цель А", "Наука"), ("Цель Б", "Наука")]:  # Описываем три действия.
    request = gds.Action()  # Создаём новую заявку.
    request.name, request.group = name, group  # Сохраняем имя и семантическую группу.
    request.modes = [gds.Mode()]  # Используем один способ исполнения длительностью одна секунда.
    problem.add_action(request)  # Передаём полностью подготовленную заявку в C++.
problem.add_precedence(gds.Precedence(0, 1, min_gap=1, max_gap=3))  # Между калибровкой и первой целью требуется разрыв от одного до трёх слотов.
problem.add_precedence(gds.Precedence(0, 2))  # Вторая цель также начинается только после калибровки.
problem.add_contiguous_group("Наука")  # Запрещаем вставлять подготовительные действия между научными.
problem.add_pair_suitability(1, 2, pair_preference)  # Передаём функцию, которая будет вычислена при компиляции.
compiled = problem.compile()  # C++ преобразует парную функцию в явные запреты и мягкие веса.
schedule = gds.greedy(compiled)  # Выполняем жадный выбор с проверкой ограничений.
if not schedule.feasible:  # Частичный жадный выбор иногда требует перестановки ранее выбранных действий.
    schedule = gds.min_conflicts(compiled, schedule.assignment)  # Исправляем наиболее конфликтные действия.
assert schedule.feasible  # Убеждаемся в выполнении всех условий.
print([(p.action, p.start, p.end) for p in schedule.placements])  # Выводим порядок и интервалы действий.
