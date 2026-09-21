"""Внешние длительности: прямой расчёт действия, предикат перехода и точная проверка."""
from pathlib import Path  # Задаём отдельный каталог рисунков данного примера.
import matplotlib  # Подключаем средство построения графиков.
matplotlib.use("Agg")  # Сохраняем рисунки без необходимости графического рабочего стола.
import matplotlib.pyplot as plt  # Сохраняем и закрываем созданные рисунки.
import gds_plot  # Отделяем визуализацию от вычислительного ядра C++.
import gds  # C++ отвечает за квантование, кэш, последовательность и проверку.


def operation_duration(request):
    """Имитирует внешний расчёт длительности операции; реальный модуль подключается здесь."""
    answer = gds.DurationResult()  # Создаём типизированный ответ внешнего модуля.
    answer.status = gds.DurationStatus.FEASIBLE  # Явно подтверждаем успешность расчёта.
    answer.seconds = 1.2  # Физическая длительность потребует двух секундных слотов.
    answer.certificate = "учебная-модель-операции"  # Сохраняем происхождение результата.
    return answer  # Возвращаем физические данные; построение нейронов выполнит C++.


def transition_predicate(request, duration):
    """Имитирует монотонный предикат: поворот допустим при длительности от 1,1 секунды."""
    return gds.DurationStatus.FEASIBLE if duration >= 1.1 else gds.DurationStatus.INFEASIBLE  # Возвращаем проверку одной пробы.


def exact_transition(request):
    """Вызывает C++-поиск минимальной сеточной длительности поверх внешнего предиката."""
    answer = gds.estimate_duration(request, transition_predicate, max_slots=5, monotone=True)  # Монотонность здесь известна по определению модели.
    answer.certificate = "учебная-модель-перехода"  # Добавляем идентификатор расчёта к метаданным полосы.
    answer.end_state = [request.state[0] + 1]  # Передаём новое состояние аппарата следующему соседнему переходу.
    return answer  # Не преобразуем результат в мягкий штраф на стороне Python.


problem = gds.Problem(12)  # Создаём горизонт двенадцать секунд с секундным шагом.
satellite = gds.Resource()  # Описываем один последовательный аппарат.
satellite.name, satellite.sequence = "Аппарат", True  # Включаем проверку соседних переходов на этой линии.
satellite.initial_state = [0]  # Задаём начальное состояние учебной внешней модели.
problem.add_resource(satellite)  # Ресурс получает индекс ноль.
for index in range(3):  # Создаём три обязательные операции.
    mode = gds.Mode()  # Используем стандартный основной ресурс ноль.
    mode.duration = 1  # Номинальная длительность заменится внешним результатом 1,2 секунды.
    operation = gds.Action()  # Создаём заявку.
    operation.name, operation.modes = f"Операция {index}", [mode]  # Связываем имя с описанным способом исполнения.
    problem.add_action(operation)  # Добавляем действие в модель.
problem.set_duration_provider(operation_duration, "операция-1")  # Подключаем прямой внешний расчёт длительности.
problem.set_transition_provider(exact_transition, "переход-1")  # Подключаем переходы между соседними действиями.
compiled = problem.compile()  # C++ строит области значений с консервативным округлением длительностей.
schedule = gds.solve_gds(compiled)  # Непоместившиеся переходы вызовут новые запреты и повторный поиск.
assert schedule.feasible  # Используем только подтверждённую последовательность.
for segment in schedule.segments:  # Просматриваем действия, переходы и свободные интервалы.
    print(segment.kind, segment.begin, segment.end, segment.metadata.certificate)  # Физическая длительность и сертификат доступны явно.
print("Вызовы и попадания в кэш:", compiled.cache_statistics())  # Проверяем повторное использование одинаковых запросов.

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
