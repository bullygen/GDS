"""Отклонение приближённого расписания точной моделью и последующее исправление."""
from pathlib import Path  # Задаём отдельный каталог рисунков данного примера.
import matplotlib  # Подключаем средство построения графиков.
matplotlib.use("Agg")  # Сохраняем рисунки без необходимости графического рабочего стола.
import matplotlib.pyplot as plt  # Сохраняем и закрываем созданные рисунки.
import gds_plot  # Отделяем визуализацию от вычислительного ядра C++.
import gds  # Используем общий внешний контракт библиотеки.


def approximate(request):
    """Возвращает заведомо оптимистичную нулевую длительность перехода."""
    answer = gds.DurationResult()  # Создаём ответ приближённой модели.
    answer.status, answer.seconds = gds.DurationStatus.FEASIBLE, 0  # Подтверждаем только модельное предположение.
    return answer  # Точный проверяющий провайдер будет задан отдельно.


def exact(request):
    """Требует две секунды на переход в окончательной учебной модели."""
    answer = gds.DurationResult()  # Создаём независимый физический ответ.
    answer.status, answer.seconds = gds.DurationStatus.FEASIBLE, 2  # Возвращаем окончательную длительность.
    answer.certificate = "окончательная-учебная-проверка"  # Сохраняем происхождение подтверждения.
    return answer  # Метаданные будут включены в полосу перехода.


problem = gds.Problem(6)  # Резервируем достаточно времени для исправленного плана.
satellite = gds.Resource()  # Описываем один аппарат.
satellite.name, satellite.sequence = "Аппарат", True  # Включаем соседние переходы.
problem.add_resource(satellite)  # Регистрируем ресурс.
for name, window in [("А", (0, 1)), ("Б", (1, 6))]:  # Первое действие закреплено, второе можно сдвигать.
    request = gds.Action()  # Создаём заявку.
    request.name, request.modes = name, [gds.Mode()]  # Каждое действие имеет длительность одну секунду.
    request.windows = [gds.Window(*window)]  # Ограничиваем разрешённый интервал исполнения.
    problem.add_action(request)  # Добавляем заявку в постановку.
problem.add_precedence(gds.Precedence(0, 1))  # Закрепляем физический порядок двух действий.
problem.set_transition_provider(approximate, "приближение-1")  # Подключаем предварительную оценку.
problem.set_exact_transition_provider(exact, "точная-модель-1")  # Подключаем окончательную модель.
compiled = problem.compile()  # Строим конечную задачу без предположения о достаточности приближения.
assert compiled.evaluate([0, 0], exact=False).feasible  # Начала ноль и один допустимы в предварительной модели.
assert not compiled.evaluate([0, 0]).feasible  # Окончательная модель отвергает отсутствие времени на переход.
schedule = gds.solve_gds(compiled)  # Решатель использует окончательную проверку и запрещает отклонённые назначения.
assert schedule.feasible and schedule.exact  # Принимаем только окончательно проверенный план.
assert schedule.placements[1].start >= 3  # Второе действие теперь начинается после завершения двухсекундного перехода.
print(schedule.stop_reason, schedule.assignment)  # Выводим причину остановки и исправленные индексы размещений.

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
