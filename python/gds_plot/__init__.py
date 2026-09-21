"""Необязательные графики проверенных расписаний; вычисления выполняет пакет gds."""


def gantt(schedule, problem, ax=None):
    """Строит диаграмму Гантта в секундах и возвращает оси Matplotlib.

    Для отклонённого плана диаграмма показывает только диагностические полосы,
    а заголовок явно указывает отсутствие подтверждённого расписания.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    colors = {"action": "tab:blue", "transition": "tab:orange", "idle": "lightgray"}
    for segment in schedule.segments:
        if segment.end <= segment.begin:
            continue
        ax.broken_barh([(segment.begin, segment.end - segment.begin)],
                       (segment.resource - 0.35, 0.7), facecolors=colors[segment.kind])
        if segment.kind == "action":
            ax.text(segment.begin, segment.resource, problem.actions[segment.action].name,
                    va="center", fontsize=8)
    ax.set_yticks(range(len(problem.resources)), [r.name for r in problem.resources])
    ax.set_xlabel("Время, с")
    ax.set_title("Проверенное расписание" if schedule.feasible and schedule.exact else "Предварительный или отклонённый план")
    ax.grid(axis="x", alpha=0.25)
    return ax


def resource_traces(schedule, slot_seconds=1.0, resource=0, ax=None):
    """Показывает норму момента и память на отдельных осях с явными единицами."""
    import math
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    trace = next(item for item in schedule.traces if item.resource == resource)
    times = [(slot + 1) * slot_seconds for slot in range(len(trace.memory))]
    ax.step(times, trace.memory, where="post", label="Занятая память", color="tab:blue")
    ax.set_ylabel("Память, единицы постановки", color="tab:blue")
    momentum_axis = ax.twinx()
    momentum_axis.step(times, [math.hypot(*v) for v in trace.momentum], where="post",
                       label="Норма момента", color="tab:orange")
    momentum_axis.set_ylabel("Момент, единицы постановки", color="tab:orange")
    ax.set_xlabel("Время, с")
    ax.set_title("Ресурсы выбранного аппарата")
    return ax, momentum_axis


def pareto(population, x=0, y=2, ax=None):
    """Показывает проекцию многокритериальной популяции и отдельно отмечает отказы."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    labels = ["Потерянный приоритет", "Простой, с", "Пиковый момент", "Непригодность", "Стоимость переходов"]
    for feasible, marker, label in [(True, "o", "Допустимые"), (False, "x", "С нарушениями")]:
        members = [s for s in population.population if s.feasible == feasible]
        ax.scatter([s.objectives[x] for s in members], [s.objectives[y] for s in members],
                   marker=marker, label=label)
    ax.set_xlabel(labels[x])
    ax.set_ylabel(labels[y])
    ax.set_title("Проекция популяции на два критерия")
    ax.legend()
    return ax


__all__ = ["gantt", "resource_traces", "pareto"]
