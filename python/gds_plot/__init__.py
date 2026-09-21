"""Необязательные графики проверенных расписаний; вычисления выполняет пакет gds."""


def gantt(schedule, problem, ax=None, *, show_labels=True):
    """Строит диаграмму Гантта в секундах и возвращает оси Matplotlib.

    Для отклонённого плана диаграмма показывает только диагностические полосы,
    а заголовок явно указывает отсутствие подтверждённого расписания.
    Параметр show_labels=False отключает названия действий внутри полос,
    сохраняя подписи осей и названия ресурсов.
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
        if show_labels and segment.kind == "action":
            ax.text(segment.begin, segment.resource, problem.actions[segment.action].name,
                    va="center", fontsize=8)
    ax.set_yticks(range(len(problem.resources)), [r.name for r in problem.resources])
    ax.set_xlabel("Время, с")
    ax.set_title("Проверенное расписание" if schedule.feasible and schedule.exact else "Предварительный или отклонённый план")
    ax.grid(axis="x", alpha=0.25)
    return ax


def resource_traces(schedule, slot_seconds=1.0, resource=None, axes=None, *, problem=None):
    """Строит вертикальные графики загрузки, памяти, нормы момента и запаса.

    По умолчанию выводятся все ресурсы расписания, для каждого — четыре
    отдельные области рисования. Параметр resource выбирает один индекс.
    Возвращается плоский кортеж осей в порядке ресурсов и перечисленных величин.
    Готовые оси можно передать через axes; их число должно совпадать с числом
    графиков. problem необязателен и используется для названий ресурсов.
    """
    import math
    import matplotlib.pyplot as plt

    if not math.isfinite(slot_seconds) or slot_seconds <= 0:
        raise ValueError("Шаг времени должен быть конечным и положительным.")
    traces = [trace for trace in schedule.traces if resource is None or trace.resource == resource]
    if not traces:
        raise ValueError("В расписании нет выбранных ресурсных рядов.")
    count = 4 * len(traces)
    if axes is None:
        _, rows = plt.subplots(count, 1, figsize=(10, 2.5 * count), sharex=True, squeeze=False)
        axes = tuple(rows[:, 0])
    else:
        axes = tuple(axes)
        if len(axes) != count or len({id(axis) for axis in axes}) != count:
            raise ValueError(f"Требуются {count} различных осей для отдельных графиков.")
    resources = problem.resources if problem is not None else None
    for index, trace in enumerate(traces):
        name = resources[trace.resource].name if resources is not None else f"Ресурс {trace.resource}"
        quantities = [
            ("Загрузка", "Единицы ёмкости", trace.load, "tab:blue", 0),
            ("Занятая память", "Единицы памяти", trace.memory, "tab:green", 1),
            ("Норма момента", "Единицы момента", [math.hypot(*v) for v in trace.momentum], "tab:orange", 1),
            ("Накопительный запас", "Единицы запаса", trace.level, "tab:purple", 1),
        ]
        for offset, (title, unit, values, color, time_offset) in enumerate(quantities):
            axis = axes[4 * index + offset]
            times = [(slot + time_offset) * slot_seconds for slot in range(len(values))]
            # Загрузка относится ко всему слоту; завершаем её ступень правой границей.
            if time_offset == 0 and values:
                times.append(len(values) * slot_seconds)
                values = [*values, values[-1]]
            axis.step(times, values, where="post", color=color)
            axis.set_title(f"{name}: {title}")
            axis.set_ylabel(unit)
            axis.set_xlabel("Время, с")
            axis.grid(alpha=0.25)
    return axes


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
