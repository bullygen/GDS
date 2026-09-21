"""Наблюдение GDE3: лучшие значения и снимки фронта после каждого поколения."""
import os
from pathlib import Path

import numpy as np

from .project import write_json
from .reports import write_csv


def pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmo-gds-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_generation(directory, snapshot, *, objective, revenue_bound, plots=True):
    """Вызывается сразу после поколения, в том числе исходного поколения 0."""
    directory = Path(directory)
    name = f"generation_{snapshot['generation']:04d}"
    write_json(directory / "generations" / f"{name}.json", snapshot)
    if objective != "pareto" or not plots:
        return
    plt = pyplot()
    destination = directory / "pareto_generations"
    destination.mkdir(exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    points = sorted((p for p in snapshot["pareto_front"] if p["critical_pct"] is not None),
                    key=lambda p: p["critical_pct"])
    if points:
        axis.scatter([p["critical_pct"] for p in points], [p["revenue_usd"] for p in points],
                     color="tab:blue", label="Недоминируемые решения всех завершённых поколений")
        axis.legend(fontsize=8, loc="lower left")
    else:
        axis.text(0.5, 0.5, "Нет заявок приоритета 3: доля не определена",
                  ha="center", transform=axis.transAxes)
    # Единый масштаб всех кадров версии, в том числе при потоковой отрисовке.
    axis.set_xlim(0, 100)
    axis.set_ylim(0, max(1.0, revenue_bound * 1.05))
    axis.set_xlabel("Завершённые заявки приоритета 3, % известных")
    axis.set_ylabel("Выручка, USD")
    axis.set_title(f"GDE3: фронт после поколения {snapshot['generation']}\n"
                   f"Оценено кандидатов: {snapshot['evaluations']}; точек: {len(snapshot['pareto_front'])}")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination / f"{name}.png", dpi=140)
    plt.close(fig)


def export_progress(directory, optimization, *, plots=True):
    directory = Path(directory)
    generations = optimization.get("generation_history")
    if not generations:
        return  # Старые результаты можно дополнить через refresh_progress.py.
    write_json(directory / "generation_history.json", generations)
    rows = []
    for snapshot in generations:
        row = {"generation": snapshot["generation"], "evaluations": snapshot["evaluations"],
               "pareto_size": len(snapshot["pareto_front"])}
        for group in ("best_so_far", "population_best"):
            row.update({f"{group}.{key}": value for key, value in snapshot[group].items()})
        rows.append(row)
    write_csv(directory / "generation_history.csv", rows)
    if not plots:
        return
    plt = pyplot()
    mode = optimization["objective"]
    quantities = [("critical_completed", "Завершённые задания приоритета 3")]
    if mode == "revenue":
        quantities = [("revenue_usd", "Выручка, USD")]
    elif mode == "pareto":
        quantities.append(("revenue_usd", "Выручка, USD"))
    fig, axes = plt.subplots(len(quantities), 2, figsize=(13, 3.8 * len(quantities)), squeeze=False)
    history = optimization["history"]
    for (iteration_axis, generation_axis), (key, label) in zip(axes, quantities):
        evaluations = [r["evaluation"] + 1 for r in history]
        values = [r[key] for r in history]
        iteration_axis.scatter(evaluations, values, s=12, color="gray", alpha=0.4, label="Оценённый кандидат")
        iteration_axis.step(evaluations, np.maximum.accumulate(values), where="post",
                            label="Лучшее найденное значение", color="tab:blue")
        iteration_axis.set_xlabel("Номер оценки кандидата GDE3")
        generation_axis.step([g["generation"] for g in generations],
                             [g["best_so_far"][key] for g in generations], where="post", marker="o",
                             label="Лучшее найденное значение", color="tab:blue")
        generation_axis.set_xlabel("Поколение GDE3 (0 — начальная популяция)")
        from matplotlib.ticker import MaxNLocator
        for axis in (iteration_axis, generation_axis):
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.legend(fontsize=8)
    fig.suptitle("Изменение целевой функции" if mode != "pareto" else
                 "Изменение критериев: максимумы могут относиться к разным расписаниям", fontsize=11)
    fig.tight_layout()
    fig.savefig(directory / "objective_progress.png", dpi=140)
    plt.close(fig)
