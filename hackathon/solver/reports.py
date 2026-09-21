"""CSV на каждой временной границе и рисунки без совмещения ресурсных рядов."""
import csv
import json
import os
from pathlib import Path


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten(row):
    result = {}
    for key, value in row.items():
        if isinstance(value, dict):
            result.update({f"{key}.{name}": item for name, item in value.items()})
        else:
            result[key] = value
    return result


def plot_schedule(record, ax=None, *, show_labels=False):
    """Одна полоса на аппарат; show_labels управляет только текстом над слотами."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    ids = sorted(s["id"] for s in record["initial_scenario"]["satellites"])
    jobs = {j["id"]: j for j in record["initial_scenario"]["jobs"]}
    for event in record["events"]:
        for job in event.get("jobs", []):
            jobs[job["id"]] = job
    if ax is None:
        _, ax = plt.subplots(figsize=(14, max(4, 0.32 * len(ids))))
    colors = {"calibrate": "#e49a24", "downlink": "#3574af", "relay": "#37986b"}
    rows = {sid: [] for sid in ids}
    for row in record["trace"]:
        if row["executed"] == "idle":
            continue
        jid = row["requested"].get("job_id")
        kind = "calibrate" if row["executed"] == "calibrate" else jobs[jid]["kind"]
        intervals = rows[row["satellite_id"]]
        if intervals and intervals[-1][1] == row["step"] and intervals[-1][2:] == [kind, jid]:
            intervals[-1][1] += 1
        else:
            intervals.append([row["step"], row["step"] + 1, kind, jid])
    for index, sid in enumerate(ids):
        for begin, end, kind, jid in rows[sid]:
            ax.broken_barh([(begin * 300, (end - begin) * 300)], (index - 0.35, 0.7), facecolors=colors[kind])
            if show_labels:
                ax.text(begin * 300, index, jid or "calibrate", fontsize=6, va="center")
    ax.set_yticks(range(len(ids)), ids)
    ax.set_xlim(0, max(300, record["steps_executed"] * 300))
    ax.set_xlabel("Время, с")
    ax.set_title("Исполненное расписание")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(handles=[Patch(color=color, label=kind) for kind, color in colors.items()], loc="upper center", ncols=3)
    return ax


def export_reports(output, record, history, *, plots=True, show_labels=False):
    write_csv(output / "metrics.csv", (flatten(row) for row in history))
    write_csv(output / "commands.csv", ({"step": c["step"], "satellite_id": c["satellite_id"], "action": c["action"], "job_id": c.get("job_id")} for c in record["commands"]))
    write_csv(output / "resources.csv", ({k: v for k, v in row.items() if k != "requested"} for row in record["trace"]))
    for directory in sorted((output / "versions").glob("*")):
        optimization = json.loads((directory / "optimization.json").read_text())
        write_csv(directory / "optimization.csv", (
            {**{k: v for k, v in row.items() if k != "guard_violations"},
             **{f"guard_{i}": v for i, v in enumerate(row["guard_violations"])}}
            for row in optimization["history"]))
    if not plots:
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmo-gds-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    axis = plot_schedule(record, show_labels=show_labels)
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "schedule.png", dpi=120)
    plt.close(axis.figure)

    scalar_keys = [k for k, v in history[0].items() if not isinstance(v, dict)]
    fig, axes = plt.subplots(len(scalar_keys), 1, figsize=(12, 2 * len(scalar_keys)), sharex=True, squeeze=False)
    steps = [r["steps_executed"] for r in history]
    for axis, key in zip(axes[:, 0], scalar_keys):
        values = [np.nan if r[key] is None else r[key] for r in history]
        axis.step(steps, values, where="post")
        axis.set_ylabel(key, fontsize=8)
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Граница шага (300 с)")
    fig.tight_layout()
    fig.savefig(figures / "metrics.png", dpi=120)
    fig.savefig(figures / "metrics.pdf")
    plt.close(fig)

    resource_dir = figures / "resources"
    resource_dir.mkdir(exist_ok=True)
    # Имена файлов основаны на индексе: пользовательский satellite_id не является путём.
    manifest = []
    with PdfPages(figures / "resources.pdf") as pdf:
        for index, sat in enumerate(sorted(record["initial_scenario"]["satellites"], key=lambda s: s["id"])):
            sid = sat["id"]
            trace = [r for r in record["trace"] if r["satellite_id"] == sid]
            curves = [
                ("Энергия, Вт·ч", [0, *[r["step"] + 1 for r in trace]], [sat["capacity_wh"] * sat["initial_soc_pct"] / 100, *[r["energy_after_wh"] for r in trace]]),
                ("Заряд, % (terminal_soc_pct)", steps, [r["terminal_soc_pct"][sid] for r in history]),
                ("Температура, °C", [0, *[r["step"] + 1 for r in trace]], [sat["initial_temp_c"], *[r["temp_after_c"] for r in trace]]),
                ("Возраст калибровки, шаги", [0, *[r["step"] + 1 for r in trace]], [sat["initial_calibration_age_steps"], *[r["calibration_age_steps"] for r in trace]]),
                ("Солнечная мощность, Вт", [r["step"] for r in trace], [r["solar_w"] for r in trace]),
                ("Мощность нагревателя, Вт", [r["step"] for r in trace], [r["heater_w"] for r in trace]),
                ("Общая нагрузка, Вт", [r["step"] for r in trace], [r["load_w"] for r in trace]),
                ("Загрузка заданиями, доля", steps, [np.nan if r["utilization"][sid] is None else r["utilization"][sid] for r in history]),
            ]
            fig, axes = plt.subplots(len(curves), 1, figsize=(11, 15), sharex=True)
            for axis, (label, xs, values) in zip(axes, curves):
                if xs and ("мощность" in label.lower() or label == "Общая нагрузка, Вт"):
                    xs, values = [*xs, xs[-1] + 1], [*values, values[-1]]
                axis.step([x * 300 for x in xs], values, where="post")
                axis.set_ylabel(label, fontsize=8)
                axis.grid(alpha=0.2)
            axes[0].set_title(sid)
            axes[-1].set_xlabel("Время, с")
            fig.tight_layout()
            filename = f"satellite_{index:02d}.png"
            fig.savefig(resource_dir / filename, dpi=110)
            pdf.savefig(fig)
            plt.close(fig)
            manifest.append({"satellite_id": sid, "file": filename, "pdf_page": index + 1})
    (resource_dir / "index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for directory in sorted((output / "versions").glob("*")):
        points = json.loads((directory / "pareto_front.json").read_text())["points"]
        evolution = json.loads((directory / "optimization.json").read_text())["history"]
        fig, axis = plt.subplots(figsize=(7, 5))
        defined = [p for p in points if p["critical_pct"] is not None]
        if defined:
            axis.scatter([p["critical_pct"] for p in defined], [p["revenue_usd"] for p in defined])
        else:
            axis.text(0.5, 0.5, "Нет заявок приоритета 3: доля не определена", ha="center", transform=axis.transAxes)
        axis.set_xlabel("Завершённые заявки приоритета 3, % известных")
        axis.set_ylabel("Выручка, USD")
        axis.set_title("Приближение фронта Парето: прогноз версии")
        axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(directory / "pareto.png", dpi=120)
        plt.close(fig)
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for axis, key in zip(axes, ("critical_completed", "revenue_usd", "U")):
            axis.plot([r["evaluation"] for r in evolution], [r[key] for r in evolution], ".-")
            axis.set_ylabel(key)
            axis.grid(alpha=0.2)
        axes[-1].set_xlabel("Оценённый кандидат GDE3")
        fig.tight_layout()
        fig.savefig(directory / "optimization.png", dpi=120)
        plt.close(fig)
