"""Два PDF с итоговыми метриками четырёх сценариев: critical и revenue."""
import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackathon.solver.project import verify_record


SCENARIOS = ("P01_intro", "P02_shift", "P03_energy", "P04_demand")
METRICS = [
    ("revenue_usd", "Выручка, USD", 2),
    ("critical_completion_pct", "Выполнено заданий приоритета 3, %", 3),
    ("planning_seconds", "Планирование с проверкой и сохранением, с", 3),
    ("candidate_seconds", "В том числе: оценка кандидатов GDS, с", 3),
    ("run_seconds", "Полный прогон с отчётами и проверками, с", 3),
    ("steps_executed", "Выполнено шагов", 0),
    ("jobs_total", "Всего известных заданий", 0),
    ("jobs_completed", "Завершено заданий", 0),
    ("jobs_due", "Заданий с наступившим сроком", 0),
    ("jobs_due_missed", "Из них не завершено к сроку", 0),
    ("critical_jobs_due", "Заданий приоритета 3 с наступившим сроком", 0),
    ("critical_jobs_completed_on_time", "Из них выполнено в срок", 0),
    ("blocked_command_count", "Отклонено команд", 0),
    ("below_reserve_satellite_steps", "Заряд ниже резерва, пар «аппарат — шаг»", 0),
    ("critical_soc_satellite_steps", "Заряд ниже критического порога, пар «аппарат — шаг»", 0),
    ("brownout_satellite_steps", "Нехватка питания, пар «аппарат — шаг»", 0),
    ("minimum_soc_pct", "Минимальный заряд за весь прогон, %", 6),
    ("work_steps_in_missed_jobs", "Рабочие шаги в не завершённых к сроку заданиях", 0),
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs(directory, mode):
    acceptance = read_json(directory / "acceptance.json")
    runs = []
    for name in SCENARIOS:
        path = directory / f"{name}_{mode}"
        record = read_json(path / "result.json")
        verify_record(record)
        history = read_json(path / "metrics.json")
        metric = history[-1]
        if record["initial_scenario"]["meta"]["id"] != name or record["run_metadata"]["objective"] != mode:
            raise ValueError(f"Scenario or objective mismatch: {path}")
        if record["steps_executed"] != record["initial_scenario"]["time"]["steps"]:
            raise ValueError(f"The scenario is not fully executed: {path}")
        if any(metric[key] != value for key, value in record["summary"].items()):
            raise ValueError(f"Final metrics differ from the result: {path}")
        due = metric["critical_jobs_due"]
        expected_pct = 100 * metric["critical_jobs_completed_on_time"] / due if due else None
        if metric["critical_completion_pct"] != expected_pct:
            raise ValueError(f"Incorrect priority completion percentage: {path}")
        measured = acceptance["runs"][path.name]
        if measured["summary"] != record["summary"]:
            raise ValueError(f"Timing refers to a different result: {path}")
        satellite_ids = sorted(s["id"] for s in record["initial_scenario"]["satellites"])
        if set(metric["terminal_soc_pct"]) != set(satellite_ids) or set(metric["utilization"]) != set(satellite_ids):
            raise ValueError("Incomplete satellite metrics")
        work = dict.fromkeys(satellite_ids, 0)
        for row in record["trace"]:
            work[row["satellite_id"]] += row["executed"] == "job"
        if any(metric["utilization"][sid] != work[sid] / record["steps_executed"] for sid in satellite_ids):
            raise ValueError("Incorrect utilization metrics")
        versions = record["run_metadata"]["versions"]
        extra = {
            "planning_seconds": sum(v["seconds"] for v in versions),
            "candidate_seconds": sum(row["seconds"] for v in versions
                                     for row in read_json(path / "versions" / v["id"] / "optimization.json")["history"]),
            "run_seconds": measured.get("seconds"),
        }
        covered = {key for key, _, _ in METRICS} | {"terminal_soc_pct", "utilization"}
        if set(metric) - covered:
            raise ValueError(f"Metrics without a table row: {set(metric) - covered}")
        runs.append({"name": name, "values": {**metric, **extra}, "satellite_ids": satellite_ids,
                     "parameters": record["run_metadata"]["parameters"], "run_id": record["run_metadata"]["run_id"]})
    return runs


def number(value, decimals):
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def make_pdf(path, runs, mode):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cosmo-gds-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    blue, light = "#173854", "#edf3f7"
    title = "Приоритет 3" if mode == "critical" else "Выручка"

    def page(subtitle, page_number):
        fig = plt.figure(figsize=(11.69, 8.27))  # A4, альбомная ориентация.
        fig.patch.set_facecolor("white")
        fig.text(0.055, 0.945, f"GDS / GDE3  ·  {title}", fontsize=18, color=blue, weight="bold")
        fig.text(0.055, 0.902, subtitle, fontsize=11, color="#455c6a")
        fig.text(0.055, 0.037, f"Режим: {mode}  ·  Итоговые показатели завершённых прогонов", fontsize=8, color="#536777")
        fig.text(0.945, 0.037, f"{page_number} / 5", ha="right", fontsize=8, color="#536777")
        return fig

    def table(axis, data, labels, widths, *, font=9.1):
        axis.axis("off")
        artist = axis.table(cellText=data, colLabels=labels, colWidths=widths,
                            cellLoc="right", colLoc="center", bbox=[0, 0, 1, 1])
        artist.auto_set_font_size(False)
        artist.set_fontsize(font)
        for (row, column), cell in artist.get_celld().items():
            cell.set_edgecolor("#d4dfe6")
            cell.set_linewidth(0.45)
            cell.PAD = 0.05
            if row == 0:
                cell.set_facecolor(blue)
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor(light if row % 2 else "white")
            if column == 0:
                cell.set_text_props(ha="left")
        return artist

    with PdfPages(path, metadata={"Title": f"Итоговые метрики четырёх сценариев — {mode}",
                                 "Subject": "Выручка, приоритетные задания, время и все итоговые метрики"}) as pdf:
        fig = page("Сравнение четырёх сценариев · все итоговые скалярные метрики", 1)
        rows = [[label, *[number(r["values"][key], decimals) for r in runs]] for key, label, decimals in METRICS]
        artist = table(fig.add_axes([0.055, 0.22, 0.89, 0.64]), rows,
                       ["Показатель", *SCENARIOS], [0.48, 0.13, 0.13, 0.13, 0.13], font=9.0)
        for row in (1, 2, 3):
            for column in range(5):
                artist[row, column].set_text_props(weight="bold")
        notes = [
            "Приоритет 3, % = выполнено в срок / все приоритетные задания с наступившим сроком × 100. Здесь все сроки уже наступили.",
            "Планирование: построение сети, GDE3, независимая проверка и сохранение версии. Оценка кандидатов — часть этого времени.",
            "Полный прогон: сохранённое измерение вместе с исполнением, отчётами и проверками. Дополнительные PDF в него не входят.",
            "Заряд каждого аппарата и его загрузка приведены на следующих страницах. Значения округлены только для отображения.",
        ]
        for y, note in zip((0.172, 0.143, 0.114, 0.085), notes):
            fig.text(0.055, y, note, fontsize=7.8, color="#455c6a")
        pdf.savefig(fig)
        plt.close(fig)

        for page_number, run in enumerate(runs, 2):
            fig = page(f"{run['name']} · итоговый заряд и загрузка каждого аппарата", page_number)
            parameters = run["parameters"]
            config = (f"Популяция: {parameters['population']}   |   Поколения: {parameters['generations']}   |   "
                      f"Проб размещения: {parameters['attempts']}   |   Seed: {parameters['seed']}   |   "
                      f"F: {parameters['differential_weight']}   |   CR: {parameters['crossover']}")
            fig.text(0.055, 0.86, config, fontsize=9, color="#455c6a")
            ids, values = run["satellite_ids"], run["values"]
            midpoint = (len(ids) + 1) // 2
            for left, group in ((0.055, ids[:midpoint]), (0.525, ids[midpoint:])):
                rows = [[sid, number(values["terminal_soc_pct"][sid], 6),
                         number(None if values["utilization"][sid] is None else 100 * values["utilization"][sid], 6)]
                        for sid in group]
                # Одинаковая высота строки для сценариев с 16 и с 48 аппаратами.
                height = 0.026 * (len(rows) + 1)
                table(fig.add_axes([left, 0.815 - height, 0.42, height]), rows,
                      ["Аппарат", "Конечный заряд, %", "Загрузка заданиями, %"], [0.18, 0.39, 0.43], font=9)
            fig.text(0.055, 0.117, "Конечный заряд: terminal_soc_pct. Загрузка: utilization × 100 — доля выполненных шагов заданий за весь прогон.", fontsize=8, color="#455c6a")
            fig.text(0.055, 0.09, f"Источник: {run['name']}_{mode}/result.json, metrics.json, версии и acceptance.json.", fontsize=8, color="#455c6a")
            fig.text(0.055, 0.065, f"Идентификатор прогона: {run['run_id']}", fontsize=7, color="#536777")
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("hackathon/solver/outputs/acceptance"))
    parser.add_argument("--output", type=Path, help="По умолчанию PDF сохраняются в --input")
    args = parser.parse_args()
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)
    for mode in ("critical", "revenue"):
        runs = load_runs(args.input, mode)
        path = output / f"summary_{mode}.pdf"
        make_pdf(path, runs, mode)
        print(path, flush=True)


if __name__ == "__main__":
    main()
