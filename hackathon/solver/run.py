"""CLI: сборка, планирование, GDE3, события, ветвление и независимая проверка."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackathon.model.operations import EVENT_SCHEMA
from hackathon.model.resource_env import load
from hackathon.solver.engine import Options
from hackathon.solver.project import ScheduleProject, verify_record


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="Скомпилировать построитель и физическую модель C++")
    verify = sub.add_parser("verify", help="Повторить исполнение оригинальным интерпретатором")
    verify.add_argument("--result", type=Path, required=True)
    solve = sub.add_parser("solve", help="Сформировать популяцию GDS, оптимизировать GDE3 и исполнить")
    source = solve.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenario", type=Path)
    source.add_argument("--resume", type=Path, help="Результат исходной ветви")
    solve.add_argument("--branch-step", type=int)
    solve.add_argument("--events", type=Path)
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument("--objective", choices=("critical", "revenue", "pareto"), default="pareto")
    solve.add_argument("--select", choices=("critical", "revenue"), default="critical", help="Выбор исполняемой точки Парето")
    solve.add_argument("--switch", action="append", default=[], metavar="STEP:OBJECTIVE")
    solve.add_argument("--until-step", type=int)
    solve.add_argument("--population", type=int, default=8)
    solve.add_argument("--generations", type=int, default=5, help="0: только исходное поколение корректных GDS-планов")
    solve.add_argument("--attempts", type=int, default=8, help="Число проб разных размещений каждой заявки")
    solve.add_argument("--seed", type=int, default=42)
    solve.add_argument("--differential-weight", type=float, default=1.5)
    solve.add_argument("--crossover", type=float, default=0.8)
    solve.add_argument("--no-plots", action="store_true", help="Сохранить только данные для пакетных экспериментов")
    solve.add_argument("--show-labels", action="store_true", help="Включить подписи временных блоков")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            from hackathon.solver.build import build
            build()
            return
        if args.command == "verify":
            session = verify_record(read_json(args.result))
            print(json.dumps(session.summary(), ensure_ascii=False, indent=2))
            return
        options = Options(args.population, args.generations, args.attempts, args.differential_weight, args.crossover, args.seed)
        scenario = load(args.scenario) if args.scenario else None
        parent = read_json(args.resume) if args.resume else None
        project = ScheduleProject(scenario, parent=parent, branch_step=args.branch_step, objective=args.objective,
                                  options=options, preference=args.select)
        events = []
        if args.events:
            payload = read_json(args.events)
            if payload.get("schema_version") != EVENT_SCHEMA or payload.get("base_scenario") != project.session.initial_scenario["meta"]["id"]:
                raise ValueError("Events schema or base scenario mismatch")
            events = payload["events"]
        switches = []
        for value in args.switch:
            step, objective = value.split(":", 1)
            switches.append({"step": int(step), "objective": objective})
        record = project.run(args.output, events=events, switches=switches, until_step=args.until_step,
                             plots=not args.no_plots, show_labels=args.show_labels)
        print(json.dumps(record["summary"], ensure_ascii=False, indent=2))
        print(f"Результат: {args.output / 'result.json'}")
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
