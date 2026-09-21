"""Дополнить сохранённые прогоны снимками поколений, повторив тот же GDE3."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from hackathon.solver.engine import Network, Options, optimize
from hackathon.solver.project import verify_record, write_json
from hackathon.solver.progress_reports import export_progress, save_generation


def refresh(output):
    for directory in sorted((output / "versions").glob("*")):
        saved = json.loads((directory / "optimization.json").read_text())
        session = verify_record(json.loads((directory / "prefix.json").read_text()))
        network = Network(session)
        print(f"Восстановление поколений: {directory}", flush=True)
        result = optimize(network, saved["objective"], Options(**saved["options"]))
        # До записи проверяем, что повторена исходная оптимизация, а не другой поиск.
        def stable(history):
            return [{k: v for k, v in row.items() if k != "seconds"} for row in history]
        if stable(saved["history"]) != stable(result["history"]):
            raise ValueError(f"Optimization replay differs from the saved run: {directory}")
        with np.load(directory / "network.npz", allow_pickle=False) as arrays:
            if len([key for key in arrays.files if key.startswith("front_plan_")]) != len(result["pareto_front"]):
                raise ValueError("Optimization replay changed the Pareto front size")
            for index, member in enumerate(result["pareto_front"]):
                if not np.array_equal(arrays[f"front_plan_{index}"], member["plan"]):
                    raise ValueError("Optimization replay changed a saved Pareto plan")
        saved["generation_history"] = result["generation_history"]
        write_json(directory / "optimization.json", saved)
        for snapshot in result["generation_history"]:
            save_generation(directory, snapshot, objective=saved["objective"],
                            revenue_bound=sum(j["value_usd"] for j in network.jobs))
        export_progress(directory, saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True,
                        help="Каталоги существующих прогонов с подкаталогом versions")
    args = parser.parse_args()
    for output in args.runs:
        if not (output / "versions").is_dir() or not (output / "result.json").is_file():
            parser.error(f"Not a saved run directory: {output}")
        refresh(output)


if __name__ == "__main__":
    main()
