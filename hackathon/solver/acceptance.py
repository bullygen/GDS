"""Все входные сценарии × три цели, затем события и ветвление результата."""
import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackathon.model.resource_env import load
from hackathon.solver.engine import Options
from hackathon.solver.project import ScheduleProject, energy_policy, metric_history, verify_record, write_json


def inspect_network_artifacts(output):
    import numpy as np
    from hackathon.model.operations import digest
    from hackathon.solver.engine import Network
    count = 0
    for directory in sorted((output / "versions").glob("*")):
        prefix = json.loads((directory / "prefix.json").read_text())
        session = verify_record(prefix)
        network = Network(session)
        spec = json.loads((directory / "network.json").read_text())
        if spec.get("energy_policy") != energy_policy(session.env.s):
            raise AssertionError("Saved network is missing the current energy policy")
        forecasts = json.loads((directory / "pareto_front.json").read_text())["points"]
        selected_index = next(i for i, member in enumerate(forecasts) if member["selected"])
        selected = forecasts[selected_index]
        options = json.loads((directory / "optimization.json").read_text())["options"]
        version = json.loads((directory / "version.json").read_text())
        if spec["prefix_commands_hash"] != digest(session.commands) or version["state_hash"] != session.state_digest():
            raise AssertionError("Network anchor does not match the immutable prefix")
        with np.load(directory / "network.npz", allow_pickle=False) as arrays:
            repeated = network.decode(arrays["genome"], options["attempts"])
            if not np.array_equal(repeated["plan"], arrays[f"front_plan_{selected_index}"]):
                raise AssertionError("Saved genome does not reproduce the selected network state")
            y = arrays["y"]
            expected_shape = (3 * len(network.satellite_ids), network.horizon)
            if y.shape != expected_shape or not np.isin(y, [0, 1]).all():
                raise AssertionError("Invalid primary activity grid")
            if (y.reshape(-1, 3, network.horizon).sum(axis=1) > 1).any():
                raise AssertionError("Overlapping actions on a satellite")
            if (y[1::3].sum(axis=0) > session.env.s["model"]["downlink_parallel_limit"]).any():
                raise AssertionError("Downlink cardinality factor is violated")
            expected = np.zeros_like(y)
            index = {sid: i for i, sid in enumerate(network.satellite_ids)}
            for command in [*session.commands, *selected["commands"]]:
                if command["action"] == "idle":
                    continue
                a = 0 if command["action"] == "calibrate" else 1 if session.env.jobs[command["job_id"]]["kind"] == "downlink" else 2
                expected[3 * index[command["satellite_id"]] + a, command["step"]] = 1
            if not np.array_equal(y, expected):
                raise AssertionError("Saved y differs from executed prefix plus selected forecast")
            if not np.array_equal(arrays["b"].ravel(), network.description["b"]):
                raise AssertionError("Saved static masks differ from received environment")
        count += 1
    return count


def inspect_result(output, result, plots):
    replay = verify_record(result)
    history = json.loads((output / "metrics.json").read_text())
    streamed = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    if history != streamed or history != metric_history(replay):
        raise AssertionError("Intermediate metrics differ from independent boundary replay")
    if len(history) != result["steps_executed"] + 1:
        raise AssertionError("Missing metric boundaries")
    if plots:
        expected = ["schedule.png", "metrics.png", "metrics.pdf", "resources.pdf"]
        for filename in expected:
            if not (output / "figures" / filename).is_file():
                raise AssertionError(f"Missing figure {filename}")
        manifest = json.loads((output / "figures/resources/index.json").read_text())
        if len(manifest) != len(result["initial_scenario"]["satellites"]):
            raise AssertionError("Missing satellite resource plots")
        for item in manifest:
            if not (output / "figures/resources" / item["file"]).is_file():
                raise AssertionError("Missing resource PNG")
        for version in (output / "versions").glob("*"):
            for filename in ("pareto.png", "optimization.png", "objective_progress.png"):
                if not (version / filename).is_file():
                    raise AssertionError("Missing version figure")
            optimization = json.loads((version / "optimization.json").read_text())
            snapshots = optimization["generation_history"]
            if [s["generation"] for s in snapshots] != list(range(optimization["options"]["generations"] + 1)):
                raise AssertionError("Missing generation snapshots")
            for snapshot in snapshots:
                prefix = optimization["history"][:snapshot["evaluations"]]
                for key in ("critical_completed", "revenue_usd"):
                    if snapshot["best_so_far"][key] != max(row[key] for row in prefix):
                        raise AssertionError("Progress curve differs from actual evaluations")
                if optimization["objective"] == "pareto":
                    if not (version / "pareto_generations" / f"generation_{snapshot['generation']:04d}.png").is_file():
                        raise AssertionError("Missing Pareto generation image")
    networks = inspect_network_artifacts(output)
    return {"summary": result["summary"], "metric_boundaries": len(history), "network_artifacts_verified": networks,
            "versions": len(result["run_metadata"]["versions"]), "verified": True, "plots": plots,
            "energy_policy": energy_policy(replay.env.s),
            "critical_floor_all_slots": True, "reserve_at_block_start": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=8)
    args = parser.parse_args()
    data = Path(__file__).resolve().parents[1] / "data"
    options = Options(args.population, args.generations, args.attempts)
    options.validate()
    report = {"parameters": options.__dict__, "runs": {}}
    for path in sorted(data.glob("*.json")):
        for objective in ("critical", "revenue", "pareto"):
            name = f"{path.stem}_{objective}"
            output = args.output / name
            started = time.perf_counter()
            result = ScheduleProject(load(path), objective=objective, options=options).run(output, plots=not args.no_plots)
            report["runs"][name] = inspect_result(output, result, not args.no_plots)
            report["runs"][name]["seconds"] = time.perf_counter() - started
            write_json(args.output / "acceptance.json", report)
            print(f"PASS {name}: {report['runs'][name]['seconds']:.2f}s", flush=True)
    events = json.loads((data.parent / "examples/events_demo.json").read_text())["events"]
    output = args.output / "P02_events"
    result = ScheduleProject(load(data / "P02_shift.json"), options=options).run(output, events=events, plots=not args.no_plots)
    report["runs"]["P02_events"] = inspect_result(output, result, not args.no_plots)
    if result["events"] != events or len(result["run_metadata"]["versions"]) != 5:
        raise AssertionError("Not all four events caused replanning")
    plain = json.loads((args.output / "P02_shift_pareto/result.json").read_text())
    for key in ("commands", "trace"):
        if [r for r in plain[key] if r["step"] < 72] != [r for r in result[key] if r["step"] < 72]:
            raise AssertionError("Future events influenced the prefix")
    output = args.output / "P02_branch"
    branch = ScheduleProject(parent=result, branch_step=74, objective="revenue", options=options).run(
        output, events=events, switches=[{"step": 180, "objective": "critical"}], plots=not args.no_plots)
    report["runs"]["P02_branch"] = inspect_result(output, branch, not args.no_plots)
    if [c for c in branch["commands"] if c["step"] < 74] != [c for c in result["commands"] if c["step"] < 74]:
        raise AssertionError("Branch rewrote the executed prefix")
    report["passed"] = True
    write_json(args.output / "acceptance.json", report)
    print(f"PASS: {len(report['runs'])} runs, including events and branching", flush=True)


if __name__ == "__main__":
    main()
