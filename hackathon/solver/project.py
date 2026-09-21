"""Исполнение, получение событий и ветвление с неизменяемым прошлым."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import time
import uuid

import numpy as np

from hackathon.model.operations import Session, digest, replay_episode
from . import __version__
from .engine import Network, Options, choose, optimize


def energy_policy(scenario):
    return {"id": "critical_always_reserve_block_start_v1", "comparison": ">=",
            "critical_soc_pct": scenario["model"]["critical_soc_pct"],
            "reserve_soc_pct": scenario["model"]["reserve_soc_pct"],
            "absolute_tolerance_wh": 1e-9,
            "critical_scope": "initial state and both endpoints of every slot; raw energy before clipping",
            "reserve_scope": "start of calibration or first slot of a contiguous job on the same satellite"}


def audit_energy_step(env, actions=None):
    """Независимый контроль неокруглённых состояний исходной Python-модели."""
    for sid, sat in env.sats.items():
        energy = env.state[sid]["energy_wh"]
        critical = sat["capacity_wh"] * env.s["model"]["critical_soc_pct"] / 100
        if energy < critical - 1e-9:
            raise ValueError(f"Critical SOC floor violated: {sid}, boundary {env.k}")
        if actions is None:
            continue
        action = actions.get(sid, {"action": "idle"})
        active = action["action"] != "idle"
        power = (sat["calibration_w"] if action["action"] == "calibrate" else
                 sat[env.jobs[action["job_id"]]["kind"] + "_w"] if active else 0.)
        raw, _, _, _ = env.transition(sid, power)
        if raw < critical - 1e-9:
            raise ValueError(f"Critical SOC floor violated: {sid}, slot {env.k}")
        continuation = action["action"] == "job" and any(
            row["step"] == env.k - 1 and row["satellite_id"] == sid
            and row["executed"] == "job" and row["requested"]["job_id"] == action["job_id"]
            for row in env.trace[-len(env.sats):])
        if active and not continuation and energy < sat["capacity_wh"] * env.s["model"]["reserve_soc_pct"] / 100 - 1e-9:
            raise ValueError(f"Start reserve violated: {sid}, slot {env.k}")


def audit_energy(session):
    # Повторение нужно только для независимой проверки результатов: trace
    # округлён до 6 знаков и не годится для проверки точного равенства порогу.
    replay = Session(session.initial_scenario)
    events, commands = {}, {}
    for event in session.events:
        events.setdefault(event["at_step"], []).append(event)
    for command in session.commands:
        commands.setdefault(command["step"], {})[command["satellite_id"]] = {
            k: v for k, v in command.items() if k not in ("step", "satellite_id")}
    while replay.env.k < session.env.k:
        k = replay.env.k
        for event in events.get(k, []):
            replay.apply_event(event)
        actions = commands.get(k, {})
        audit_energy_step(replay.env, actions)
        replay.advance(actions)
    audit_energy_step(replay.env)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def audit_continuity(session):
    steps = {}
    for row in session.env.trace:
        if row["executed"] == "job":
            steps.setdefault(row["requested"]["job_id"], []).append((row["step"], row["satellite_id"]))
    for jid, rows in steps.items():
        if len({sid for _, sid in rows}) != 1 or [t for t, _ in rows] != list(range(rows[0][0], rows[0][0] + len(rows))):
            raise ValueError(f"Job {jid} is split or migrated")
        job = session.env.jobs[jid]
        if job["completed_step"] is not None and len(rows) != job["work_steps"]:
            raise ValueError(f"Job {jid} has invalid completed work")


def verify_record(record):
    if record.get("schema_version") != "cosmo-B-ops-result-1.0":
        raise ValueError("Unsupported result schema")
    if record.get("initial_scenario_hash", digest(record["initial_scenario"])) != digest(record["initial_scenario"]):
        raise ValueError("Initial scenario hash mismatch")
    session = replay_episode(record["initial_scenario"], record["events"], record["commands"], record["steps_executed"])
    if session.summary() != record["summary"]:
        raise ValueError("Summary differs from independent replay")
    if "trace" in record and session.env.trace != record["trace"]:
        raise ValueError("Trace differs from independent replay")
    if session.summary()["blocked_command_count"]:
        raise ValueError("Result contains rejected commands")
    audit_continuity(session)
    audit_energy(session)
    return session


def verify_candidate(session, network, member):
    """Проверка полного будущего плана оригинальной моделью, включая все состояния."""
    native = network.native.inspect(member["plan"])
    if native["U"]:
        raise ValueError("Candidate has nonzero correctness functional")
    replay = session.fork()
    energy, temp, ages = [], [], []

    def capture():
        audit_energy_step(replay.env)
        energy.append([replay.env.state[s]["energy_wh"] for s in network.satellite_ids])
        temp.append([replay.env.state[s]["temp_c"] for s in network.satellite_ids])
        ages.append([replay.env.state[s]["calibration_age_steps"] for s in network.satellite_ids])

    capture()
    while replay.env.k < network.horizon:
        actions = network.commands(member["plan"], replay.env.k)
        audit_energy_step(replay.env, actions)
        replay.advance(actions)
        capture()
    for key, reference in (("energy_wh", energy), ("temp_c", temp), ("calibration_age_steps", ages)):
        if not np.allclose(np.array(reference).T.ravel(), native[key], atol=1e-10, rtol=1e-12):
            raise ValueError(f"C++/reference mismatch: {key}")
    if any(replay.env.jobs[j["id"]]["completed_step"] != (None if end < 0 else end)
           for j, end in zip(network.jobs, native["completed"])):
        raise ValueError("C++/reference completion mismatch")
    summary = replay.summary()
    if summary["blocked_command_count"] or summary["revenue_usd"] != member["revenue_usd"]:
        raise ValueError("Candidate failed independent execution")
    if summary["critical_jobs_completed_on_time"] != member["critical_completed"]:
        raise ValueError("Candidate priority objective mismatch")
    audit_continuity(replay)
    return {"U": 0, "blocked_commands": 0, "full_horizon_physics_parity": True,
            "critical_floor_all_slots": True, "reserve_at_block_start": True,
            "contiguous_jobs": True, "revenue_usd": summary["revenue_usd"]}


def metrics(session):
    result = session.summary()
    critical_due = result["critical_jobs_due"]
    result["critical_completion_pct"] = (100 * result["critical_jobs_completed_on_time"] / critical_due
                                         if critical_due else None)
    counts = dict.fromkeys(session.env.sats, 0)
    for row in session.env.trace:
        counts[row["satellite_id"]] += row["executed"] == "job"
    result["utilization"] = {sid: count / session.env.k if session.env.k else None for sid, count in counts.items()}
    return result


def metric_history(session):
    replay = Session(session.initial_scenario)
    by_step = {}
    for command in session.commands:
        by_step.setdefault(command["step"], {})[command["satellite_id"]] = {
            k: v for k, v in command.items() if k not in ("step", "satellite_id")}
    events = {}
    for event in session.events:
        events.setdefault(event["at_step"], []).append(event)
    history = []
    while True:
        for event in events.get(replay.env.k, []):
            replay.apply_event(event)
        history.append(metrics(replay))
        if replay.env.k == session.env.k:
            return history
        replay.advance(by_step.get(replay.env.k, {}))


def encode_y(plan, network):
    grid = np.zeros((3 * len(network.satellite_ids), network.horizon), dtype=np.uint8)
    for s in range(len(network.satellite_ids)):
        for t in range(network.start, network.horizon):
            code = plan[s * network.horizon + t]
            if code:
                action = 0 if code == 1 else 1 if network.jobs[code-2]["kind"] == "downlink" else 2
                grid[3*s+action, t] = 1
    return grid


def export_network(directory, network, selected, result, session):
    directory.mkdir(parents=True, exist_ok=True)
    description = copy.deepcopy(network.description)
    b = np.array(description.pop("b"), dtype=np.uint8).reshape(3 * len(network.satellite_ids), network.horizon)
    domains = description.pop("domains")
    offsets = np.cumsum([0, *map(len, domains)], dtype=np.int64)
    placements = np.array([p for domain in domains for p in domain], dtype=np.int32).reshape(-1, 2)
    y = encode_y(selected["plan"], network)
    # Матрица y сохраняет также выполненный префикс, включая частичные задания.
    sid_index = {sid: i for i, sid in enumerate(network.satellite_ids)}
    for row in session.env.trace:
        if row["executed"] == "idle":
            continue
        action = 0 if row["executed"] == "calibrate" else 1 if session.env.jobs[row["requested"]["job_id"]]["kind"] == "downlink" else 2
        y[3*sid_index[row["satellite_id"]]+action, row["step"]] = 1
    # W задаётся точными разреженными факторами: пара действий одного аппарата
    # и кардинальное ограничение downlink. Последнее при limit>1 не квадратично.
    pairs = [(3*s+a, 3*s+c) for s in range(len(network.satellite_ids)) for a, c in ((0, 1), (0, 2), (1, 2))]
    np.savez_compressed(directory / "network.npz", y=y, b=b, W_pairs=np.array(pairs),
                        placement_offsets=offsets, placements=placements,
                        genome=selected["genome"], proposal_U=selected["proposal_U"],
                        **{f"front_plan_{i}": np.array(m["plan"], dtype=np.int32)
                           for i, m in enumerate(result["pareto_front"])})
    description.update({"schema_version": "cosmo-gds-network-1.0", "beta": 1,
                        "primary_y_shape": list(y.shape), "action_order": ["calibrate", "downlink", "relay"],
                        "U_components": ["binary", "shared_capacity", "energy", "thermal", "calibration", "target", "continuity"],
                        "downlink_parallel_limit": session.env.s["model"]["downlink_parallel_limit"],
                        "same_satellite_pair_penalty": 1,
                        "dynamic_guards": "exact full editable horizon replay after every proposal",
                        "energy_policy": energy_policy(session.env.s),
                        "compiled_model_hash": digest(session.env.s), "prefix_commands_hash": digest(session.commands),
                        "state_at_boundary": session.observation(),
                        "grid_gdsw_scope": "one-hot rows and static biases only; dynamic guards require native builder"})
    write_json(directory / "network.json", description)
    write_json(directory / "known_scenario.json", session.env.s)
    write_json(directory / "prefix.json", session.result())
    network.native.save_network(selected["plan"], str(directory / "grid.gdsw"))


class ScheduleProject:
    def __init__(self, scenario=None, *, parent=None, branch_step=None, objective="pareto", options=None, preference="critical"):
        self.options = options or Options()
        self.options.validate()
        self.objective, self.preference = objective, preference
        if objective not in ("critical", "revenue", "pareto") or preference not in ("critical", "revenue"):
            raise ValueError("Invalid objective or Pareto selection preference")
        metadata = {"run_id": str(uuid.uuid4()), "objective": objective, "initial_objective": objective,
                    "algorithm": "block-GDS + GDE3 DE/rand/1/bin", "algorithm_version": __version__,
                    "parameters": {**self.options.__dict__, "pareto_selection": preference},
                    "objective_switches": [], "versions": []}
        if parent is not None:
            verify_record(parent)
            boundary = parent["steps_executed"] if branch_step is None else branch_step
            if type(boundary) is not int or not 0 <= boundary <= parent["steps_executed"]:
                raise ValueError("Branch must lie in the executed parent prefix")
            self.session = replay_episode(parent["initial_scenario"],
                                          [e for e in parent["events"] if e["at_step"] <= boundary],
                                          [c for c in parent["commands"] if c["step"] < boundary], boundary)
            parent_meta = parent.get("run_metadata", {})
            metadata.update({"parent_run_id": parent_meta.get("run_id", digest(parent)), "branch_step": boundary,
                             "parent_prefix_hash": digest(self.session.commands),
                             "parent_objective_switches": [s for s in parent_meta.get("objective_switches", []) if s["step"] <= boundary]})
        else:
            if scenario is None or branch_step is not None:
                raise ValueError("Provide a scenario, or a parent result with an optional branch step")
            self.session = Session(scenario)
        self.session.run_metadata = metadata
        metadata["energy_policy"] = energy_policy(self.session.env.s)
        self.history = metric_history(self.session)
        self.versions = []

    def replan(self, output, reason, *, plots=True):
        started = time.perf_counter()
        version_id = f"v{len(self.versions):03d}_step{self.session.env.k:03d}"
        print(f"{version_id}: {reason}, цель={self.objective}, известных заявок={len(self.session.env.jobs)}", flush=True)
        network = Network(self.session)
        seed = self.options.seed + self.session.env.k
        directory = output / "versions" / version_id
        from .progress_reports import save_generation, export_progress
        revenue_bound = sum(j["value_usd"] for j in network.jobs)
        result = optimize(network, self.objective, replace(self.options, seed=seed),
                          on_generation=lambda snapshot: save_generation(
                              directory, snapshot, objective=self.objective,
                              revenue_bound=revenue_bound, plots=plots))
        selected = choose(result, self.preference)
        checks = [verify_candidate(self.session, network, member) for member in result["pareto_front"]]
        export_network(directory, network, selected, result, self.session)
        forecasts = []
        for member in result["pareto_front"]:
            commands = []
            for step in range(network.start, network.horizon):
                commands.extend(dict(a, step=step, satellite_id=sid) for sid, a in network.commands(member["plan"], step).items())
            forecasts.append({**{k: member[k] for k in ("evaluation", "critical_completed", "critical_total", "critical_pct", "revenue_usd", "jobs_completed")},
                              "commands": commands, "selected": member is selected})
        write_json(directory / "pareto_front.json", {"scope": "forecast using only received events", "points": forecasts})
        optimization = {"options": result["options"], "objective": self.objective, "history": result["history"],
                        "generation_history": result["generation_history"]}
        write_json(directory / "optimization.json", optimization)
        export_progress(directory, optimization, plots=plots)
        version = {"id": version_id, "step": self.session.env.k, "reason": reason, "objective": self.objective,
                   "parent_version": self.versions[-1]["id"] if self.versions else None,
                   "prefix_commands_hash": digest(self.session.commands), "state_hash": self.session.state_digest(),
                   "received_event_ids": [e["id"] for e in self.session.events], "known_jobs": len(network.jobs),
                   "abandoned_jobs": network.description["abandoned_jobs"], "selected_evaluation": selected["evaluation"],
                   "front_size": len(forecasts), "validation": checks, "seconds": time.perf_counter() - started}
        self.versions.append(version)
        self.session.run_metadata["versions"] = self.versions
        write_json(directory / "version.json", version)
        return network, selected

    def run(self, output, *, events=(), switches=(), until_step=None, plots=True, show_labels=False):
        output = Path(output)
        # Результат проекта не перезаписывается случайным повторным запуском.
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        horizon, start = self.session.env.s["time"]["steps"], self.session.env.k
        stop = horizon if until_step is None else until_step
        if type(stop) is not int or not start <= stop <= horizon:
            raise ValueError("Invalid stopping boundary")
        event_map, previous, ids = {}, -1, set()
        received = {e["id"]: e for e in self.session.events}
        for event in events:
            at = event.get("at_step")
            eid = event.get("id")
            if type(at) is not int or not 0 <= at < horizon or at < previous or not isinstance(eid, str) or not eid or eid in ids:
                raise ValueError("Invalid event receipt order, step, or ID")
            previous = at
            ids.add(eid)
            if eid in received:
                if event != received[eid]:
                    raise ValueError("Event conflicts with immutable history")
                continue
            if at < start:
                raise ValueError("Cannot insert an event in the past")
            event_map.setdefault(at, []).append(copy.deepcopy(event))
        switch_map = {}
        for switch in switches:
            step, objective = switch["step"], switch["objective"]
            if type(step) is not int or not start <= step < horizon or step in switch_map or objective not in ("critical", "revenue", "pareto"):
                raise ValueError("Invalid objective switch")
            switch_map[step] = objective
        network = selected = None
        with (output / "metrics.jsonl").open("w", encoding="utf-8") as metric_stream, (output / "resources.jsonl").open("w", encoding="utf-8") as resource_stream:
            for metric in self.history[:-1]:
                metric_stream.write(json.dumps(metric, ensure_ascii=False, allow_nan=False) + "\n")
            for row in self.session.env.trace:
                resource_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            while True:
                k = self.session.env.k
                reasons = []
                for event in event_map.get(k, []):
                    self.session.apply_event(event)
                    reasons.append(event["id"])
                if k in switch_map:
                    self.objective = switch_map[k]
                    self.session.run_metadata["objective"] = self.objective
                    self.session.run_metadata["objective_switches"].append({"step": k, "objective": self.objective})
                    reasons.append("objective_switch")
                # При поступлении заявок на текущей границе меняются знаменатели.
                self.history[-1] = metrics(self.session)
                metric_stream.write(json.dumps(self.history[-1], ensure_ascii=False, allow_nan=False) + "\n")
                metric_stream.flush()
                if k == stop:
                    break
                if selected is None or reasons:
                    network, selected = self.replan(output, ", ".join(reasons) or "initial", plots=plots)
                    write_json(output / "checkpoint.json", self.session.result())
                rows = self.session.advance(network.commands(selected["plan"], k))
                for row in rows:
                    resource_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                resource_stream.flush()
                if any(r["reason"] not in ("accepted", "idle") for r in rows):
                    raise RuntimeError("Execution diverged from validated forecast")
                self.history.append(metrics(self.session))
        record = self.session.result()
        verify_record(record)
        write_json(output / "result.json", record)
        write_json(output / "metrics.json", self.history)
        write_json(output / "project.json", self.session.run_metadata)
        from .reports import export_reports
        export_reports(output, record, self.history, plots=plots, show_labels=show_labels)
        write_json(output / "validation.json", {"independent_replay": True, "contiguous_jobs": True,
                                               "blocked_commands": 0, "metric_boundaries": len(self.history),
                                               "forecast_versions": len(self.versions)})
        return record
