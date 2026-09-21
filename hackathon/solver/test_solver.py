"""Контрактные проверки по независимому интерпретатору хакатона."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from hackathon.model.operations import Session
from hackathon.solver.engine import Network, Options, choose, crowding, dominates, front, optimize
from hackathon.solver.project import ScheduleProject, audit_continuity, metric_history, verify_candidate, verify_record


DATA = Path(__file__).resolve().parents[1] / "data"


def small(steps=8, satellites=2):
    source = json.loads((DATA / "P01_intro.json").read_text())
    source["time"]["steps"] = steps
    source["satellites"] = source["satellites"][:satellites]
    source["jobs"], source["failures"] = [], []
    source["environment"] = {}
    for sat in source["satellites"]:
        sat.update(capacity_wh=100, initial_soc_pct=80, initial_temp_c=15,
                   base_w=0, heater_w=0, calibration_w=0, downlink_w=0,
                   relay_w=0, initial_calibration_age_steps=0)
        source["environment"][sat["id"]] = {
            "solar_w": [0.] * steps, "thermal_target_c": [15.] * steps,
            "downlink_available": [True] * steps, "relay_available": [True] * steps}
    return source


def job(jid="J1", *, work=2, release=0, deadline=8, priority=3, value=10, kind="relay", eligible=None):
    return {"id": jid, "kind": kind, "release_step": release, "deadline_step": deadline,
            "work_steps": work, "eligible_satellites": eligible or ["S01"], "priority": priority, "value_usd": value}


def quiet(*args, **kwargs):
    pass


@pytest.mark.parametrize("name", ["P01_intro", "P02_shift", "P03_energy", "P04_demand"])
def test_cpp_full_horizon_parity_on_supplied_scenarios(name):
    session = Session(json.loads((DATA / f"{name}.json").read_text()))
    network = Network(session)
    member = network.decode(np.random.default_rng(7).random(3 * len(network.jobs)), 4)
    assert member["jobs_completed"] > 0
    assert verify_candidate(session, network, member)["full_horizon_physics_parity"]


@pytest.mark.parametrize("initial,solar,payload,temperature", [
    (30., 20., 20., 15.),  # Равенство резерву до и после операции допустимо.
    (30., 0., 20., 15.),   # После начала разрешено уйти ниже reserve, сохранив critical.
    (80., 100., 0., -2.),  # Вне диапазона зарядки избыток энергии не накапливается.
    (20., 0., 20., 15.),  # Простой на critical допустим, действие ниже reserve — нет.
    (80., 0., 0., 45.),
])
def test_physics_boundaries_against_reference(initial, solar, payload, temperature):
    scenario = small(steps=2, satellites=1)
    sat = scenario["satellites"][0]
    sat.update(initial_soc_pct=initial, initial_temp_c=temperature, calibration_w=payload)
    scenario["environment"]["S01"]["solar_w"] = [solar] * 2
    scenario["environment"]["S01"]["thermal_target_c"] = [temperature] * 2
    session = Session(scenario)
    network = Network(session)
    result = network.native.inspect([1, 0])
    session.advance({"S01": {"action": "calibrate"}})
    session.advance({})
    assert result["energy_wh"][-1] == pytest.approx(session.env.state["S01"]["energy_wh"], abs=1e-10)
    assert result["temp_c"][-1] == pytest.approx(session.env.state["S01"]["temp_c"], abs=1e-10)
    assert (result["U"] == 0) == (session.summary()["blocked_command_count"] == 0)


@pytest.mark.parametrize("initial", [20., 20. - 1e-6])
def test_critical_floor_includes_initial_boundary_even_if_charging(initial):
    scenario = small(steps=1, satellites=1)
    scenario["satellites"][0]["initial_soc_pct"] = initial
    scenario["environment"]["S01"]["solar_w"] = [120.]
    session = Session(scenario)
    if initial < 20:
        with pytest.raises(ValueError, match="Initial state below critical"):
            Network(session)
        session.advance({})
        with pytest.raises(ValueError, match="Critical SOC floor"):
            verify_record(session.result())
    else:
        assert Network(session).native.inspect([0])["U"] == 0


@pytest.mark.parametrize("initial", [25., 25. - 1e-6])
def test_idle_terminal_floor_and_unavailable_satellite(initial):
    scenario = small(steps=1, satellites=1)
    scenario["model"]["discharge_efficiency"] = 1.
    scenario["satellites"][0].update(initial_soc_pct=initial, base_w=60.)
    scenario["failures"] = [{"satellite_id": "S01", "start_step": 0, "end_step": 1}]
    session = Session(scenario)
    if initial < 25:
        with pytest.raises(ValueError, match="No admissible idle baseline"):
            Network(session)
    else:
        result = Network(session).native.inspect([0])
        assert result["U"] == 0 and result["energy_wh"][-1] == 20.
    session.advance({})
    if initial < 25:
        with pytest.raises(ValueError, match="Critical SOC floor"):
            verify_record(session.result())
        with pytest.raises(ValueError, match="Initial state below critical"):
            Network(session)  # Проверяется и нулевой остаточный горизонт.
    else:
        verify_record(session.result())
        assert Network(session).native.inspect([0])["U"] == 0


@pytest.mark.parametrize("action", ["calibrate", "downlink", "relay"])
@pytest.mark.parametrize("initial", [30., 30. - 1e-6])
def test_reserve_only_at_start_of_each_action_with_inclusive_threshold(action, initial):
    scenario = small(steps=1, satellites=1)
    scenario["model"]["discharge_efficiency"] = 1.
    scenario["satellites"][0].update(initial_soc_pct=initial, calibration_w=60., downlink_w=60., relay_w=60.)
    if action != "calibrate":
        scenario["jobs"] = [job(work=1, deadline=1, kind=action)]
    code = 1 if action == "calibrate" else 2
    result = Network(Session(scenario)).native.inspect([code])
    assert (result["U"] == 0) == (initial == 30.)
    assert result["energy_wh"][-1] == pytest.approx(25. if initial == 30. else initial)


@pytest.mark.parametrize("kind", ["downlink", "relay"])
def test_job_continues_below_reserve_including_replanning_boundary(kind):
    scenario = small(steps=2, satellites=1)
    scenario["model"]["discharge_efficiency"] = 1.
    scenario["satellites"][0].update(initial_soc_pct=30., downlink_w=60., relay_w=60.)
    scenario["jobs"] = [job(work=2, deadline=2, kind=kind)]
    session = Session(scenario)
    network = Network(session)
    result = network.native.inspect([2, 2])
    assert result["U"] == 0 and result["energy_wh"] == [30., 25., 20.]
    member = network.decode(np.array([0., 0., 1.]), 8)
    assert member["jobs_completed"] == 1
    assert verify_candidate(session, network, member)["reserve_at_block_start"]
    session.advance({"S01": {"action": "job", "job_id": "J1"}})
    resumed = Network(session)
    member = resumed.decode(np.array([0., 0., 1.]), 8)
    assert member["jobs_completed"] == 1
    verify_candidate(session, resumed, member)
    session.advance({"S01": {"action": "job", "job_id": "J1"}})
    verify_record(session.result())


@pytest.mark.parametrize("next_action", ["calibrate", "new_job", "interrupted_job"])
def test_below_reserve_cannot_start_new_or_interrupted_action(next_action):
    scenario = small(steps=3, satellites=1)
    scenario["model"]["discharge_efficiency"] = 1.
    scenario["satellites"][0].update(initial_soc_pct=30., relay_w=60.)
    scenario["jobs"] = [job(work=2 if next_action == "interrupted_job" else 1, deadline=3)]
    if next_action == "new_job":
        scenario["jobs"].append(job("J2", work=1, deadline=3))
    code = {"calibrate": 1, "new_job": 3, "interrupted_job": 2}[next_action]
    result = Network(Session(scenario)).native.inspect([2, 0, code])
    assert result["violations"][2] > 0
    session = Session(scenario)
    session.advance({"S01": {"action": "job", "job_id": "J1"}})
    session.advance({})
    action = {"action": "calibrate"} if code == 1 else {"action": "job", "job_id": "J2" if code == 3 else "J1"}
    assert session.advance({"S01": action})[0]["reason"] == "energy_reserve"


def test_continuing_job_cannot_cross_critical_floor():
    scenario = small(steps=3, satellites=1)
    scenario["model"]["discharge_efficiency"] = 1.
    scenario["satellites"][0].update(initial_soc_pct=30., relay_w=60.)
    scenario["jobs"] = [job(work=3, deadline=3)]
    session = Session(scenario)
    network = Network(session)
    result = network.native.inspect([2, 2, 2])
    assert result["energy_wh"] == [30., 25., 20., 15.]
    assert result["violations"][2] > 0
    assert network.decode(np.array([0., 0., 1.]), 8)["jobs_completed"] == 0
    for _ in range(3):
        session.advance({"S01": {"action": "job", "job_id": "J1"}})
    with pytest.raises(ValueError, match="Critical SOC floor"):
        verify_record(session.result())


@pytest.mark.parametrize("reserve,critical,completed", [(40., 10., 1), (41., 10., 0), (40., 11., 0)])
def test_energy_thresholds_come_from_scenario_model(reserve, critical, completed):
    scenario = small(steps=3, satellites=1)
    scenario["model"].update(reserve_soc_pct=reserve, critical_soc_pct=critical, discharge_efficiency=1.)
    scenario["satellites"][0].update(capacity_wh=200., initial_soc_pct=40., relay_w=240.)
    scenario["jobs"] = [job(work=3, deadline=3)]
    session = Session(scenario)
    network = Network(session)
    member = network.decode(np.array([0., 0., 1.]), 8)
    assert member["jobs_completed"] == completed
    verify_candidate(session, network, member)
    if completed:
        assert network.native.inspect(member["plan"])["energy_wh"] == [80., 60., 40., 20.]


def test_action_rejected_for_delayed_critical_deficit_during_idle():
    scenario = small(steps=4, satellites=1)
    scenario["model"]["discharge_efficiency"] = 1.
    scenario["satellites"][0].update(initial_soc_pct=50., base_w=60., relay_w=132.)
    scenario["jobs"] = [job(work=1, deadline=1)]
    session = Session(scenario)
    network = Network(session)
    invalid = network.native.inspect([2, 0, 0, 0])
    assert invalid["energy_wh"][1] == 34.  # Всё действие выше reserve=30.
    assert invalid["energy_wh"][-1] == 19.  # Нарушение лишь в будущем простое.
    assert invalid["violations"][2] > 0
    member = network.decode(np.array([0., 0., 1.]), 8)
    assert member["jobs_completed"] == 0 and member["U"] == 0
    assert verify_candidate(session, network, member)["critical_floor_all_slots"]
    session.advance({"S01": {"action": "job", "job_id": "J1"}})
    for _ in range(3):
        session.advance({})
    assert session.summary()["blocked_command_count"] == 0  # Старый oracle пропускает.
    with pytest.raises(ValueError, match="Critical SOC floor"):
        verify_record(session.result())


def test_running_job_abandoned_if_continuation_causes_later_idle_deficit():
    scenario = small(steps=7, satellites=1)
    scenario["model"]["discharge_efficiency"] = 1.
    scenario["satellites"][0].update(initial_soc_pct=64., base_w=60., relay_w=60.)
    scenario["jobs"] = [job(work=2, deadline=2)]
    session = Session(scenario)
    session.advance({"S01": {"action": "job", "job_id": "J1"}})
    network = Network(session)
    member = network.decode(np.array([0., 0., 1.]), 8)
    assert member["U"] == 0 and member["jobs_completed"] == 0
    assert network.native.inspect(member["plan"])["energy_wh"][-1] == 24.
    verify_candidate(session, network, member)


def test_release_deadline_inclusive_and_calibration_expiry():
    scenario = small(steps=4, satellites=1)
    scenario["jobs"] = [job(work=2, release=2, deadline=4)]
    scenario["model"]["calibration_valid_steps"] = 2
    network = Network(Session(scenario))
    assert network.native.inspect([0, 0, 2, 2])["violations"][4] > 0
    assert network.native.inspect([0, 1, 2, 2])["U"] == 0
    candidate = network.decode(np.array([0., 0., 1.]), 8)
    assert candidate["critical_completed"] == 1


def test_shared_capacity_and_continuity_are_real_guards():
    scenario = small(steps=4)
    scenario["model"]["downlink_parallel_limit"] = 1
    scenario["jobs"] = [job("A", work=1, deadline=4, kind="downlink"),
                        job("B", work=1, deadline=4, kind="downlink", eligible=["S02"])]
    assert Network(Session(scenario)).native.inspect([2, 0, 0, 0, 3, 0, 0, 0])["violations"][1] == 1
    scenario["jobs"] = [job(work=2, deadline=4, eligible=["S01", "S02"])]
    net = Network(Session(scenario))
    assert net.native.inspect([2, 0, 2, 0, 0, 0, 0, 0])["violations"][6] > 0
    assert net.native.inspect([2, 0, 0, 0, 0, 2, 0, 0])["violations"][6] > 0


def test_three_modes_and_known_tradeoff():
    scenario = small(steps=2, satellites=1)
    scenario["jobs"] = [job("critical", work=2, deadline=2, priority=3, value=1),
                        job("money", work=2, deadline=2, priority=1, value=100)]
    network = Network(Session(scenario))
    options = Options(population=4, generations=3, attempts=4)
    for mode in ("critical", "revenue", "pareto"):
        result = optimize(network, mode, options, progress=quiet)
        assert all(m["U"] == 0 for m in result["population"])
        assert len(result["history"]) == options.population * (options.generations + 1)
        selected = choose(result)
        if mode == "critical":
            assert selected["critical_completed"] == 1
        elif mode == "revenue":
            assert selected["revenue_usd"] == 100
        else:
            assert {(m["critical_completed"], m["revenue_usd"]) for m in result["pareto_front"]} == {(1, 1), (0, 100)}


def test_front_and_crowding_keep_extremes():
    members = [{"critical_completed": c, "revenue_usd": r} for c, r in [(0, 100), (1, 50), (2, 0), (0, 0)]]
    assert front(members) == members[:3]
    distances = crowding(members[:3], "pareto")
    assert np.isinf(distances[0]) and np.isinf(distances[2]) and np.isfinite(distances[1])
    assert not dominates((1, 2), (1, 2))


def test_event_delivery_has_no_foresight_and_preserves_prefix(tmp_path):
    scenario = small()
    scenario["jobs"] = [job(work=3)]
    event = {"id": "E1", "at_step": 4, "type": "add_jobs", "jobs": [job("new", work=2, release=4, priority=3)]}
    options = Options(population=4, generations=1, attempts=4)
    plain = ScheduleProject(scenario, options=options).run(tmp_path / "plain", plots=False)
    changed = ScheduleProject(scenario, options=options).run(tmp_path / "event", events=[event], plots=False)
    assert [c for c in plain["commands"] if c["step"] < 4] == [c for c in changed["commands"] if c["step"] < 4]
    assert [r for r in plain["trace"] if r["step"] < 4] == [r for r in changed["trace"] if r["step"] < 4]
    prefix = json.loads((tmp_path / "event/versions/v000_step000/prefix.json").read_text())
    assert prefix["events"] == [] and len(prefix["initial_scenario"]["jobs"]) == 1
    assert changed["summary"]["jobs_total"] == 2
    assert len(changed["run_metadata"]["versions"]) == 2
    early = ScheduleProject(scenario, options=options).run(tmp_path / "early", events=[event], until_step=2, plots=False)
    assert early["events"] == []


def test_outage_abandons_partial_job_without_rewriting_or_migration(tmp_path):
    scenario = small()
    scenario["jobs"] = [job(work=4, eligible=["S01", "S02"])]
    parent = Session(scenario)
    for _ in range(2):
        parent.advance({"S01": {"action": "job", "job_id": "J1"}})
    options = Options(population=4, generations=0, attempts=2)
    project = ScheduleProject(parent=parent.result(), options=options)
    result = project.run(tmp_path / "outage", events=[{"id": "O", "at_step": 2, "type": "satellite_outage", "satellite_ids": ["S01"], "end_step": 6}], plots=False)
    assert result["commands"] == parent.commands
    assert result["summary"]["work_steps_in_missed_jobs"] == 2
    assert result["summary"]["jobs_due_missed"] == 1
    assert result["run_metadata"]["versions"][0]["abandoned_jobs"] == ["J1"]
    assert result["run_metadata"]["branch_step"] == 2


def test_branch_preserves_running_block_and_objective_switch(tmp_path):
    scenario = small()
    scenario["jobs"] = [job(work=4)]
    session = Session(scenario)
    session.advance({"S01": {"action": "job", "job_id": "J1"}})
    project = ScheduleProject(parent=session.result(), options=Options(4, 0, 2))
    result = project.run(tmp_path / "branch", switches=[{"step": 3, "objective": "revenue"}], plots=False)
    assert result["commands"][:1] == session.commands
    assert result["summary"]["jobs_completed"] == 1
    assert result["run_metadata"]["objective_switches"] == [{"step": 3, "objective": "revenue"}]
    assert result["run_metadata"]["versions"][1]["step"] == 3


def test_zero_denominators_and_no_repricing_or_denominator_deletion(tmp_path):
    scenario = small(steps=4, satellites=1)
    scenario["jobs"] = [job("impossible", work=2, deadline=4, value=777)]
    scenario["environment"]["S01"]["relay_available"] = [False] * 4
    project = ScheduleProject(scenario, options=Options(4, 0, 2))
    record = project.run(tmp_path / "metrics", plots=False)
    history = json.loads((tmp_path / "metrics/metrics.json").read_text())
    assert history[0]["critical_completion_pct"] is None
    assert history[0]["utilization"]["S01"] is None
    assert history[-1]["critical_completion_pct"] == 0
    assert history[-1]["critical_jobs_due"] == 1
    assert history[-1]["revenue_usd"] == 0
    assert record["initial_scenario"]["jobs"][0]["value_usd"] == 777
    assert history == metric_history(verify_record(record))
    damaged = copy.deepcopy(record)
    damaged["summary"]["revenue_usd"] = 777
    with pytest.raises(ValueError, match="Summary"):
        verify_record(damaged)


def test_close_downlink_and_boundary_event_receipt(tmp_path):
    scenario = small(steps=4, satellites=1)
    project = ScheduleProject(scenario, options=Options(4, 0, 2))
    event = {"id": "close", "at_step": 2, "type": "close_downlink", "satellite_ids": ["S01"], "end_step": 4}
    record = project.run(tmp_path / "boundary", events=[event], until_step=2, plots=False)
    assert record["events"] == [event]
    assert verify_record(record).env.s["environment"]["S01"]["downlink_available"] == [True, True, False, False]


def test_invalid_inputs_fail_and_result_directory_is_preserved(tmp_path):
    with pytest.raises(ValueError):
        Options(population=3).validate()
    options = Options(4, 0, 2)
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep").write_text("untouched")
    with pytest.raises(ValueError, match="not empty"):
        ScheduleProject(small(), options=options).run(target)
    assert (target / "keep").read_text() == "untouched"
    with pytest.raises(ValueError, match="Invalid event"):
        ScheduleProject(small(), options=options).run(tmp_path / "bad", events=[{"id": "bad", "at_step": -1}])


def test_schedule_label_keyword():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hackathon.solver.reports import plot_schedule
    scenario = small()
    scenario["jobs"] = [job(work=1)]
    session = Session(scenario)
    session.advance({"S01": {"action": "job", "job_id": "J1"}})
    hidden = plot_schedule(session.result(), show_labels=False)
    shown = plot_schedule(session.result(), show_labels=True)
    assert not hidden.texts
    assert [text.get_text() for text in shown.texts] == ["J1"]
    plt.close("all")


def test_generation_snapshots_record_actual_past_and_best_values():
    class EvolvingNetwork:
        jobs = []
        evaluated = 0

        def decode(self, genome, attempts):
            # В первом поколении доминирующая новая точка удаляет старые.
            scores = [(0, 10), (1, 8), (0, 9), (1, 5), (2, 12), (1, 11), (0, 3), (2, 9)]
            c, r = scores[self.evaluated]
            self.evaluated += 1
            return {"genome": genome, "critical_completed": c, "critical_total": 2,
                    "critical_pct": 50 * c, "revenue_usd": r, "jobs_completed": c,
                    "U": 0, "proposals": 1, "accepted_blocks": 1,
                    "guard_violations": [0] * 7, "seconds": 0}

    callbacks = []
    def observe(snapshot):
        callbacks.append(copy.deepcopy(snapshot))
        snapshot["pareto_front"].clear()  # Наблюдатель не должен менять историю поиска.

    result = optimize(EvolvingNetwork(), "pareto", Options(4, 1, 1), progress=quiet, on_generation=observe)
    snapshots = result["generation_history"]
    assert snapshots == callbacks
    assert [s["generation"] for s in snapshots] == [0, 1]
    assert [s["evaluations"] for s in snapshots] == [4, 8]
    assert {(p["critical_completed"], p["revenue_usd"]) for p in snapshots[0]["pareto_front"]} == {(0, 10), (1, 8)}
    assert {(p["critical_completed"], p["revenue_usd"]) for p in snapshots[1]["pareto_front"]} == {(2, 12)}
    for snapshot in snapshots:
        prefix = result["history"][:snapshot["evaluations"]]
        for key in ("critical_completed", "critical_pct", "revenue_usd"):
            assert snapshot["best_so_far"][key] == max(row[key] for row in prefix)
        assert all(p["evaluation"] < snapshot["evaluations"] for p in snapshot["pareto_front"])


@pytest.mark.parametrize("mode,generations,plots", [("critical", 1, True), ("revenue", 0, True), ("pareto", 2, True), ("pareto", 1, False)])
def test_generation_report_files(tmp_path, mode, generations, plots):
    scenario = small(steps=2, satellites=1)
    scenario["jobs"] = [job(work=2, deadline=2)]
    output = tmp_path / mode
    ScheduleProject(scenario, objective=mode, options=Options(4, generations, 1)).run(output, plots=plots)
    directory = output / "versions/v000_step000"
    optimization = json.loads((directory / "optimization.json").read_text())
    history = json.loads((directory / "generation_history.json").read_text())
    assert history == optimization["generation_history"]
    assert len(history) == generations + 1
    assert len(list((directory / "generations").glob("*.json"))) == generations + 1
    assert (directory / "objective_progress.png").exists() == plots
    assert len(list((directory / "pareto_generations").glob("*.png"))) == (generations + 1 if mode == "pareto" and plots else 0)
    assert (directory / "generation_history.csv").is_file()
