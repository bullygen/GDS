"""Построение блочной GDS-сети и внешняя многокритериальная эволюция GDE3."""
from dataclasses import asdict, dataclass
import copy
import time

import numpy as np

from hackathon.model.resource_env import validate


@dataclass(frozen=True)
class Options:
    population: int = 8
    generations: int = 5
    attempts: int = 8
    differential_weight: float = 0.5
    crossover: float = 0.9
    seed: int = 42

    def validate(self):
        for name in ("population", "generations", "attempts", "seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.population < 4 or self.generations < 0 or self.attempts < 1 or self.seed < 0:
            raise ValueError("Require population >= 4, generations >= 0, attempts >= 1, seed >= 0")
        if not 0 < self.differential_weight <= 2 or not 0 <= self.crossover <= 1:
            raise ValueError("Require 0 < F <= 2 and 0 <= CR <= 1")


class Network:
    """Снимок только известных данных и состояния на неизменяемой границе."""

    def __init__(self, session):
        try:
            from ._native import CompiledNetwork
        except ImportError as exc:
            raise RuntimeError("Сначала выполните python hackathon/solver/run.py build") from exc
        validate(session.env.s)
        self.jobs = list(session.env.jobs.values())
        self.start = session.env.k
        running = {
            r["requested"]["job_id"]: r["satellite_id"]
            for r in session.env.trace
            if r["step"] == self.start - 1 and r["executed"] == "job"
            and session.env.jobs[r["requested"]["job_id"]]["remaining_steps"] > 0
        }
        self.native = CompiledNetwork(session.env.s, session.env.state, self.jobs, self.start, running)
        self.description = self.native.description()
        self.satellite_ids = self.description["satellite_ids"]
        self.horizon = self.description["horizon"]

    def commands(self, plan, step):
        actions = {}
        for s, sid in enumerate(self.satellite_ids):
            code = int(plan[s * self.horizon + step])
            if code == 1:
                actions[sid] = {"action": "calibrate"}
            elif code >= 2:
                actions[sid] = {"action": "job", "job_id": self.jobs[code - 2]["id"]}
        return actions

    def decode(self, genome, attempts):
        started = time.perf_counter()
        result = self.native.decode(genome.tolist(), attempts)
        # Стоимость и приоритет не входят в функционал корректности C++.
        done = [j for j, end in zip(self.jobs, result["completed"]) if 0 <= end <= j["deadline_step"]]
        result["critical_completed"] = sum(j["priority"] == 3 for j in done)
        result["critical_total"] = sum(j["priority"] == 3 for j in self.jobs)
        result["critical_pct"] = (100 * result["critical_completed"] / result["critical_total"]
                                  if result["critical_total"] else None)
        result["revenue_usd"] = round(sum(j["value_usd"] for j in done), 6)
        result["jobs_completed"] = len(done)
        result["genome"] = genome
        result["seconds"] = time.perf_counter() - started
        return result


def dominates(a, b):
    """Минимизация: строгая доминация хотя бы по одной компоненте."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def objectives(member, objective):
    if objective == "critical":
        return (-member["critical_completed"],)
    if objective == "revenue":
        return (-member["revenue_usd"],)
    if objective == "pareto":
        return (-member["critical_completed"], -member["revenue_usd"])
    raise ValueError(f"Unknown objective: {objective}")


def front(members, objective="pareto"):
    values = [objectives(m, objective) for m in members]
    return [m for i, m in enumerate(members) if not any(dominates(v, values[i]) for v in values)]


def crowding(members, objective):
    distance = np.zeros(len(members))
    if len(members) <= 2:
        return np.full(len(members), np.inf)
    values = np.array([objectives(m, objective) for m in members], dtype=float)
    for k in range(values.shape[1]):
        indices = np.argsort(values[:, k], kind="stable")
        low, high = values[indices[0], k], values[indices[-1], k]
        if high == low:
            continue
        distance[indices[0]] = distance[indices[-1]] = np.inf
        for p in range(1, len(indices) - 1):
            distance[indices[p]] += (values[indices[p+1], k] - values[indices[p-1], k]) / (high - low)
    return distance


def truncate(members, size, objective):
    remaining = list(members)
    chosen = []
    while remaining and len(chosen) < size:
        layer = front(remaining, objective)
        if len(layer) > size - len(chosen):
            distances = crowding(layer, objective)
            layer = [layer[i] for i in np.argsort(-distances, kind="stable")[:size - len(chosen)]]
        chosen.extend(layer)
        selected = {id(m) for m in layer}
        remaining = [m for m in remaining if id(m) not in selected]
    return chosen


def optimize(network, objective, options, progress=print, on_generation=None):
    """GDE3: DE/rand/1/bin, попарный отбор, сортировка фронтов и crowding.

    Геном задаёт порядок блочных предложений, начальную пробу размещения и
    разрешение включения. Каждый кандидат восстанавливается охранной GDS-сетью
    до U=0; эволюционный отбор получает уже корректные расписания.
    """
    options.validate()
    objectives({"critical_completed": 0, "revenue_usd": 0}, objective)
    rng = np.random.default_rng(options.seed)
    dimensions = 3 * len(network.jobs)
    history, archive, generation_history = [], [], []

    def evaluate(genome, generation):
        member = network.decode(genome, options.attempts)
        member["evaluation"] = len(history)
        history.append({k: member[k] for k in (
            "evaluation", "U", "proposals", "accepted_blocks", "guard_violations",
            "critical_completed", "critical_pct", "revenue_usd", "jobs_completed", "seconds")})
        history[-1]["generation"] = generation
        return member

    def archive_add(members):
        nonlocal archive
        unique = {}
        for member in [*archive, *members]:
            unique.setdefault((member["critical_completed"], member["revenue_usd"]), member)
        archive = front(list(unique.values()))

    def record_generation(generation):
        def point(member):
            return {k: member[k] for k in ("evaluation", "critical_completed", "critical_total", "critical_pct", "revenue_usd")}

        def maxima(members):
            return {key: max((m[key] for m in members if m[key] is not None), default=None)
                    for key in ("critical_completed", "critical_pct", "revenue_usd")}

        snapshot = {"generation": generation, "evaluations": len(history),
                    "best_so_far": maxima(archive), "population_best": maxima(population),
                    "pareto_front": [point(m) for m in archive],
                    "population_front": [point(m) for m in front(population)]}
        generation_history.append(snapshot)
        if on_generation is not None:
            on_generation(copy.deepcopy(snapshot))

    population = []
    for i in range(options.population):
        genes = rng.random(dimensions)
        if i < 3 and dimensions:
            # Посевы задаются вне U; остальные члены полностью случайны.
            score = np.array([
                (j["priority"] == 3) if i == 0 else
                j["value_usd"] / j["work_steps"] if i == 1 else -j["deadline_step"]
                for j in network.jobs], dtype=float)
            spread = float(np.ptp(score))
            genes[0::3] = (score.max() - score) / spread if spread else 0.5
            genes[1::3] = 0.0
            genes[2::3] = 1.0
        population.append(evaluate(genes, 0))
    archive_add(population)
    record_generation(0)
    progress(f"  GDS: {len(population)} корректных расписаний, {len(archive)} точек фронта", flush=True)
    for generation in range(1, options.generations + 1):
        candidates = []
        for i, target in enumerate(population):
            indices = [j for j in range(len(population)) if j != i]
            a, b, c = (population[j]["genome"] for j in rng.choice(indices, 3, replace=False))
            mutant = np.clip(a + options.differential_weight * (b - c), 0.0, 1.0)
            mask = rng.random(dimensions) < options.crossover
            if dimensions:
                mask[rng.integers(dimensions)] = True
            trial = evaluate(np.where(mask, mutant, target["genome"]), generation)
            tv, pv = objectives(trial, objective), objectives(target, objective)
            if dominates(tv, pv):
                candidates.append(trial)
            elif dominates(pv, tv):
                candidates.append(target)
            else:
                candidates.extend((trial, target))
        archive_add(candidates)
        population = truncate(candidates, options.population, objective)
        record_generation(generation)
        progress(f"  GDE3 {generation}/{options.generations}: "
                 f"priority3={max(m['critical_completed'] for m in population)}, "
                 f"revenue={max(m['revenue_usd'] for m in population):.2f}, "
                 f"Pareto={len(archive)}", flush=True)
    return {"population": population, "pareto_front": archive, "history": history,
            "generation_history": generation_history,
            "options": asdict(options), "objective": objective}


def choose(result, preference="critical"):
    objective = result["objective"]
    members = result["pareto_front"]
    if objective == "revenue" or (objective == "pareto" and preference == "revenue"):
        return min(members, key=lambda m: (-m["revenue_usd"], -m["critical_completed"]))
    return min(members, key=lambda m: (-m["critical_completed"], -m["revenue_usd"]))
