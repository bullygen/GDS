"""Воспроизводимое измерение построения, GDS-декодирования и проверки физики."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from hackathon.model.operations import Session
from hackathon.model.resource_env import load
from hackathon.solver.engine import Network
from hackathon.solver.project import verify_candidate, write_json


def benchmark(scenario, *, repeats=3, attempts=4):
    session = Session(scenario)
    start = time.perf_counter()
    network = Network(session)
    build_seconds = time.perf_counter() - start
    genome = np.random.default_rng(42).random(3 * len(network.jobs))
    member = network.decode(genome, attempts)
    verify_candidate(session, network, member)
    start = time.perf_counter()
    for _ in range(repeats):
        network.native.inspect(member["plan"])
    native_seconds = (time.perf_counter() - start) / repeats
    commands = [network.commands(member["plan"], k) for k in range(network.horizon)]
    start = time.perf_counter()
    for _ in range(repeats):
        replay = Session(scenario)
        for actions in commands:
            replay.advance(actions)
    reference_seconds = (time.perf_counter() - start) / repeats
    return {"scenario": scenario["meta"]["id"], "satellites": len(network.satellite_ids),
            "steps": network.horizon, "jobs": len(network.jobs), "build_seconds": build_seconds,
            "gds_decode_seconds": member["seconds"], "gds_proposals": member["proposals"],
            "native_inspect_seconds": native_seconds, "reference_session_seconds": reference_seconds,
            "replay_ratio": reference_seconds / native_seconds,
            "completed_jobs": member["jobs_completed"], "full_horizon_parity": True,
            "parameters": {"repeats": repeats, "attempts": attempts, "seed": 42},
            "scope": "native inspect includes boundary arrays; reference Session includes trace and validation; not an isolated arithmetic benchmark"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    results = []
    for path in args.scenario:
        result = benchmark(load(path), repeats=args.repeats)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    write_json(args.output, {"platform": platform.platform(), "python": sys.version,
                             "processor": platform.processor(), "results": results})


if __name__ == "__main__":
    main()
