"""Main simulation loop.

Usage (from repo root):
    python src/simulate.py --scenario S1 --reps 500
    python src/simulate.py --eventstudy --reps 200

Results are written incrementally to output/raw/<scenario>.npz and
output/raw/eventstudy_<scenario>.npz so long runs can be split across
invocations.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dgp import (SCENARIOS, EVENT_STUDY_SCENARIOS, ES_BIN_EDGES, TAU,
                 make_design, simulate_rep)
from estimators import run_estimators, event_study_ddd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "output" / "raw"

# deterministic per-scenario seed offsets (index in SCENARIOS order)
_SEED_BASE = {name: 1000 * (i + 1) for i, name in enumerate(SCENARIOS)}


def run_scenario(name, reps, design):
    cfg = SCENARIOS[name]
    est_names = (["E1", "EB", "CEM"] if cfg["regime"] == "binary"
                 else ["E6"])
    draws = {e: np.full((reps, 4), np.nan) for e in est_names}
    t0 = time.time()
    for r in range(reps):
        rep = simulate_rep(cfg, design, seed=_SEED_BASE[name] * 100000 + r)
        res = run_estimators(rep, design, cfg)
        for e in est_names:
            draws[e][r] = res[e]
        if (r + 1) % 100 == 0:
            print(f"  {name}: {r + 1}/{reps} reps "
                  f"({(time.time() - t0) / (r + 1) * 1000:.0f} ms/rep)",
                  flush=True)
    meta = dict(scenario=name, cfg={k: v for k, v in cfg.items()},
                reps=reps, tau=TAU, phi_true=cfg["phi"],
                runtime_sec=time.time() - t0,
                eb_converged=bool(design.eb_converged))
    arrays = {f"{e}_{k}": draws[e][:, i]
              for e in est_names
              for i, k in enumerate(["phi_hat", "phi_se", "tau_hat", "tau_se"])}
    np.savez(RAW / f"{name}.npz", meta=json.dumps(meta), **arrays)
    print(f"  saved {RAW / (name + '.npz')} "
          f"({meta['runtime_sec']:.1f}s total)")
    return meta


def run_event_study(reps, design):
    bin_edges = np.array(ES_BIN_EDGES)
    n_bins = len(bin_edges) - 1
    for name in EVENT_STUDY_SCENARIOS:
        cfg = SCENARIOS[name]
        coefs = np.full((reps, n_bins), np.nan)
        t0 = time.time()
        for r in range(reps):
            rep = simulate_rep(cfg, design,
                               seed=(_SEED_BASE[name] + 77) * 100000 + r)
            coefs[r] = event_study_ddd(rep, design, cfg["regime"], bin_edges)
        np.savez(RAW / f"eventstudy_{name}.npz",
                 bin_edges=bin_edges, coefs=coefs)
        print(f"  event-study {name}: {reps} reps "
              f"({time.time() - t0:.1f}s), saved.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS))
    ap.add_argument("--eventstudy", action="store_true")
    ap.add_argument("--reps", type=int, default=500)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    design = make_design()
    print(f"design ready (EB converged: {design.eb_converged})")
    if args.eventstudy:
        run_event_study(args.reps, design)
    elif args.scenario:
        run_scenario(args.scenario, args.reps, design)
    else:
        for name in SCENARIOS:
            run_scenario(name, args.reps, design)
        run_event_study(200, design)


if __name__ == "__main__":
    main()
