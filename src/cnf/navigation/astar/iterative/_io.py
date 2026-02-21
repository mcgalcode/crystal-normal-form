"""I/O utilities for iterative A* search results."""

import json
from pathlib import Path


def write_round_json(output_dir, round_num, round_data):
    """Write a single round's results to rounds/round_NNN.json."""
    round_path = Path(output_dir) / "rounds" / f"round_{round_num:03d}.json"
    with open(round_path, "w") as f:
        json.dump(round_data, f, indent=2)


def write_energy_cache(output_dir, cache):
    """Write energy cache as {string_key: energy} JSON."""
    serializable = {str(k): v for k, v in cache.items()}
    cache_path = Path(output_dir) / "energy_cache.json"
    with open(cache_path, "w") as f:
        json.dump(serializable, f, indent=2)


def write_manifest(output_dir, params, result, timing):
    """Write manifest.json with parameters, results, and timing."""
    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"parameters": params, "result": result, "timing": timing}, f, indent=2)


def serialize_result(r):
    """Make a result dict JSON-serializable."""
    s = {"ceiling": r["ceiling"], "found": r["found"],
         "iterations": r["iterations"]}
    if r["found"]:
        s["barrier"] = r["barrier"]
        s["path_length"] = r["path_length"]
        s["path"] = [list(pt) for pt in r["path"]]
        s["energies"] = r["energies"]
    return s


def ceiling_params_dict(
    xi, delta, step_per_atom, num_ceilings, attempts_per_ceiling,
    max_passes, max_sweep_rounds,
    xi_factor, delta_factor, dropout, min_dropout,
    beam_width, heuristic_mode, heuristic_weight, n_workers,
    start_cnfs, goal_cnfs, relax_endpoints=False,
):
    """Build params dict for manifest.json."""
    return {
        "algorithm": "ceiling_barrier_search",
        "xi_initial": xi, "delta_initial": delta,
        "step_per_atom": step_per_atom,
        "num_ceilings": num_ceilings,
        "attempts_per_ceiling": attempts_per_ceiling,
        "max_passes": max_passes,
        "max_sweep_rounds": max_sweep_rounds,
        "xi_factor": xi_factor, "delta_factor": delta_factor,
        "dropout": dropout, "min_dropout": min_dropout,
        "beam_width": beam_width,
        "heuristic_mode": heuristic_mode,
        "heuristic_weight": heuristic_weight,
        "n_workers": n_workers,
        "relax_endpoints": relax_endpoints,
        "elements": start_cnfs[0].elements if start_cnfs else [],
        "start_cnf_coords": [list(c.coords) for c in start_cnfs],
        "goal_cnf_coords": [list(c.coords) for c in goal_cnfs],
    }
