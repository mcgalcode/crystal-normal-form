from ..crystal_normal_form import CrystalNormalForm
from ..calculation.base_calculator import BaseCalculator
from ..calculation.grace import GraceCalculator
from ..navigation import find_neighbors

from dataclasses import dataclass
import heapq
import os
import pickle
import random

DEFAULT_CALC = GraceCalculator()


@dataclass
class WaterfillState:
    frontier: list          # heap of (energy, counter, cnf)
    seen: set               # all discovered CNFs
    explored_energies: list # energies of explored points
    counter: int            # tie-breaker counter
    goal_found: CrystalNormalForm | None
    iteration: int
    dropped: set | None     # dropped CNFs (dropout)
    node_energies: dict | None  # graph tracking
    edge_set: set | None        # graph tracking


def _checkpoint_bg(state, path):
    """Fork a child process to serialize the checkpoint via copy-on-write.

    The parent returns immediately. The child inherits a frozen snapshot
    of the in-memory state (via OS-level COW pages), pickles it to a temp
    file, then atomically renames it into place.
    """
    tmp_path = path + ".tmp"
    pid = os.fork()
    if pid == 0:
        # Child process
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(state, f)
            os.replace(tmp_path, path)
        finally:
            os._exit(0)
    return pid


def _reap_checkpoint(pid):
    """Non-blocking check if a background checkpoint process finished."""
    if pid is None:
        return None
    result = os.waitpid(pid, os.WNOHANG)
    if result == (0, 0):
        return pid   # still running
    return None      # finished


def waterfill(start_cnfs: list[CrystalNormalForm],
              goal_cnfs: list[CrystalNormalForm],
              energy_calc: BaseCalculator = None,
              max_iters: int = 10_000,
              track_graph: bool = False,
              dropout: float = 0.0,
              checkpoint_path: str = None,
              checkpoint_interval: int = 500,
              resume: str = None,
              batch_size: int = 1):

    if energy_calc is None:
        energy_calc = DEFAULT_CALC

    goal_set = set(goal_cnfs)

    if resume:
        with open(resume, "rb") as f:
            state = pickle.load(f)
        frontier = state.frontier
        seen = state.seen
        explored_energies = state.explored_energies
        counter = state.counter
        goal_found = state.goal_found
        start_iteration = state.iteration
        dropped = state.dropped
        node_energies = state.node_energies
        edge_set = state.edge_set
        print(f"Resumed from checkpoint at iteration {start_iteration} "
              f"(frontier: {len(frontier)}, explored: {len(seen)})")
    else:
        explored_energies = []
        frontier = []
        seen = set()
        counter = 0
        dropped = set() if dropout > 0 else None
        node_energies = {} if track_graph else None
        edge_set = set() if track_graph else None

        start_energies = energy_calc.calculate_energies_batch(start_cnfs)
        for cnf, energy in zip(start_cnfs, start_energies):
            heapq.heappush(frontier, (energy, counter, cnf))
            counter += 1
            seen.add(cnf)
            if track_graph:
                node_energies[cnf] = energy

        goal_found = None
        start_iteration = 0

    # Track total points explored for max_iters and checkpointing
    points_explored = 0
    last_checkpoint = start_iteration
    checkpoint_pid = None

    while points_explored < max_iters:
        if goal_found is not None:
            print(f"Found goal at point {start_iteration + points_explored}")
            break

        if not frontier:
            print(f"Frontier exhausted at point {start_iteration + points_explored}")
            break

        # Pop up to batch_size points from frontier
        n_to_pop = min(batch_size, max_iters - points_explored, len(frontier))
        batch = []
        for _ in range(n_to_pop):
            energy, _, pt = heapq.heappop(frontier)
            batch.append((energy, pt))
            explored_energies.append(energy)

        points_explored += len(batch)
        total = start_iteration + points_explored
        max_e = max(e for e, _ in batch)
        min_e = min(e for e, _ in batch)

        if batch_size == 1:
            print(f"Iter {total - 1}: exploring point with energy {min_e:.4f} eV, "
                  f"frontier size: {len(frontier)}, explored: {len(seen)}")
        else:
            print(f"Batch {total - len(batch)}-{total - 1}: "
                  f"energy [{min_e:.4f}, {max_e:.4f}] eV, "
                  f"frontier: {len(frontier)}, explored: {len(seen)}")

        # Find neighbors for all batch points
        all_new_nbs = []
        for energy, pt in batch:
            nbs = find_neighbors(pt)

            if dropped is not None:
                nbs = [nb for nb in nbs if nb not in dropped]

            for nb in nbs:
                if track_graph:
                    edge_set.add((pt, nb))
                if nb in goal_set:
                    goal_found = nb
                    break
                if dropped is not None and nb not in seen and random.random() < dropout:
                    dropped.add(nb)
                    continue
                if nb not in seen:
                    seen.add(nb)
                    all_new_nbs.append(nb)

            if goal_found is not None:
                break

        if goal_found is not None:
            goal_energy = energy_calc.calculate_energy(goal_found)
            explored_energies.append(goal_energy)
            if track_graph:
                node_energies[goal_found] = goal_energy
            break

        # Batch evaluate ALL new neighbors at once
        if all_new_nbs:
            nb_energies = energy_calc.calculate_energies_batch(all_new_nbs)
            for nb, nb_energy in zip(all_new_nbs, nb_energies):
                heapq.heappush(frontier, (nb_energy, counter, nb))
                counter += 1
                if track_graph:
                    node_energies[nb] = nb_energy

        # Checkpoint periodically (background fork to avoid blocking)
        if checkpoint_path and total // checkpoint_interval > last_checkpoint // checkpoint_interval:
            checkpoint_pid = _reap_checkpoint(checkpoint_pid)
            if checkpoint_pid is not None:
                print(f"  Skipping checkpoint at {total} (previous still writing)")
            else:
                last_checkpoint = total
                state = WaterfillState(
                    frontier=frontier,
                    seen=seen,
                    explored_energies=explored_energies,
                    counter=counter,
                    goal_found=goal_found,
                    iteration=total,
                    dropped=dropped,
                    node_energies=node_energies,
                    edge_set=edge_set,
                )
                checkpoint_pid = _checkpoint_bg(state, checkpoint_path)
                print(f"  Checkpoint started at point {total} -> {checkpoint_path}")

    # Wait for any pending checkpoint to finish before returning
    if checkpoint_pid is not None:
        print("Waiting for final checkpoint to finish...")
        os.waitpid(checkpoint_pid, 0)
        print("Final checkpoint complete.")

    if goal_found is None:
        print(f"Goal not found after {points_explored} points "
              f"(total: {start_iteration + points_explored})")

    barrier_height = max(explored_energies)

    if track_graph:
        return barrier_height, explored_energies, goal_found, node_energies, edge_set
    return barrier_height, explored_energies, goal_found
