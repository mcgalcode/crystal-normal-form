
import numpy as np

from typing import List

from cnf import CrystalNormalForm
from ...utils.pdd import pdd_for_cnfs, pdd_amd_for_cnfs

def pdd_heuristic(cnf: tuple, goals: list[CrystalNormalForm]) -> float:
    xi = goals[0].xi
    delta = goals[0].delta
    els = goals[0].elements

    pt = CrystalNormalForm.from_tuple(cnf, els, xi, delta)
    dists = [pdd_for_cnfs(pt, g, k=20) for g in goals]
    return (min(dists) * 100) ** 2

def pdd_and_manhattan(cnf: tuple, goals: list[CrystalNormalForm]) -> float:
    xi = goals[0].xi
    delta = goals[0].delta
    els = goals[0].elements

    pt = CrystalNormalForm.from_tuple(cnf, els, xi, delta)
    dists = [pdd_for_cnfs(pt, g, k=20) for g in goals]
    return 10000 * min(dists) + manhattan_distance(cnf, goals)

def pdd_amd_heuristic(cnf: tuple, goals: list[CrystalNormalForm]) -> float:
    xi = goals[0].xi
    delta = goals[0].delta
    els = goals[0].elements

    pt = CrystalNormalForm.from_tuple(cnf, els, xi, delta)
    dists = [pdd_amd_for_cnfs(pt, g, k=20) for g in goals]
    return (min(dists) * 100) ** 3    


def manhattan_distance(cnf: tuple, goals: list[CrystalNormalForm]) -> float:
    manhattan_dist = float('inf')
    current_coords = np.array(cnf)

    for goal in goals:
        goal_coords = np.array(goal.coords)
        curr_dist = np.sum(np.abs(current_coords - goal_coords))
        manhattan_dist = min(manhattan_dist, curr_dist)

    return manhattan_dist * 2

def squared_euclidean_heuristic(cnf: tuple, goals: List[CrystalNormalForm]) -> float:

    min_dist_sq = float('inf')

    for goal in goals:
        goal_coords = np.array(goal.coords)
        dist_sq = np.sum((np.array(cnf) - goal_coords) ** 2)
        min_dist_sq = min(min_dist_sq, dist_sq)

    return min_dist_sq