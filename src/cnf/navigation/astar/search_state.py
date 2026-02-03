"""A* pathfinding for CNF navigation with customizable heuristics and filters"""

import heapq
import json

from typing import List, Optional, Set
from dataclasses import dataclass, field
from .node import AStarNode

from cnf import CrystalNormalForm

@dataclass
class AStarSearchState:
    """Complete state of an A* search, for saving/resuming or extracting frontier points"""
    path: Optional[List[tuple]] = field(default=None) # The path if found, None otherwise
    open_set: list[AStarNode] = field(default_factory=list)  # List of AStarNode
    closed_set: Set[tuple] = field(default_factory=set)
    came_from: dict = field(default_factory=dict) # point -> parent point
    g_score: dict  = field(default_factory=dict) # point -> cost from start
    iterations: int = field(default=0)
    found_goal: bool = field(default=False)
    max_iterations_reached: bool = field(default=False)

    # Store CNF metadata for reconstructing CNF objects
    xi: float = field(default=0.2)
    delta: int = field(default=30)
    elements: tuple = field(default_factory=tuple)

    def get_top_frontier_points(self, n: int = 10, by: str = 'h_score') -> List[tuple]:
        """
        Get the n best points from the open set.

        Args:
            n: Number of points to return
            by: Sort criterion - 'h_score' (closest to goal), 'f_score' (best overall),
                or 'g_score' (most explored)

        Returns:
            List of point tuples (vonorms + coords concatenated)
        """
        if not self.open_set:
            return []

        if by == 'h_score':
            sorted_nodes = sorted(self.open_set, key=lambda x: x.h_score)
        elif by == 'f_score':
            sorted_nodes = sorted(self.open_set, key=lambda x: x.f_score)
        elif by == 'g_score':
            sorted_nodes = sorted(self.open_set, key=lambda x: -x.g_score)  # Higher g = more explored
        else:
            raise ValueError(f"Unknown sort criterion: {by}")

        return [node.point for node in sorted_nodes[:n]]

    def get_top_frontier_cnfs(self, n: int = 10, by: str = 'h_score') -> List['CrystalNormalForm']:
        """
        Get the n best points from the open set as CNF objects.

        Args:
            n: Number of CNFs to return
            by: Sort criterion - 'h_score', 'f_score', or 'g_score'

        Returns:
            List of CrystalNormalForm objects
        """
        points = self.get_top_frontier_points(n, by)
        return [
            CrystalNormalForm.from_tuple(pt, self.elements, self.xi, self.delta)
            for pt in points
        ]
    
    def get_cnfs_on_path(self):
        if self.path is None:
            return None
        else:
            return [
                CrystalNormalForm.from_tuple(pt, self.elements, self.xi, self.delta)
                for pt in self.path
            ]   

    def frontier_stats(self) -> dict:
        """Get statistics about the current frontier (open set)"""
        if not self.open_set:
            return {'size': 0}

        h_scores = [n.h_score for n in self.open_set]
        g_scores = [n.g_score for n in self.open_set]
        f_scores = [n.f_score for n in self.open_set]

        return {
            'size': len(self.open_set),
            'closed_size': len(self.closed_set),
            'h_min': min(h_scores),
            'h_max': max(h_scores),
            'h_mean': sum(h_scores) / len(h_scores),
            'g_min': min(g_scores),
            'g_max': max(g_scores),
            'g_mean': sum(g_scores) / len(g_scores),
            'f_min': min(f_scores),
            'f_max': max(f_scores),
        }

    def to_dict(self) -> dict:
        """Serialize the search state to a dictionary for JSON saving"""
        return {
            'open_set': [
                {
                    'f_score': node.f_score,
                    'g_score': node.g_score,
                    'h_score': node.h_score,
                    'point': list(node.point),
                }
                for node in self.open_set
            ],
            'closed_set': [list(pt) for pt in self.closed_set],
            'came_from': {str(list(k)): list(v) for k, v in self.came_from.items()},
            'g_score': {str(list(k)): v for k, v in self.g_score.items()},
            'iterations': self.iterations,
            'path': [list(pt) for pt in self.path] if self.path else None,
            'found_goal': self.found_goal,
            'max_iterations_reached': self.max_iterations_reached,
            'xi': self.xi,
            'delta': self.delta,
            'elements': list(self.elements)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AStarSearchState':
        """Deserialize a search state from a dictionary"""
        open_set = [
            AStarNode(
                f_score=node['f_score'],
                g_score=node['g_score'],
                h_score=node['h_score'],
                point=tuple(node['point']),
            )
            for node in data['open_set']
        ]
        heapq.heapify(open_set)

        closed_set = {tuple(pt) for pt in data['closed_set']}
        came_from = {tuple(json.loads(k)): tuple(v) for k, v in data['came_from'].items()}
        g_score = {tuple(json.loads(k)): v for k, v in data['g_score'].items()}
        path = [tuple(pt) for pt in data['path']] if data['path'] else None

        return cls(
            open_set=open_set,
            closed_set=closed_set,
            came_from=came_from,
            g_score=g_score,
            iterations=data['iterations'],
            path=path,
            found_goal=data['found_goal'],
            max_iterations_reached=data['max_iterations_reached'],
            xi=data['xi'],
            delta=data['delta'],
            elements=tuple(data['elements'])
        )