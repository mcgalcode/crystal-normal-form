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
    
    def get_cnfs_on_path(self):
        if self.path is None:
            return None
        else:
            return [
                CrystalNormalForm.from_tuple(pt, self.elements, self.xi, self.delta)
                for pt in self.path
            ]