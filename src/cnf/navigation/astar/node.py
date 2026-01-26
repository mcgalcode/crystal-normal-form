from dataclasses import dataclass, field

@dataclass(order=True)
class AStarNode:
    """Node in the A* search priority queue"""
    f_score: float
    g_score: float = field(compare=False)
    h_score: float = field(compare=True)
    point: tuple = field(compare=False)