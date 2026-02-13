import dataclasses

@dataclasses.dataclass
class PathFindingParameters():

    xi: float
    delta: int
    min_distance: float
    max_iterations: int
    heuristic_mode: str
    heuristic_weight: float

    beam_width: int = 1000
    greedy: bool = False
    min_atoms: int = None
    dropout: float = None