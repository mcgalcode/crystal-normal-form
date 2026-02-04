import json

from typing import Optional, Set
from dataclasses import dataclass, field

from cnf import CrystalNormalForm

from dataclasses import asdict

@dataclass
class PathSearchResult:
    """Complete state of an A* search, for saving/resuming or extracting frontier points"""

    # Store CNF metadata for reconstructing CNF objects
    xi: float = field()
    delta: int = field()
    elements: tuple = field()
    n_atoms: int
    num_iterations: int

    greedy: bool
    beam_width: Optional[int]
    dropout: float
    max_iterations: int

    metadata: dict

    path: Optional[list[list[int]]] = field(default=None) # The path if found, None otherwise
    max_iterations_reached: bool = field(default=False)

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_json_file(cls, fpath: str):
        with open(fpath, 'r+') as f:
            kwargs = json.load(f)
            return cls(**kwargs)

    def get_cnfs_on_path(self):
        if self.path is None:
            return None
        else:
            return [
                CrystalNormalForm.from_tuple(pt, self.elements, self.xi, self.delta)
                for pt in self.path
            ]