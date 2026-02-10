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

    _loaded_path = None

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_json_file(cls, fpath: str):
        with open(fpath, 'r+') as f:
            kwargs = json.load(f)
            return cls(**kwargs)

    def get_step_types(self):
        """Label each step on the path as 'lattice' or 'motif'.

        A 'lattice' step is one where the vonorms (first 7 coords) changed.
        A 'motif' step is one where only the motif coords (index 7+) changed.
        Each neighbor-finding step changes exactly one of these.

        Returns:
            List of strings, length len(path) - 1. Each entry is
            'lattice' or 'motif'.
        """
        if self.path is None:
            return None
        labels = []
        for i in range(len(self.path) - 1):
            vonorms_changed = tuple(self.path[i][:7]) != tuple(self.path[i + 1][:7])
            labels.append('lattice' if vonorms_changed else 'motif')
        return labels

    def get_cnfs_on_path(self):
        if self.path is None:
            return None
        else:
            if self._loaded_path is not None:
                return self._loaded_path
            else:
                self._loaded_path = [
                    CrystalNormalForm.from_tuple(pt, self.elements, self.xi, self.delta)
                    for pt in self.path
                ]
                return self._loaded_path