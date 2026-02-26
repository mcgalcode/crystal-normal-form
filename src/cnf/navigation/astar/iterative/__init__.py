"""Iterative A* barrier search algorithms.

Four main algorithms:
- search: Parameter search to find optimal xi/delta/min_distance (Phase 1)
- sample: Path sampling to discover initial energy ceiling (Phase 2)
- sweep: Parallel ceiling sweep with multi-resolution refinement (Phase 3)
- ratchet: Serial barrier refinement with ratcheting ceiling (Phase 4)
"""

from .search import search
from .sample import sample
from .sweep import sweep
from .ratchet import ratchet

__all__ = ['search', 'sample', 'sweep', 'ratchet']
