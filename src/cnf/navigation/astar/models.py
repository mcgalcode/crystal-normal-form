"""Data models for A* pathfinding results and parameters."""

from dataclasses import dataclass, field, asdict
from typing import Any
import json

from cnf import CrystalNormalForm


@dataclass
class PathContext:
    """Immutable CNF context shared across paths."""

    xi: float
    delta: int
    elements: tuple[str, ...]

    # Optional endpoint structures (pymatgen Structure dicts or CIF paths)
    start_structure: dict | str | None = None
    end_structure: dict | str | None = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            'xi': self.xi,
            'delta': self.delta,
            'elements': list(self.elements),
            'start_structure': self.start_structure,
            'end_structure': self.end_structure,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'PathContext':
        """Deserialize from a dictionary."""
        return cls(
            xi=d['xi'],
            delta=d['delta'],
            elements=tuple(d['elements']),
            start_structure=d.get('start_structure'),
            end_structure=d.get('end_structure'),
        )


@dataclass
class Path:
    """A single path through CNF space - lightweight, no context embedded."""

    coords: list[tuple[int, ...]]

    # Energy data (optional, filled in after evaluation)
    energies: list[float] | None = None
    barrier: float | None = None

    # Path-specific metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.coords)

    def get_cnfs(self, context: PathContext) -> list[CrystalNormalForm]:
        """Reconstruct CNF objects using the provided context."""
        return [
            CrystalNormalForm.from_tuple(coord, context.elements, context.xi, context.delta)
            for coord in self.coords
        ]

    def get_cnf(self, index: int, context: PathContext) -> CrystalNormalForm:
        """Get a single CNF at the given path index."""
        return CrystalNormalForm.from_tuple(
            self.coords[index], context.elements, context.xi, context.delta
        )

    def get_structures(self, context: PathContext) -> list:
        """Reconstruct pymatgen Structure objects along the path."""
        return [cnf.to_structure() for cnf in self.get_cnfs(context)]

    def compute_barrier(self) -> float | None:
        """Compute and store the barrier (max energy) from energies."""
        if self.energies is None:
            return None
        self.barrier = max(self.energies)
        return self.barrier

    def get_step_types(self) -> list[str] | None:
        """Label each step as 'lattice' or 'motif'.

        A 'lattice' step is one where the vonorms (first 7 coords) changed.
        A 'motif' step is one where only the motif coords (index 7+) changed.

        Returns:
            List of 'lattice' or 'motif' strings, length len(path) - 1.
        """
        if not self.coords:
            return None
        labels = []
        for i in range(len(self.coords) - 1):
            vonorms_changed = self.coords[i][:7] != self.coords[i + 1][:7]
            labels.append('lattice' if vonorms_changed else 'motif')
        return labels

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            'coords': [[int(x) for x in c] for c in self.coords],
            'energies': self.energies,
            'barrier': float(self.barrier) if self.barrier is not None else None,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Path':
        """Deserialize from a dictionary."""
        return cls(
            coords=[tuple(c) for c in d['coords']],
            energies=d.get('energies'),
            barrier=d.get('barrier'),
            metadata=d.get('metadata', {}),
        )


@dataclass
class Attempt:
    """A single search attempt."""

    path: Path | None
    found: bool
    iterations: int
    elapsed_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            'path': self.path.to_dict() if self.path else None,
            'found': self.found,
            'iterations': self.iterations,
            'elapsed_seconds': self.elapsed_seconds,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Attempt':
        """Deserialize from a dictionary."""
        return cls(
            path=Path.from_dict(d['path']) if d.get('path') else None,
            found=d['found'],
            iterations=d['iterations'],
            elapsed_seconds=d.get('elapsed_seconds'),
            metadata=d.get('metadata', {}),
        )


@dataclass
class SearchParameters:
    """Fully specified parameters for a search."""

    max_iterations: int = 100_000
    beam_width: int = 1000
    dropout: float = 0.0
    greedy: bool = False
    heuristic: str = "manhattan"

    # Filters - list of filter definitions, e.g.:
    # [{"type": "min_distance", "value": 0.5}, {"type": "energy_ceiling", "value": -8.5}]
    filters: list[dict[str, Any]] = field(default_factory=list)

    def get_filter(self, filter_type: str) -> dict | None:
        """Get a filter by type, or None if not present."""
        for f in self.filters:
            if f.get('type') == filter_type:
                return f
        return None

    def get_filter_value(self, filter_type: str) -> Any | None:
        """Get the value of a filter by type, or None if not present."""
        f = self.get_filter(filter_type)
        return f.get('value') if f else None

    @property
    def min_distance(self) -> float | None:
        """Convenience property for min_distance filter."""
        return self.get_filter_value('min_distance')

    @property
    def energy_ceiling(self) -> float | None:
        """Convenience property for energy_ceiling filter."""
        return self.get_filter_value('energy_ceiling')

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            'max_iterations': self.max_iterations,
            'beam_width': self.beam_width,
            'dropout': self.dropout,
            'greedy': self.greedy,
            'heuristic': self.heuristic,
            'filters': self.filters,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SearchParameters':
        """Deserialize from a dictionary."""
        return cls(
            max_iterations=d.get('max_iterations', 100_000),
            beam_width=d.get('beam_width', 1000),
            dropout=d.get('dropout', 0.0),
            greedy=d.get('greedy', False),
            heuristic=d.get('heuristic', 'manhattan'),
            filters=d.get('filters', []),
        )


@dataclass
class SearchResult:
    """Result of one or more attempts with IDENTICAL parameters."""

    context: PathContext
    parameters: SearchParameters
    attempts: list[Attempt]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def paths(self) -> list[Path]:
        """Get all successful paths."""
        return [a.path for a in self.attempts if a.path is not None]

    @property
    def best_path(self) -> Path | None:
        """Get the path with lowest barrier, or None if no paths have barriers."""
        paths_with_barriers = [p for p in self.paths if p.barrier is not None]
        if not paths_with_barriers:
            # Fall back to first path if none have barriers
            return self.paths[0] if self.paths else None
        return min(paths_with_barriers, key=lambda p: p.barrier)

    @property
    def best_barrier(self) -> float | None:
        """Get the lowest barrier across all paths."""
        best = self.best_path
        return best.barrier if best else None

    @property
    def success_rate(self) -> float:
        """Fraction of attempts that found a path."""
        if not self.attempts:
            return 0.0
        return len(self.paths) / len(self.attempts)

    def get_cnfs(self, path_index: int = 0) -> list[CrystalNormalForm] | None:
        """Get CNFs for a path by index (default: first successful path)."""
        if path_index >= len(self.paths):
            return None
        return self.paths[path_index].get_cnfs(self.context)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            'context': self.context.to_dict(),
            'parameters': self.parameters.to_dict(),
            'attempts': [a.to_dict() for a in self.attempts],
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SearchResult':
        """Deserialize from a dictionary."""
        return cls(
            context=PathContext.from_dict(d['context']),
            parameters=SearchParameters.from_dict(d['parameters']),
            attempts=[Attempt.from_dict(a) for a in d['attempts']],
            metadata=d.get('metadata', {}),
        )

    def to_json(self, path: str) -> None:
        """Write to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'SearchResult':
        """Load from a JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Higher-level phase containers
# =============================================================================


@dataclass
class CeilingSweepResult:
    """Phase 3: Multiple searches at different ceiling levels."""

    results: list[SearchResult]  # one per ceiling
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_paths(self) -> list[Path]:
        """Get all successful paths across all ceiling levels."""
        paths = []
        for result in self.results:
            paths.extend(result.paths)
        return paths

    @property
    def best_path(self) -> Path | None:
        """Get the path with lowest barrier across all results."""
        paths_with_barriers = [p for p in self.all_paths if p.barrier is not None]
        if not paths_with_barriers:
            return self.all_paths[0] if self.all_paths else None
        return min(paths_with_barriers, key=lambda p: p.barrier)

    @property
    def best_barrier(self) -> float | None:
        """Get the lowest barrier across all results."""
        best = self.best_path
        return best.barrier if best else None

    @property
    def context(self) -> PathContext | None:
        """Get the shared context (from first result)."""
        return self.results[0].context if self.results else None

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            'results': [r.to_dict() for r in self.results],
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CeilingSweepResult':
        """Deserialize from a dictionary."""
        return cls(
            results=[SearchResult.from_dict(r) for r in d['results']],
            metadata=d.get('metadata', {}),
        )

    def to_json(self, path: str) -> None:
        """Write to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'CeilingSweepResult':
        """Load from a JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class ParameterSearchResult:
    """Phase 1: Parameter search to find optimal xi/delta/min_distance."""

    # List of (xi, delta, min_distance) tuples that successfully found paths
    successful_params: list[tuple[float, int, float]]

    # Results for each resolution tried (including failures)
    results: list[SearchResult]

    # The recommended parameters (finest resolution that worked)
    recommended_xi: float | None = None
    recommended_delta: int | None = None
    recommended_min_distance: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether any parameters successfully found a path."""
        return len(self.successful_params) > 0

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            'successful_params': [list(p) for p in self.successful_params],
            'results': [r.to_dict() for r in self.results],
            'recommended_xi': self.recommended_xi,
            'recommended_delta': self.recommended_delta,
            'recommended_min_distance': self.recommended_min_distance,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ParameterSearchResult':
        """Deserialize from a dictionary."""
        return cls(
            successful_params=[tuple(p) for p in d['successful_params']],
            results=[SearchResult.from_dict(r) for r in d['results']],
            recommended_xi=d.get('recommended_xi'),
            recommended_delta=d.get('recommended_delta'),
            recommended_min_distance=d.get('recommended_min_distance'),
            metadata=d.get('metadata', {}),
        )

    def to_json(self, path: str) -> None:
        """Write to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'ParameterSearchResult':
        """Load from a JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class RefinementResult:
    """Phase 4: Serial refinement with ratcheting ceiling."""

    results: list[SearchResult]  # one per round
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_paths(self) -> list[Path]:
        """Get all successful paths across all rounds."""
        paths = []
        for result in self.results:
            paths.extend(result.paths)
        return paths

    @property
    def best_path(self) -> Path | None:
        """Get the path with lowest barrier across all rounds."""
        paths_with_barriers = [p for p in self.all_paths if p.barrier is not None]
        if not paths_with_barriers:
            return self.all_paths[0] if self.all_paths else None
        return min(paths_with_barriers, key=lambda p: p.barrier)

    @property
    def best_barrier(self) -> float | None:
        """Get the lowest barrier across all rounds."""
        best = self.best_path
        return best.barrier if best else None

    @property
    def context(self) -> PathContext | None:
        """Get the shared context (from first result)."""
        return self.results[0].context if self.results else None

    @property
    def final_ceiling(self) -> float | None:
        """Get the final ceiling value (from last result's parameters)."""
        if not self.results:
            return None
        return self.results[-1].parameters.energy_ceiling

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            'results': [r.to_dict() for r in self.results],
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'RefinementResult':
        """Deserialize from a dictionary."""
        return cls(
            results=[SearchResult.from_dict(r) for r in d['results']],
            metadata=d.get('metadata', {}),
        )

    def to_json(self, path: str) -> None:
        """Write to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'RefinementResult':
        """Load from a JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
