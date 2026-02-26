"""Tests for A* pathfinding data models."""

import pytest
import json
import tempfile
from pathlib import Path

from monty.json import jsanitize, MontyDecoder

from cnf.navigation.astar.models import (
    PathContext,
    Path as PathModel,
    Attempt,
    SearchParameters,
    SearchResult,
    CeilingSweepResult,
    RefinementResult,
    ParameterSearchResult,
)


class TestPathContext:
    def test_to_dict_and_from_dict(self):
        ctx = PathContext(xi=1.5, delta=10, elements=("Zr", "O"))
        d = ctx.to_dict()

        assert d["xi"] == 1.5
        assert d["delta"] == 10
        assert d["elements"] == ["Zr", "O"]

        ctx2 = PathContext.from_dict(d)
        assert ctx2.xi == ctx.xi
        assert ctx2.delta == ctx.delta
        assert ctx2.elements == ctx.elements


class TestPath:
    def test_len(self):
        path = PathModel(coords=[(1, 2), (3, 4), (5, 6)])
        assert len(path) == 3

    def test_barrier_computed_from_energies(self):
        path = PathModel(
            coords=[(1,), (2,), (3,)],
            energies=[-10.0, -8.5, -9.0],
        )
        barrier = path.compute_barrier()
        assert barrier == -8.5
        assert path.barrier == -8.5

    def test_barrier_none_without_energies(self):
        path = PathModel(coords=[(1,), (2,)])
        assert path.compute_barrier() is None

    def test_to_dict_and_from_dict(self):
        path = PathModel(
            coords=[(1, 2, 3), (4, 5, 6)],
            energies=[-10.0, -9.5],
            barrier=-9.5,
            metadata={"test": "value"},
        )
        d = path.to_dict()
        path2 = PathModel.from_dict(d)

        assert path2.coords == path.coords
        assert path2.energies == path.energies
        assert path2.barrier == path.barrier
        assert path2.metadata == path.metadata

    def test_get_step_types(self):
        # Vonorms are first 7 coords
        path = PathModel(coords=[
            (1, 2, 3, 4, 5, 6, 7, 0, 0, 0),  # lattice
            (1, 2, 3, 4, 5, 6, 7, 1, 0, 0),  # motif (only index 7 changed)
            (2, 2, 3, 4, 5, 6, 7, 1, 0, 0),  # lattice (index 0 changed)
        ])
        labels = path.get_step_types()
        assert labels == ['motif', 'lattice']


class TestAttempt:
    def test_to_dict_and_from_dict_with_path(self):
        path = PathModel(coords=[(1,), (2,)], barrier=-9.5)
        attempt = Attempt(path=path, found=True, iterations=100, elapsed_seconds=1.5)

        d = attempt.to_dict()
        attempt2 = Attempt.from_dict(d)

        assert attempt2.found == attempt.found
        assert attempt2.iterations == attempt.iterations
        assert attempt2.elapsed_seconds == attempt.elapsed_seconds
        assert attempt2.path.coords == attempt.path.coords

    def test_to_dict_and_from_dict_without_path(self):
        attempt = Attempt(path=None, found=False, iterations=500)

        d = attempt.to_dict()
        attempt2 = Attempt.from_dict(d)

        assert attempt2.found is False
        assert attempt2.path is None


class TestSearchParameters:
    def test_get_filter(self):
        params = SearchParameters(
            filters=[
                {"type": "min_distance", "value": 0.5},
                {"type": "energy_ceiling", "value": -8.5},
            ]
        )

        f = params.get_filter("min_distance")
        assert f["value"] == 0.5

        assert params.get_filter("nonexistent") is None

    def test_min_distance_property(self):
        params = SearchParameters(
            filters=[{"type": "min_distance", "value": 1.2}]
        )
        assert params.min_distance == 1.2

    def test_energy_ceiling_property(self):
        params = SearchParameters(
            filters=[{"type": "energy_ceiling", "value": -7.0}]
        )
        assert params.energy_ceiling == -7.0

    def test_to_dict_and_from_dict(self):
        params = SearchParameters(
            max_iterations=50000,
            beam_width=500,
            dropout=0.3,
            filters=[{"type": "min_distance", "value": 0.8}],
        )

        d = params.to_dict()
        params2 = SearchParameters.from_dict(d)

        assert params2.max_iterations == params.max_iterations
        assert params2.beam_width == params.beam_width
        assert params2.dropout == params.dropout
        assert params2.filters == params.filters


class TestSearchResult:
    def test_paths_property(self):
        ctx = PathContext(xi=1.5, delta=10, elements=("Zr",))
        params = SearchParameters()

        result = SearchResult(
            context=ctx,
            parameters=params,
            attempts=[
                Attempt(path=PathModel(coords=[(1,)]), found=True, iterations=10),
                Attempt(path=None, found=False, iterations=20),
                Attempt(path=PathModel(coords=[(2,)]), found=True, iterations=15),
            ],
        )

        assert len(result.paths) == 2

    def test_best_path_by_barrier(self):
        ctx = PathContext(xi=1.5, delta=10, elements=("Zr",))
        params = SearchParameters()

        result = SearchResult(
            context=ctx,
            parameters=params,
            attempts=[
                Attempt(path=PathModel(coords=[(1,)], barrier=-8.0), found=True, iterations=10),
                Attempt(path=PathModel(coords=[(2,)], barrier=-9.5), found=True, iterations=20),
            ],
        )

        best = result.best_path
        assert best.barrier == -9.5

    def test_success_rate(self):
        ctx = PathContext(xi=1.5, delta=10, elements=("Zr",))
        params = SearchParameters()

        result = SearchResult(
            context=ctx,
            parameters=params,
            attempts=[
                Attempt(path=PathModel(coords=[(1,)]), found=True, iterations=10),
                Attempt(path=None, found=False, iterations=20),
                Attempt(path=None, found=False, iterations=30),
                Attempt(path=PathModel(coords=[(2,)]), found=True, iterations=15),
            ],
        )

        assert result.success_rate == 0.5

    def test_to_json_and_from_json(self):
        ctx = PathContext(xi=1.5, delta=10, elements=("Zr",))
        params = SearchParameters(max_iterations=1000)
        result = SearchResult(
            context=ctx,
            parameters=params,
            attempts=[
                Attempt(path=PathModel(coords=[(1, 2, 3)], barrier=-9.0), found=True, iterations=50),
            ],
            metadata={"test": "value"},
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            result.to_json(path)
            result2 = SearchResult.from_json(path)

            assert result2.context.xi == result.context.xi
            assert result2.parameters.max_iterations == result.parameters.max_iterations
            assert len(result2.attempts) == 1
            assert result2.metadata == result.metadata
        finally:
            Path(path).unlink()


class TestParameterSearchResult:
    def test_success_property(self):
        result = ParameterSearchResult(
            successful_params=[(1.5, 10, 0.8)],
            results=[],
        )
        assert result.success is True

        result2 = ParameterSearchResult(
            successful_params=[],
            results=[],
        )
        assert result2.success is False

    def test_to_dict_and_from_dict(self):
        result = ParameterSearchResult(
            successful_params=[(1.5, 10, 0.8), (1.0, 15, 0.6)],
            results=[],
            recommended_xi=1.0,
            recommended_delta=15,
            recommended_min_distance=0.6,
            metadata={"test": "value"},
        )

        d = result.to_dict()
        result2 = ParameterSearchResult.from_dict(d)

        assert result2.successful_params == [(1.5, 10, 0.8), (1.0, 15, 0.6)]
        assert result2.recommended_xi == 1.0
        assert result2.recommended_delta == 15
        assert result2.recommended_min_distance == 0.6
        assert result2.metadata == result.metadata

    def test_to_json_and_from_json(self):
        result = ParameterSearchResult(
            successful_params=[(1.5, 10, 0.8)],
            results=[],
            recommended_xi=1.5,
            recommended_delta=10,
            recommended_min_distance=0.8,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            result.to_json(path)
            result2 = ParameterSearchResult.from_json(path)

            assert result2.successful_params == result.successful_params
            assert result2.recommended_xi == result.recommended_xi
        finally:
            Path(path).unlink()


class TestCeilingSweepResult:
    def test_all_paths_aggregates_across_results(self):
        ctx = PathContext(xi=1.5, delta=10, elements=("Zr",))
        params = SearchParameters()

        r1 = SearchResult(
            context=ctx, parameters=params,
            attempts=[Attempt(path=PathModel(coords=[(1,)]), found=True, iterations=10)]
        )
        r2 = SearchResult(
            context=ctx, parameters=params,
            attempts=[Attempt(path=PathModel(coords=[(2,)]), found=True, iterations=20)]
        )

        sweep = CeilingSweepResult(results=[r1, r2])
        assert len(sweep.all_paths) == 2

    def test_best_barrier(self):
        ctx = PathContext(xi=1.5, delta=10, elements=("Zr",))
        params = SearchParameters()

        r1 = SearchResult(
            context=ctx, parameters=params,
            attempts=[Attempt(path=PathModel(coords=[(1,)], barrier=-8.0), found=True, iterations=10)]
        )
        r2 = SearchResult(
            context=ctx, parameters=params,
            attempts=[Attempt(path=PathModel(coords=[(2,)], barrier=-9.5), found=True, iterations=20)]
        )

        sweep = CeilingSweepResult(results=[r1, r2])
        assert sweep.best_barrier == -9.5


class TestRefinementResult:
    def test_final_ceiling(self):
        ctx = PathContext(xi=1.5, delta=10, elements=("Zr",))

        params1 = SearchParameters(filters=[{"type": "energy_ceiling", "value": -7.0}])
        params2 = SearchParameters(filters=[{"type": "energy_ceiling", "value": -8.5}])

        r1 = SearchResult(context=ctx, parameters=params1, attempts=[])
        r2 = SearchResult(context=ctx, parameters=params2, attempts=[])

        refine = RefinementResult(results=[r1, r2])
        assert refine.final_ceiling == -8.5


class TestMSONableCompatibility:
    """Test MSONable serialization for jobflow compatibility."""

    def test_path_context_msonable(self):
        ctx = PathContext(xi=1.5, delta=10, elements=("Zr", "O"))
        d = ctx.as_dict()

        assert d["@module"] == "cnf.navigation.astar.models"
        assert d["@class"] == "PathContext"

        # Round-trip through MontyDecoder
        decoded = MontyDecoder().process_decoded(d)
        assert isinstance(decoded, PathContext)
        assert decoded.xi == ctx.xi
        assert decoded.elements == ctx.elements

    def test_path_msonable(self):
        path = PathModel(coords=[(1, 2, 3), (4, 5, 6)], energies=[-10.0, -9.5], barrier=-9.5)
        d = path.as_dict()

        assert d["@module"] == "cnf.navigation.astar.models"
        assert d["@class"] == "Path"

        decoded = MontyDecoder().process_decoded(d)
        assert isinstance(decoded, PathModel)
        assert decoded.coords == path.coords
        assert decoded.barrier == path.barrier

    def test_attempt_msonable(self):
        attempt = Attempt(
            path=PathModel(coords=[(1,)], barrier=-9.0),
            found=True,
            iterations=100,
        )
        d = attempt.as_dict()

        assert d["@module"] == "cnf.navigation.astar.models"
        assert d["@class"] == "Attempt"

        decoded = MontyDecoder().process_decoded(d)
        assert isinstance(decoded, Attempt)
        assert decoded.found is True
        assert decoded.iterations == 100

    def test_search_parameters_msonable(self):
        params = SearchParameters(
            max_iterations=50000,
            beam_width=500,
            filters=[{"type": "min_distance", "value": 0.8}],
        )
        d = params.as_dict()

        assert d["@module"] == "cnf.navigation.astar.models"
        assert d["@class"] == "SearchParameters"

        decoded = MontyDecoder().process_decoded(d)
        assert isinstance(decoded, SearchParameters)
        assert decoded.max_iterations == 50000
        assert decoded.min_distance == 0.8

    def test_search_result_msonable(self):
        ctx = PathContext(xi=1.5, delta=10, elements=("Zr",))
        params = SearchParameters()
        result = SearchResult(
            context=ctx,
            parameters=params,
            attempts=[Attempt(path=PathModel(coords=[(1,)], barrier=-9.0), found=True, iterations=50)],
        )
        d = result.as_dict()

        assert d["@module"] == "cnf.navigation.astar.models"
        assert d["@class"] == "SearchResult"

        decoded = MontyDecoder().process_decoded(d)
        assert isinstance(decoded, SearchResult)
        assert decoded.context.xi == 1.5
        assert len(decoded.attempts) == 1

    def test_parameter_search_result_msonable(self):
        result = ParameterSearchResult(
            successful_params=[(1.5, 10, 0.8)],
            results=[],
            recommended_xi=1.5,
            recommended_delta=10,
            recommended_min_distance=0.8,
        )
        d = result.as_dict()

        assert d["@module"] == "cnf.navigation.astar.models"
        assert d["@class"] == "ParameterSearchResult"

        decoded = MontyDecoder().process_decoded(d)
        assert isinstance(decoded, ParameterSearchResult)
        assert decoded.recommended_xi == 1.5
        assert decoded.success is True

    def test_ceiling_sweep_result_msonable(self):
        ctx = PathContext(xi=1.5, delta=10, elements=("Zr",))
        params = SearchParameters()
        r1 = SearchResult(context=ctx, parameters=params, attempts=[])

        sweep = CeilingSweepResult(results=[r1])
        d = sweep.as_dict()

        assert d["@module"] == "cnf.navigation.astar.models"
        assert d["@class"] == "CeilingSweepResult"

        decoded = MontyDecoder().process_decoded(d)
        assert isinstance(decoded, CeilingSweepResult)
        assert len(decoded.results) == 1

    def test_refinement_result_msonable(self):
        ctx = PathContext(xi=1.5, delta=10, elements=("Zr",))
        params = SearchParameters(filters=[{"type": "energy_ceiling", "value": -8.0}])
        r1 = SearchResult(context=ctx, parameters=params, attempts=[])

        refine = RefinementResult(results=[r1])
        d = refine.as_dict()

        assert d["@module"] == "cnf.navigation.astar.models"
        assert d["@class"] == "RefinementResult"

        decoded = MontyDecoder().process_decoded(d)
        assert isinstance(decoded, RefinementResult)
        assert decoded.final_ceiling == -8.0

    def test_jsanitize_works(self):
        """Test that jsanitize (used by jobflow) works on all models."""
        result = ParameterSearchResult(
            successful_params=[(1.5, 10, 0.8)],
            results=[],
            recommended_xi=1.5,
            recommended_delta=10,
            recommended_min_distance=0.8,
        )

        # This is what jobflow uses internally
        sanitized = jsanitize(result, strict=True, allow_bson=True)
        assert sanitized is not None
        assert sanitized["@class"] == "ParameterSearchResult"

    def test_nested_msonable_roundtrip(self):
        """Test complex nested structure round-trips correctly."""
        ctx = PathContext(xi=1.5, delta=10, elements=("Ni", "Ti"))
        params = SearchParameters(
            max_iterations=5000,
            beam_width=500,
            dropout=0.3,
            filters=[
                {"type": "min_distance", "value": 0.5},
                {"type": "energy_ceiling", "value": -8.0},
            ],
        )
        path = PathModel(
            coords=[(1, 2, 3, 4, 5, 6, 7, 0, 0, 0), (1, 2, 3, 4, 5, 6, 7, 1, 0, 0)],
            energies=[-10.0, -9.5],
            barrier=-9.5,
            metadata={"source": "test"},
        )
        attempt = Attempt(path=path, found=True, iterations=100, elapsed_seconds=1.5)
        result = SearchResult(
            context=ctx,
            parameters=params,
            attempts=[attempt],
            metadata={"ceiling": -8.0},
        )

        # Full round-trip
        d = result.as_dict()
        decoded = MontyDecoder().process_decoded(d)

        assert isinstance(decoded, SearchResult)
        assert decoded.context.xi == 1.5
        assert decoded.context.elements == ("Ni", "Ti")
        assert decoded.parameters.max_iterations == 5000
        assert decoded.parameters.min_distance == 0.5
        assert len(decoded.attempts) == 1
        assert decoded.attempts[0].path.barrier == -9.5
        assert decoded.attempts[0].path.metadata == {"source": "test"}
        assert decoded.metadata == {"ceiling": -8.0}
