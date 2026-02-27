"""Tests for worker pool initialization and calculator propagation."""

import pytest
from unittest.mock import MagicMock


class TestWorkerCalculatorPropagation:
    """Tests that workers use the correct calculator, not a default."""

    def test_init_search_worker_uses_provided_calc_provider(self):
        """Worker init should use the provided calc_provider."""
        from cnf.navigation.astar.iterative.sweep import workers
        from cnf.navigation.astar.iterative.sweep.workers import init_search_worker

        # Create a custom calculator that we want workers to use
        custom_calc = MagicMock()
        custom_calc.identifier.return_value = "CustomModel(/path/to/finetuned)"
        calc_provider = MagicMock(return_value=custom_calc)

        # Initialize the worker with our custom provider
        init_search_worker(calc_provider=calc_provider)

        # Verify our provider was called
        calc_provider.assert_called_once()

        # Verify the worker's calculator is our custom one
        assert workers._worker_calc is custom_calc
