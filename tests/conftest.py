import pytest

def pytest_addoption(parser):
    """Adds the --run-debug command-line option to pytest."""
    parser.addoption(
        "--run-debug", action="store_true", default=False, help="run debug tests"
    )

def pytest_configure(config):
    """Registers the 'debug' marker to avoid warnings."""
    config.addinivalue_line("markers", "debug: mark test to run only with --run-debug")

def pytest_runtest_setup(item):
    """
    Hook to skip tests marked with 'debug' unless --run-debug is given.
    This runs before the test setup.
    """
    if "debug" in item.keywords and not item.config.getoption("--run-debug"):
        pytest.skip("skipping debug test; enable with --run-debug")

def pytest_report_teststatus(report):
    """
    Hook to change the reporting output for skipped debug tests.
    It changes the character from 's' (skip) to 'd' (debug).
    """
    # Check if the test was skipped during the setup phase
    if report.when == "setup" and report.skipped:
        # Check if the 'debug' marker was on the test
        if "debug" in report.keywords:
            # Return a tuple: (category, short_letter, verbose_word)
            return "skipped", "d", "DEBUG"