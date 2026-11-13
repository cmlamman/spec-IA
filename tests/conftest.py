import pytest


def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', default=False,
                     help='also run tests marked slow (converged k_z-spline sampling; minutes)')


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: converged-sampling tests that take minutes; run with --runslow')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--runslow'):
        return
    skip_slow = pytest.mark.skip(reason='slow: run with --runslow')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)
