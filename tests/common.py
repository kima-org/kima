import pytest

@pytest.fixture
def cleanup_after_running():
    from kima.pykima.cli import cli_clean
    yield
    cli_clean(check=False, output=True)