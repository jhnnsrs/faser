import pytest
import json
from unittest.mock import patch, mock_open, MagicMock
from click.testing import CliRunner
from faser.cli.main import main


@patch("tifffile.imsave")  # Mocking tifffile to prevent file writing during tests
def test_main(mock_tifffile):
    # Setup a runner for CLI testing
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["--a1", "0.5", "--a2", "0.5"])

        assert result.exit_code == 0
