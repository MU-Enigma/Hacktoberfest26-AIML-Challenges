from pathlib import Path

import pytest

from lunchlab.cli import main

ROOT = Path(__file__).resolve().parents[1]
DATA = str(ROOT / "data" / "lunches.csv")
SAMPLE = str(ROOT / "samples" / "new_lunches.csv")


def read_metric(output, name):
    for line in output.splitlines():
        if line.startswith(name):
            return float(line.split(":")[1])
    raise AssertionError(f"{name} not found in output:\n{output}")


def test_missing_data_file_returns_error(tmp_path, capsys):
    assert main(["train", "--data", str(tmp_path / "nope.csv"), "--out", str(tmp_path / "m.json")]) == 1
    assert "error" in capsys.readouterr().err


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["dance"])
