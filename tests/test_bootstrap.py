"""Smoke tests for initial project bootstrap behavior."""

from main import main


def test_main_bootstrap_message(capsys) -> None:
    """main() should print the bootstrap message."""
    main()
    captured = capsys.readouterr()
    assert "project bootstrapped" in captured.out
