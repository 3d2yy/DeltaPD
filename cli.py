"""Compatibility wrapper: use `python -m deltapd` or `deltapd` script."""

from deltapd.__main__ import cli


if __name__ == "__main__":
    raise SystemExit(cli())
