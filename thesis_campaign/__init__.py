"""Backward-compatible namespace for legacy imports."""

from deltapd.campaign import run_material_state, run_thesis_campaign

__all__ = ["run_thesis_campaign", "run_material_state"]
