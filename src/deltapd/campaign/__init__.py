"""Campaign mode for thesis-aligned processing."""

from .comparative_thesis_study import run_comparative_thesis_study
from .descriptor_study import run_descriptor_study
from .mat_series_study import run_mat_series_study
from .material_state import run_material_state
from .state_alarm_batch import run_state_alarm_batch
from .thesis_campaign import run_thesis_campaign

__all__ = [
    "run_comparative_thesis_study",
    "run_thesis_campaign",
    "run_material_state",
    "run_descriptor_study",
    "run_mat_series_study",
    "run_state_alarm_batch",
]
