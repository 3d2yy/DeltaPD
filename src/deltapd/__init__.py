"""DeltaPD — UHF partial-discharge Δt tracking framework."""

__version__ = "0.3.0"

from deltapd.decision_layer import campaign_summary, evaluate_campaign, evaluate_window
from deltapd.descriptors import (
    compute_delta_t,
    detect_pulses,
    detect_pulses_cfar,
    extract_delta_t_vector,
)
from deltapd.pipeline import main as run_legacy_pipeline
from deltapd.pipeline import run_empirical_pipeline
from deltapd.signal_model import (
    generate_uhf_pd_signal_physical,
    monte_carlo_wavelet_optimization,
    wavelet_denoise_parametric,
)
from deltapd.trackers import (
    AdaptiveEWMATracker,
    CUSUMDetector,
    KalmanDeltaTTracker,
    apply_delta_t_tracking,
)
from deltapd.validation import (
    generate_convergence_confusion_matrix,
    generate_phase4_report,
    measure_all_tracking_complexities,
)

__all__ = [
    "__version__",
    "generate_uhf_pd_signal_physical",
    "wavelet_denoise_parametric",
    "monte_carlo_wavelet_optimization",
    "detect_pulses",
    "detect_pulses_cfar",
    "compute_delta_t",
    "extract_delta_t_vector",
    "KalmanDeltaTTracker",
    "AdaptiveEWMATracker",
    "CUSUMDetector",
    "apply_delta_t_tracking",
    "measure_all_tracking_complexities",
    "generate_convergence_confusion_matrix",
    "generate_phase4_report",
    "evaluate_window",
    "evaluate_campaign",
    "campaign_summary",
    "run_empirical_pipeline",
    "run_legacy_pipeline",
]
