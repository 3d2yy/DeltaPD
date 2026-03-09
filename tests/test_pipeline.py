"""End-to-end smoke test for the full pipeline."""

from deltapd.pipeline import run_phase1, run_phase2, run_phase3


def test_pipeline_end_to_end():
    """Full pipeline should run without errors and produce non-empty results."""
    mc_result, clean, noisy = run_phase1(
        n_samples=2048, fs=1e9, n_iterations=10, seed=42, verbose=False
    )
    assert mc_result.best_wavelet is not None
    assert mc_result.best_rmse_mean > 0

    delta_t, denoised = run_phase2(noisy, fs=1e9, mc_result=mc_result, verbose=False)
    assert len(delta_t) >= 1

    tracking = run_phase3(delta_t, verbose=False)
    assert tracking.kalman.filtered.shape == delta_t.shape
