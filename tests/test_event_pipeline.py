import numpy as np

from deltapd.campaign.event_pipeline import extract_event_series


def test_extract_event_series_aligns_events_with_delta_t():
    signal = np.array([0.0, 0.05, 3.0, 0.1, 0.0, 4.0, 0.1, 0.0, 5.0, 0.05], dtype=np.float64)
    absolute_times_s = np.arange(len(signal), dtype=np.float64) * 1e-7

    extracted = extract_event_series(
        signal,
        fs_hz=1.0e7,
        absolute_times_s=absolute_times_s,
        threshold_sigma=0.5,
        min_separation_s=2e-7,
        detection_method="threshold",
        wavelet_denoise=False,
    )

    np.testing.assert_array_equal(extracted.pulse_indices, np.array([2, 5, 8]))
    np.testing.assert_allclose(extracted.pulse_toa_s, np.array([2e-7, 5e-7, 8e-7]))
    np.testing.assert_allclose(extracted.event_toa_s, np.array([5e-7, 8e-7]))
    np.testing.assert_allclose(extracted.event_delta_t_s, np.array([3e-7, 3e-7]))
    np.testing.assert_allclose(extracted.event_peaks_v, np.array([4.0, 5.0]))
