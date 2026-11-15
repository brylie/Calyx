//! tests/adversarial_common_oracle_challenge.rs
//!
//! Empirical Adversarial Verification Harness for Calyx DSP Common Test Utilities
//! and Analytical Reference Oracles (`tests/common/mod.rs`).

#[path = "common/mod.rs"]
mod common;

use calyx_dsp::math::Complex;
use std::f64::consts::PI;

// ============================================================================
// 1. Sine Generator Analytical Reference Stress Tests
// ============================================================================

#[test]
fn challenge_analytical_sine_zero_samples_and_bounds() {
    let empty = common::generate_analytical_sine(440.0, 1.0, 0.0, 44100.0, 0);
    assert!(empty.is_empty(), "Zero samples must return empty vector");

    let dc = common::generate_analytical_sine(0.0, 1.0, PI / 2.0, 44100.0, 100);
    assert_eq!(dc.len(), 100);
    for &s in &dc {
        assert!((s - 1.0).abs() < 1e-12, "0 Hz sine with PI/2 phase must be 1.0");
    }
}

#[test]
fn challenge_analytical_sine_zero_sample_rate_produces_nan() {
    let bad = common::generate_analytical_sine(440.0, 1.0, 0.0, 0.0, 10);
    // When sample_rate = 0.0, t = n / 0.0 = NaN (for n=0) or Inf (for n>0).
    // sin(NaN) is NaN; sin(Inf) is NaN.
    for &s in &bad {
        assert!(s.is_nan(), "Zero sample rate must yield NaN in sine evaluation");
    }
}

// ============================================================================
// 2. Harmonic Series Analytical Reference Stress Tests
// ============================================================================

#[test]
fn challenge_harmonic_series_zero_partials_or_samples() {
    let empty_samples = common::generate_analytical_harmonic_series(440.0, 1.0, 5, 0.0, None, 44100.0, 0);
    assert!(empty_samples.is_empty());

    let zero_partials = common::generate_analytical_harmonic_series(440.0, 1.0, 0, 0.0, None, 44100.0, 100);
    assert_eq!(zero_partials.len(), 100);
    assert!(zero_partials.iter().all(|&s| s == 0.0), "0 partials must be pure silence");
}

#[test]
fn challenge_harmonic_series_culls_super_nyquist() {
    let fs = 44100.0;
    // When normalized so total sum of amplitudes <= 1.0 (amplitude = 0.5):
    let sig = common::generate_analytical_harmonic_series(10000.0, 0.5, 5, 0.0, None, fs, 512);
    assert_eq!(sig.len(), 512);
    common::assert_clean_audio(&sig);
}

#[test]
#[should_panic]
fn challenge_harmonic_series_unnormalized_amplitude_clips() {
    let fs = 44100.0;
    // With amplitude = 1.0, 1.0 + 0.5 = 1.5 > 1.0001 clipping headroom!
    let sig = common::generate_analytical_harmonic_series(10000.0, 1.0, 5, 0.0, None, fs, 512);
    common::assert_clean_audio(&sig);
}

#[test]
#[should_panic]
fn challenge_band_power_empty_slice_panics() {
    common::compute_band_power(&[], 100.0, 1000.0, 44100.0);
}

#[test]
#[should_panic]
fn challenge_spectral_centroid_empty_slice_panics() {
    common::compute_spectral_centroid(&[], 44100.0);
}

#[test]
fn challenge_harmonic_series_zero_decay_tau_singularity() {
    // If decay_tau is Some(0.0), at sample 0 (t = 0.0):
    // -t / tau = -0.0 / 0.0 = NaN -> exp(NaN) = NaN!
    let sig = common::generate_analytical_harmonic_series(440.0, 1.0, 3, 0.0, Some(0.0), 44100.0, 10);
    let has_nan = sig.iter().any(|s| s.is_nan());
    assert!(
        has_nan,
        "decay_tau = Some(0.0) causes 0.0 / 0.0 = NaN at sample index 0; this documents an oracle edge condition"
    );
}

// ============================================================================
// 3. Chirp Analytical Reference Stress Tests
// ============================================================================

#[test]
fn challenge_chirp_linear_zero_duration() {
    let empty = common::generate_analytical_chirp(100.0, 1000.0, 1.0, 0.0, 44100.0, false);
    assert!(empty.is_empty(), "Zero duration chirp must yield empty buffer");
}

#[test]
fn challenge_chirp_log_identical_start_end_frequency_singularity() {
    // When f_start == f_end in log chirp, ln(f_end / f_start) == ln(1.0) == 0.0.
    // factor = 2.0 * PI * f_start * duration / 0.0 = Inf.
    // norm_t = 0 -> (1.0)^0 - 1.0 = 0.0.
    // phase = Inf * 0.0 = NaN.
    // Thus, log chirp with f_start == f_end produces NaNs!
    let chirp = common::generate_analytical_chirp(440.0, 440.0, 1.0, 0.1, 44100.0, true);
    let has_nan = chirp.iter().any(|s| s.is_nan());
    assert!(
        has_nan,
        "Log chirp with f_start == f_end divides by ln(1)=0 yielding NaN; this documents an oracle edge condition"
    );
}

// ============================================================================
// 4. Exact DFT & Parseval Energy Stress Tests
// ============================================================================

#[test]
fn challenge_exact_dft_empty_and_single_sample() {
    let empty = common::compute_exact_dft(&[]);
    assert!(empty.is_empty(), "DFT of empty slice must be empty");

    let single = common::compute_exact_dft(&[4.5]);
    assert_eq!(single.len(), 1);
    assert!((single[0].re - 4.5).abs() < 1e-14);
    assert!(single[0].im.abs() < 1e-14);
}

#[test]
fn challenge_exact_dft_parseval_conservation_on_various_signals() {
    let sizes = &[2, 4, 7, 16, 33, 64];
    for &n in sizes {
        // Delta impulse
        let mut delta = vec![0.0; n];
        delta[0] = 1.0;
        let dft_delta = common::compute_exact_dft(&delta);
        common::assert_parseval_energy(&delta, &dft_delta);

        // Sinusoid
        let sine: Vec<f64> = (0..n).map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin()).collect();
        let dft_sine = common::compute_exact_dft(&sine);
        common::assert_parseval_energy(&sine, &dft_sine);

        // Pure silence
        let silence = vec![0.0; n];
        let dft_silence = common::compute_exact_dft(&silence);
        common::assert_parseval_energy(&silence, &dft_silence);
    }
}

#[test]
#[should_panic(expected = "Time and frequency lengths must match")]
fn challenge_parseval_length_mismatch_panics() {
    common::assert_parseval_energy(&[1.0, 0.0], &[Complex::new(1.0, 0.0)]);
}

// ============================================================================
// 5. Clean Audio Assertion Edge Cases
// ============================================================================

#[test]
#[should_panic(expected = "Audio signal must not be empty")]
fn challenge_assert_clean_audio_empty_panics() {
    common::assert_clean_audio(&[]);
}

#[test]
#[should_panic(expected = "clipped beyond headroom")]
fn challenge_assert_clean_audio_clipping_panics() {
    common::assert_clean_audio(&[0.1, 1.05, -0.2]);
}

#[test]
#[should_panic(expected = "is NaN")]
fn challenge_assert_clean_audio_nan_panics() {
    common::assert_clean_audio(&[0.1, f64::NAN, -0.2]);
}

#[test]
#[should_panic(expected = "is Infinite")]
fn challenge_assert_clean_audio_inf_panics() {
    common::assert_clean_audio(&[0.1, f64::INFINITY, -0.2]);
}

// ============================================================================
// 6. Band Power & Spectral Centroid Oracles
// ============================================================================

#[test]
fn challenge_band_power_and_spectral_centroid_accuracy() {
    let fs = 48000.0;
    let n = 256;
    // Pure sine tone at 3000 Hz
    let sine: Vec<f64> = (0..n).map(|i| (2.0 * PI * 3000.0 * i as f64 / fs).sin()).collect();

    // Centroid of 3000 Hz sine should be close to 3000 Hz
    let centroid = common::compute_spectral_centroid(&sine, fs);
    assert!((centroid - 3000.0).abs() < 200.0, "Centroid expected ~3000 Hz, got {}", centroid);

    // Power around 3000 Hz ([2000, 4000]) should capture most of the energy
    let in_band_power = common::compute_band_power(&sine, 2000.0, 4000.0, fs);
    let out_of_band_power = common::compute_band_power(&sine, 10000.0, 15000.0, fs);
    assert!(in_band_power > 0.4, "Expected high power in 3 kHz band, got {}", in_band_power);
    assert!(out_of_band_power < 1e-4, "Expected negligible power out of band, got {}", out_of_band_power);

    // Silence centroid should return 0.0
    let silence = vec![0.0; 128];
    assert_eq!(common::compute_spectral_centroid(&silence, fs), 0.0);
}
