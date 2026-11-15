//! Common test utilities, analytical ground truth generators, assertion macros,
//! and tolerance constants for the Calyx DSP E2E Test Suite.
//!
//! Complies with Demo Integrity Mode: 100% pure standard library, safe Rust.

#![allow(dead_code)]

use calyx_dsp::math::Complex;
use std::f64::consts::PI;

// ============================================================================
// Standard Audio Sampling Rates & Physical Constants
// ============================================================================

/// Standard Audio CD sampling rate (44,100 Hz).
pub const DEFAULT_SR_441: f64 = 44100.0;

/// Standard Studio / Broadcast sampling rate (48,000 Hz).
pub const DEFAULT_SR_48: f64 = 48000.0;

/// High-resolution audio sampling rate (96,000 Hz).
pub const DEFAULT_SR_96: f64 = 96000.0;

/// Division-by-zero protection floor epsilon.
pub const EPSILON: f64 = 1e-12;

/// Threshold floor delta for Log-Spectral Distance (-120 dBFS).
pub const DELTA_FLOOR: f64 = 1e-6;

// ============================================================================
// Quality Tier Threshold Constants
// ============================================================================

/// Reference Tier minimum SNR threshold (dB).
pub const TIER_REF_MIN_SNR: f64 = 35.0;

/// Production Tier minimum SNR threshold (dB).
pub const TIER_PROD_MIN_SNR: f64 = 28.0;

/// Preview Tier minimum SNR threshold (dB).
pub const TIER_PREV_MIN_SNR: f64 = 18.0;

/// Reference Tier maximum Log-Spectral Distance threshold (dB).
pub const TIER_REF_MAX_LSD: f64 = 0.8;

/// Production Tier maximum Log-Spectral Distance threshold (dB).
pub const TIER_PROD_MAX_LSD: f64 = 1.5;

/// Preview Tier maximum Log-Spectral Distance threshold (dB).
pub const TIER_PREV_MAX_LSD: f64 = 2.8;

// ============================================================================
// Assertion Macros
// ============================================================================

/// Asserts that a result is an `Err` matching the specified `DspError` pattern.
#[macro_export]
macro_rules! assert_dsp_err {
    ($expr:expr, $expected_pat:pat) => {
        match $expr {
            Err($expected_pat) => {} // Passed
            Err(ref other) => panic!(
                "Expected error pattern {}, got {:?}",
                stringify!($expected_pat),
                other
            ),
            Ok(ref val) => panic!(
                "Expected error pattern {}, but operation succeeded with {:?}",
                stringify!($expected_pat),
                val
            ),
        }
    };
}

// ============================================================================
// Analytical Ground Truth Generators (Mathematical Reference Oracles)
// ============================================================================

/// Analytical pure sine wave generator computing $s[n] = A \sin(2\pi f_0 n / f_s + \phi)$.
pub fn generate_analytical_sine(
    frequency: f64,
    amplitude: f64,
    phase: f64,
    sample_rate: f64,
    num_samples: usize,
) -> Vec<f64> {
    (0..num_samples)
        .map(|n| {
            let t = n as f64 / sample_rate;
            amplitude * (2.0 * PI * frequency * t + phase).sin()
        })
        .collect()
}

/// Analytical multi-tone harmonic series with inharmonicity $f_k = k f_0 \sqrt{1 + B k^2}$
/// and optional frequency-dependent exponential decay $\exp(-t / \tau_k)$ where $\tau_k = \tau_0 / \sqrt{k}$.
pub fn generate_analytical_harmonic_series(
    f0: f64,
    amplitude: f64,
    num_partials: usize,
    inharmonicity_b: f64,
    decay_tau: Option<f64>,
    sample_rate: f64,
    num_samples: usize,
) -> Vec<f64> {
    let mut buffer = vec![0.0; num_samples];
    let nyquist = sample_rate / 2.0;

    for k in 1..=num_partials {
        let fk = (k as f64) * f0 * (1.0 + inharmonicity_b * (k as f64).powi(2)).sqrt();
        if fk >= nyquist {
            continue;
        }

        let ak = amplitude / (k as f64);
        let tau_k = decay_tau.map(|tau0| tau0 / (k as f64).sqrt());

        for (n, sample) in buffer.iter_mut().enumerate() {
            let t = n as f64 / sample_rate;
            let env = match tau_k {
                Some(tau) => (-t / tau).exp(),
                None => 1.0,
            };
            *sample += ak * env * (2.0 * PI * fk * t).sin();
        }
    }

    buffer
}

/// Analytical continuous frequency sweep (linear or logarithmic).
pub fn generate_analytical_chirp(
    f_start: f64,
    f_end: f64,
    amplitude: f64,
    duration: f64,
    sample_rate: f64,
    is_logarithmic: bool,
) -> Vec<f64> {
    let num_samples = (duration * sample_rate).round() as usize;
    let mut buffer = Vec::with_capacity(num_samples);

    if is_logarithmic {
        assert!(f_start > 0.0 && f_end > 0.0, "Log chirp requires f > 0");
        let factor = 2.0 * PI * f_start * duration / (f_end / f_start).ln();
        for n in 0..num_samples {
            let t = n as f64 / sample_rate;
            let norm_t = t / duration;
            let phase = factor * ((f_end / f_start).powf(norm_t) - 1.0);
            buffer.push(amplitude * phase.sin());
        }
    } else {
        let k = (f_end - f_start) / duration;
        for n in 0..num_samples {
            let t = n as f64 / sample_rate;
            let phase = 2.0 * PI * (f_start * t + 0.5 * k * t * t);
            buffer.push(amplitude * phase.sin());
        }
    }

    buffer
}

/// Exact $O(N^2)$ direct Discrete Fourier Transform for ground-truth verification.
pub fn compute_exact_dft(signal: &[f64]) -> Vec<Complex<f64>> {
    let n = signal.len();
    let mut spectrum = Vec::with_capacity(n);
    let two_pi_over_n = 2.0 * PI / (n as f64);

    for k in 0..n {
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        for (idx, &x) in signal.iter().enumerate() {
            let angle = two_pi_over_n * ((k * idx) as f64);
            sum_re += x * angle.cos();
            sum_im -= x * angle.sin();
        }
        spectrum.push(Complex::new(sum_re, sum_im));
    }

    spectrum
}

// ============================================================================
// Fidelity and Stability Verification Assertions
// ============================================================================

/// Asserts that Parseval's energy conservation theorem holds between time and frequency domains:
/// $\sum |x[n]|^2 == \frac{1}{N} \sum |X[k]|^2$.
pub fn assert_parseval_energy(time_samples: &[f64], freq_bins: &[Complex<f64>]) {
    let n = time_samples.len();
    assert_eq!(n, freq_bins.len(), "Time and frequency lengths must match");
    let time_energy: f64 = time_samples.iter().map(|&s| s * s).sum();
    let freq_energy: f64 = freq_bins
        .iter()
        .map(|c: &Complex<f64>| c.norm_sq())
        .sum::<f64>()
        / (n as f64);
    let diff = (time_energy - freq_energy).abs();
    let rel_diff = diff / time_energy.max(1e-12);
    assert!(
        rel_diff < 1e-10,
        "Parseval energy violation: time_energy={:.8}, freq_energy={:.8}, rel_diff={:.2e}",
        time_energy,
        freq_energy,
        rel_diff
    );
}

/// Asserts that an audio buffer contains strictly valid, finite, normalized samples
/// (zero NaNs, zero Infs, bounded within $[-1.0001, 1.0001]$).
pub fn assert_clean_audio(signal: &[f64]) {
    assert!(!signal.is_empty(), "Audio signal must not be empty");
    for (i, &s) in signal.iter().enumerate() {
        assert!(!s.is_nan(), "Sample at index {} is NaN", i);
        assert!(!s.is_infinite(), "Sample at index {} is Infinite", i);
        assert!(
            s.abs() <= 1.0001,
            "Sample at index {} clipped beyond headroom: {:.6}",
            i,
            s
        );
    }
}

/// Computes the signal power within a specific frequency band $[f_{\text{low}}, f_{\text{high}}]$.
pub fn compute_band_power(
    signal: &[f64],
    f_low: f64,
    f_high: f64,
    sample_rate: f64,
) -> f64 {
    let n = signal.len();
    let spectrum = compute_exact_dft(signal);
    let df = sample_rate / (n as f64);

    let k_start = (f_low / df).floor() as usize;
    let k_end = ((f_high / df).ceil() as usize).min(n / 2);

    let mut band_energy = 0.0;
    for k in k_start..=k_end {
        let mag_sq = spectrum[k].norm_sq();
        let weight = if k == 0 || (n % 2 == 0 && k == n / 2) {
            1.0
        } else {
            2.0
        };
        band_energy += weight * mag_sq;
    }

    band_energy / ((n * n) as f64)
}

/// Computes the spectral centroid (center of mass of the magnitude spectrum) in Hz.
pub fn compute_spectral_centroid(signal: &[f64], sample_rate: f64) -> f64 {
    let n = signal.len();
    let spectrum = compute_exact_dft(signal);
    let df = sample_rate / (n as f64);

    let mut weighted_sum = 0.0;
    let mut total_magnitude = 0.0;

    for k in 0..=(n / 2) {
        let freq = (k as f64) * df;
        let mag = spectrum[k].norm();
        weighted_sum += freq * mag;
        total_magnitude += mag;
    }

    if total_magnitude < EPSILON {
        0.0
    } else {
        weighted_sum / total_magnitude
    }
}
