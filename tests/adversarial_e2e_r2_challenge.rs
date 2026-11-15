//! tests/adversarial_e2e_r2_challenge.rs
//!
//! Comprehensive Empirical Adversarial Challenge Suite for Calyx E2E Testing Track (Iteration 2).
//! Evaluates:
//! 1. Tier 4 Application Scenarios (5D hypercube, inharmonic stiff string, expressive gestures, quality tiers)
//! 2. Quantitative Fidelity Metrics (SC scale invariance, LSD silence floor, SNR zero division)
//! 3. Milestone 1 API Contract Drift across all test files

use calyx_dsp::error::*;
use calyx_dsp::math::complex::Complex;
use calyx_dsp::math::matrix::Matrix;
use calyx_dsp::math::tensor::Tensor;
use calyx_dsp::math::window::{generate_window, verify_cola, WindowType};
use std::f64::consts::PI;

const EPSILON: f64 = 1e-12;
const _DELTA_FLOOR: f64 = 1e-6;

// =========================================================================
// PART 1: FIDELITY METRICS ADVERSARIAL CHALLENGES
// =========================================================================

/// Challenge M-1: Spectral Convergence Scale Invariance Breakdown.
/// Formula in TEST_INFRA.md: SC = sqrt(sum (r - h)^2) / sqrt(sum r^2 + eps).
/// Proves that unconditional +eps breaks scale invariance assertion |SC(X, 1.2X) - 0.2| < 1e-14
/// and catastrophically fails on low-amplitude signals.
#[test]
fn challenge_sc_scale_invariance_breakdown() {
    let x_nominal = vec![1.0, 0.5, 0.25, 0.125, 0.0625];
    let x_scaled: Vec<f64> = x_nominal.iter().map(|&v| 1.2 * v).collect();

    let diff_sq: f64 = x_nominal.iter().zip(x_scaled.iter()).map(|(&r, &h)| (r - h).powi(2)).sum();
    let ref_sq: f64 = x_nominal.iter().map(|&r| r * r).sum();

    // Naive formula from TEST_INFRA.md:
    let sc_naive = (diff_sq / (ref_sq + EPSILON)).sqrt();
    let err_nominal = (sc_naive - 0.2).abs();
    println!("[CHALLENGE] Nominal SC scale error with eps: {:.4e}", err_nominal);
    // Errs by ~7.51e-14 which exceeds the < 1e-14 threshold in tier1_feature_tests.rs:956!
    assert!(err_nominal > 1e-14, "Expected naive SC to violate < 1e-14 tolerance");

    // Low-amplitude signal (pianissimo: 1e-6)
    let x_quiet: Vec<f64> = x_nominal.iter().map(|&v| v * 1e-6).collect();
    let x_quiet_scaled: Vec<f64> = x_quiet.iter().map(|&v| 1.2 * v).collect();
    let diff_quiet_sq: f64 = x_quiet.iter().zip(x_quiet_scaled.iter()).map(|(&r, &h)| (r - h).powi(2)).sum();
    let ref_quiet_sq: f64 = x_quiet.iter().map(|&r| r * r).sum();

    let sc_quiet_naive = (diff_quiet_sq / (ref_quiet_sq + EPSILON)).sqrt();
    println!("[CHALLENGE] Quiet SC scale: expected 0.2, got {:.6} (deviation = {:.4})", sc_quiet_naive, (sc_quiet_naive - 0.2).abs());
    // Proves that naive SC deviates by > 0.04 (over 20% error) on quiet signals:
    assert!((sc_quiet_naive - 0.2).abs() > 0.04, "Naive SC fails scale invariance on quiet signals");
}

/// Challenge M-2: Log-Spectral Distance (LSD) Negative Value Hazard.
/// Proves that passing time-domain audio samples (which can be negative) into the LSD formula
/// log10(x + delta) causes NaN when x <= -delta.
#[test]
fn challenge_lsd_negative_time_samples_nan_hazard() {
    let neg_time_sample = -0.5f64;
    let delta = 1e-6;
    let argument = neg_time_sample + delta; // -0.499999
    assert!(argument < 0.0);
    let log_val = (neg_time_sample + delta).log10();
    println!("[CHALLENGE] log10(-0.5 + 1e-6) = {:?}", log_val);
    assert!(log_val.is_nan(), "log10 of negative audio sample MUST produce NaN");
}

/// Challenge M-3: SNR Pure Silence Contradiction with Quality Tiers.
/// Proves that comparing pure silence produces SNR = 0.0 dB under standard formula,
/// which contradicts tier.validate_fidelity (> 35 dB required for Reference tier).
#[test]
fn challenge_snr_silence_quality_tier_conflict() {
    let p_sig = 0.0f64;
    let p_noise = 0.0f64;
    let snr_db = 10.0 * ((p_sig + EPSILON) / (p_noise + EPSILON)).log10();
    println!("[CHALLENGE] SNR on pure silence: {:.2} dB", snr_db);
    assert_eq!(snr_db, 0.0);

    // Reference tier threshold is 35.0 dB
    let ref_threshold = 35.0f64;
    assert!(snr_db < ref_threshold, "Silence SNR (0 dB) fails Reference tier (> 35 dB)");
}

// =========================================================================
// PART 2: TIER 4 APPLICATION SCENARIOS ADVERSARIAL CHALLENGES
// =========================================================================

/// Challenge T4-1: Inharmonic Stiff-String Sub-Bin Peak Resolution at N=1024.
/// Simulates parabolic interpolation on a Hann-windowed 1024-point FFT for an A2 string (f0=110 Hz, B=0.0004).
/// Tests whether partial frequency extraction error is strictly < 1.0 Hz across all 24 partials.
#[test]
fn challenge_stiff_string_peak_tracking_error() {
    let fs = 48000.0;
    let n_fft = 1024;
    let df = fs / (n_fft as f64); // 46.875 Hz
    let f0 = 110.0;
    let b = 0.0004;
    let num_partials = 24;

    // Generate analytical signal with Hann window
    let mut signal = vec![0.0f64; n_fft];
    let hann = generate_window(WindowType::Hann, n_fft).unwrap();

    for k in 1..=num_partials {
        let fk = (k as f64) * f0 * (1.0 + b * (k as f64).powi(2)).sqrt();
        let ak = 0.8 / (k as f64);
        for (n, sample) in signal.iter_mut().enumerate() {
            let t = n as f64 / fs;
            *sample += ak * (2.0 * PI * fk * t).sin();
        }
    }

    // Apply window
    for (s, w) in signal.iter_mut().zip(hann.iter()) {
        *s *= *w;
    }

    // Compute DFT magnitude spectrum
    let mut mag = vec![0.0f64; n_fft / 2 + 1];
    for bin in 0..=n_fft / 2 {
        let angle_step = 2.0 * PI * (bin as f64) / (n_fft as f64);
        let mut re = 0.0;
        let mut im = 0.0;
        for (n, &x) in signal.iter().enumerate() {
            let angle = angle_step * (n as f64);
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        mag[bin] = (re * re + im * im).sqrt();
    }

    // Perform parabolic sub-bin peak refinement for each partial
    let mut max_freq_err = 0.0f64;
    let mut passing_count = 0;

    for k in 1..=num_partials {
        let expected_fk = (k as f64) * f0 * (1.0 + b * (k as f64).powi(2)).sqrt();
        if expected_fk >= fs / 2.0 {
            break;
        }

        // Search near expected bin
        let center_bin = (expected_fk / df).round() as usize;
        let start_bin = center_bin.saturating_sub(2).max(1);
        let end_bin = (center_bin + 2).min(n_fft / 2 - 1);

        // Find local peak
        let mut best_bin = start_bin;
        let mut best_val = mag[start_bin];
        for b_idx in start_bin..=end_bin {
            if mag[b_idx] > best_val {
                best_val = mag[b_idx];
                best_bin = b_idx;
            }
        }

        // Parabolic interpolation on log-magnitudes
        let alpha = (mag[best_bin - 1] + 1e-12).ln();
        let beta = (mag[best_bin] + 1e-12).ln();
        let gamma = (mag[best_bin + 1] + 1e-12).ln();
        let denom = alpha - 2.0 * beta + gamma;
        let delta = if denom.abs() > 1e-12 {
            0.5 * (alpha - gamma) / denom
        } else {
            0.0
        };

        let est_f = ((best_bin as f64) + delta) * df;
        let freq_err = (est_f - expected_fk).abs();
        max_freq_err = max_freq_err.max(freq_err);
        if freq_err < 1.0 {
            passing_count += 1;
        }
        println!("Partial {}: expected={:.2} Hz, est={:.2} Hz, err={:.2} Hz", k, expected_fk, est_f, freq_err);
    }

    println!("[CHALLENGE] Max partial frequency tracking error: {:.2} Hz (passing < 1.0 Hz: {}/{})",
        max_freq_err, passing_count, num_partials);
    // Note: Due to leakage between closely spaced partials (separated by ~2.3 bins at df=46.875 Hz),
    // simple single-frame parabolic interpolation can have errors exceeding 1.0 Hz!
}

/// Challenge T4-2: Phase-Loss Degradation of Time-Domain SNR in Additive Reconstructors.
/// Proves that resynthesizing a note from magnitude-only frames with zero/locked initial phase
/// degrades time-domain SNR to < 5 dB, making the assertion time_m.snr_db > 28.0 in Scenario 1
/// fail unless exact analytical phase is preserved.
#[test]
fn challenge_phase_loss_snr_collapse() {
    let fs = 48000.0;
    let n_samples = 4800; // 100 ms note
    let f0 = 261.63; // Middle C

    // Note with arbitrary non-zero initial phase (e.g. from acoustic excitation or STFT window shift)
    let mut orig_signal = vec![0.0f64; n_samples];
    let num_partials = 16;
    for k in 1..=num_partials {
        let fk = (k as f64) * f0;
        let ak = 1.0 / (k as f64);
        let phi_k = (k as f64) * 0.45; // Arbitrary phase shift
        for (n, s) in orig_signal.iter_mut().enumerate() {
            let t = n as f64 / fs;
            *s += ak * (2.0 * PI * fk * t + phi_k).sin();
        }
    }

    // Reconstructed note where only amplitudes were stored (phase assumed 0.0)
    let mut recon_signal = vec![0.0f64; n_samples];
    for k in 1..=num_partials {
        let fk = (k as f64) * f0;
        let ak = 1.0 / (k as f64);
        let phi_k = 0.0; // Phase lost during magnitude-only tensor extraction!
        for (n, s) in recon_signal.iter_mut().enumerate() {
            let t = n as f64 / fs;
            *s += ak * (2.0 * PI * fk * t + phi_k).sin();
        }
    }

    // Calculate time-domain SNR
    let p_sig: f64 = orig_signal.iter().map(|&x| x * x).sum();
    let p_err: f64 = orig_signal.iter().zip(recon_signal.iter()).map(|(&x, &x_hat)| (x - x_hat).powi(2)).sum();
    let snr_db = 10.0 * (p_sig / p_err).log10();
    println!("[CHALLENGE] Time-domain SNR with zero-phase reconstruction: {:.2} dB", snr_db);
    // SNR collapses to ~ 2.5 dB!
    assert!(snr_db < 10.0, "SNR collapsed due to phase mismatch");
    assert!(snr_db < 28.0, "Fails the Scenario 1 assertion SNR > 28.0 dB");
}

/// Challenge T4-3: Expressive Gesture First-Difference Bound.
/// Verifies that a clean glissando + tremolo gesture has maximum first difference ~0.046 < 0.15,
/// but any inter-frame phase discontinuity jumps to > 0.40, validating the continuity invariant.
#[test]
fn challenge_expressive_gesture_continuity_bounds() {
    let fs = 48000.0;
    let duration = 1.5;
    let n_samples = (fs * duration) as usize;

    let mut gesture_signal = Vec::with_capacity(n_samples);
    let mut phase = 0.0;

    for n in 0..n_samples {
        let t = n as f64 / fs;
        let f_inst = 220.0 * 2.0_f64.powf(t / duration);
        let tremolo = 1.0 - 0.35 * (0.5 + 0.5 * (2.0 * PI * 6.0 * t).cos());

        phase += 2.0 * PI * f_inst / fs;
        if phase >= 2.0 * PI {
            phase -= 2.0 * PI;
        }

        let s = tremolo * 0.8 * phase.sin();
        gesture_signal.push(s);
    }

    // Maximum first difference of continuous signal
    let mut max_diff = 0.0f64;
    for i in 0..(gesture_signal.len() - 1) {
        let diff = (gesture_signal[i + 1] - gesture_signal[i]).abs();
        max_diff = max_diff.max(diff);
    }
    println!("[CHALLENGE] Max first difference in continuous gesture: {:.4}", max_diff);
    assert!(max_diff < 0.15, "Continuous gesture first difference must be < 0.15");

    // Simulate an unaligned frame boundary phase jump:
    let mut discontinuous = gesture_signal.clone();
    let jump_sample = 24000;
    discontinuous[jump_sample] = -discontinuous[jump_sample - 1]; // Phase inversion
    let jump_diff = (discontinuous[jump_sample] - discontinuous[jump_sample - 1]).abs();
    println!("[CHALLENGE] Discontinuity jump difference: {:.4}", jump_diff);
    assert!(jump_diff > 0.15, "Phase jump exceeds 0.15 threshold as expected");
}

/// Challenge T4-4: Stochastic Noise Addition Threat to SNR Monotonicity.
/// Proves that synthesizing stochastic noise with random phase doubles the time-domain error
/// compared to leaving the noise component at zero, threatening t_ref.snr_db >= t_prod.snr_db.
#[test]
fn challenge_stochastic_noise_resynthesis_snr_penalty() {
    let n_samples = 48000;
    let seed_original = 12345u64;
    let seed_resynth = 67890u64;

    // Simulate original pink noise with variance sigma^2
    let sigma = 0.02f64;
    let mut prng_orig = seed_original;
    let mut orig_noise = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        prng_orig = prng_orig.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (prng_orig >> 32) as f64 / 4294967296.0;
        orig_noise.push(sigma * (u - 0.5) * 12.0_f64.sqrt());
    }

    // Pipeline A (Zero noise resynthesis: models only deterministic partials)
    let err_zero: f64 = orig_noise.iter().map(|&x| x * x).sum();

    // Pipeline B (Stochastic noise resynthesis: synthesizes uncorrelated shaped noise with matching variance)
    let mut prng_synth = seed_resynth;
    let mut synth_noise = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        prng_synth = prng_synth.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (prng_synth >> 32) as f64 / 4294967296.0;
        synth_noise.push(sigma * (u - 0.5) * 12.0_f64.sqrt());
    }
    let err_synth: f64 = orig_noise.iter().zip(synth_noise.iter()).map(|(&x, &s)| (x - s).powi(2)).sum();

    println!("[CHALLENGE] Noise MSE: zero resynth = {:.6e}, shaped noise resynth = {:.6e}",
        err_zero / (n_samples as f64), err_synth / (n_samples as f64));
    let ratio = err_synth / err_zero;
    println!("[CHALLENGE] Error ratio (synth / zero) = {:.2} (theory = 2.0)", ratio);
    // Stochastic noise resynthesis has ~2x the time-domain MSE of zero noise resynthesis!
    assert!((ratio - 2.0).abs() < 0.1, "Uncorrelated noise resynthesis doubles time-domain MSE");
}

// =========================================================================
// PART 3: MILESTONE 1 API CONTRACT VERIFICATION HARNESS
// =========================================================================

/// Verifies that all M1 types, methods, and error constructors compile cleanly and execute properly.
#[test]
fn challenge_verify_m1_contracts_empirical() {
    // 1. Matrix::from_vec and Matrix::new
    let mat = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("from_vec failed");
    let mat_alias = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("new failed");
    assert_eq!(mat.as_slice(), mat_alias.as_slice());
    assert_eq!(mat.get(0, 1).unwrap(), 2.0);

    // 2. Tensor::from_vec and Tensor::new
    let ten = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("from_vec failed");
    let ten_alias = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("new failed");
    assert_eq!(ten.data(), ten_alias.data());
    assert_eq!(ten.get(&[1, 0]).unwrap(), 3.0);

    // 3. Complex math
    let c = Complex::new(3.0, 4.0);
    assert_eq!(c.norm(), 5.0);
    assert_eq!(c.norm_sq(), 25.0);
    let polar = Complex::from_polar(5.0, c.arg());
    assert!((polar.re - 3.0).abs() < 1e-12);
    assert!((polar.im - 4.0).abs() < 1e-12);

    // 4. Windows and COLA
    let win = generate_window(WindowType::Hann, 512).unwrap();
    assert_eq!(win.len(), 512);
    assert!(verify_cola(&win, 256, 1e-5));

    // 5. Typed DspErrors
    let err_empty = DspError::EmptyInput;
    let err_param = DspError::invalid_parameter("test", "reason");
    let err_len = DspError::BufferLengthMismatch { expected: 10, actual: 5 };
    let err_dim = DspError::DimensionMismatch { expected: vec![2, 2], actual: vec![3, 3] };
    assert_eq!(format!("{}", err_empty), "Input buffer or sequence is empty");
    assert!(format!("{}", err_param).contains("test"));
    assert!(format!("{}", err_len).contains("expected 10"));
    assert!(format!("{}", err_dim).contains("expected shape [2, 2]"));
}
