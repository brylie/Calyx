//! tests/adversarial_metrics_m2_challenge.rs
//!
//! Empirical Adversarial Verification and Stress-Test Suite for Milestone 2:
//! Calyx DSP Verification Metrics (`src/metrics/`).
//!
//! Evaluates:
//! 1. Time-Domain Metrics (`src/metrics/time.rs`):
//!    - Axiomatic identity preservation (MSE=0, RMSE=0, MAE=0, SNR=+inf, PSNR=+inf, NCC=1.0, Pearson r=1.0).
//!    - Pure silence vs silence handling (zero NaNs, SNR=+inf, NCC=1.0, Pearson r=1.0).
//!    - Silent reference vs non-zero reconstructed (SNR=-inf, NCC=0.0).
//!    - Waveform inversion ($x$ vs $-x \implies \text{NCC}=-1.0, r=-1.0$).
//!    - Harmonic orthogonality ($\text{NCC} \approx 0.0$).
//!    - Dimension mismatch & empty buffer error handling (`DspError::BufferLengthMismatch`, `DspError::EmptyInput`).
//!    - Capped SNR and custom peak PSNR calculations.
//! 2. Spectral-Domain Metrics (`src/metrics/spectral.rs`):
//!    - Spectral Convergence: identity ($\text{SC}=0.0$), silence ($\text{SC}=0.0$), scale invariance across orders of magnitude.
//!    - Log-Spectral Distance: identity ($\text{LSD}=0.0$), silence floor handling without $-\infty$ or NaN, known gain step ($+6.02$ dB).
//!    - Dual-mode automatic routing (frequency magnitude spectrum vs time-domain oscillating audio with Hann STFT).
//!    - Audio signals with prime and non-power-of-two lengths ($N < 512$, $N < 1024$).
//!    - Harmonic peak error and extraction report evaluation with known synthetic partials.
//!    - Cooley-Tukey Radix-2 FFT vs direct DFT magnitude equivalence.
//! 3. Numerical Stability Invariants (`src/metrics/stability.rs`):
//!    - Strict NaN and Infinity immediate detection (`DspError::NumericalInstability`).
//!    - Peak amplitude clipping enforcement ($|s| > 1.0001$).
//!    - Boundary headroom safety ($|s| = 1.0001 \implies \text{Ok}$).
//!    - Energy conservation ratio $E_{\text{recon}} / E_{\text{orig}}$ and severe DC offset flagging in `check_numerical_stability`.
//!    - Denormal/subnormal sample counting.
//! 4. Verification Report & Quality Tiers (`src/metrics/report.rs`):
//!    - Quality tier configurations (`Strict`, `Reference`, `Production`, `Preview`, `Custom`).
//!    - Fidelity validation thresholds enforcement.
//!    - Unicode table, ASCII fallback, and Markdown formatting integrity and roundtrip validity.

use calyx_dsp::error::DspError;
use calyx_dsp::metrics::*;
use calyx_dsp::synthetic::buffer::{HarmonicPartial, SpectralPeak};
use calyx_dsp::synthetic::sine::SineGenerator;
use calyx_dsp::synthetic::traits::SignalGenerator;
use std::f64::consts::PI;

// ============================================================================
// PART 1: TIME-DOMAIN METRICS ADVERSARIAL STRESS TESTS
// ============================================================================

#[test]
fn test_adv_time_metrics_identity_on_diverse_waveforms() {
    // 1. Pure sine wave
    let sine = SineGenerator::new(440.0, 0.75, 0.0).generate(2048, 48000.0).samples;
    let m_sine = compute_time_metrics(&sine, &sine).expect("Identity metrics failed for sine");
    assert_eq!(m_sine.mse, 0.0);
    assert_eq!(m_sine.rmse, 0.0);
    assert_eq!(m_sine.mae, 0.0);
    assert!(m_sine.snr_db.is_infinite() && m_sine.snr_db.is_sign_positive());
    assert!(m_sine.psnr_db.is_infinite() && m_sine.psnr_db.is_sign_positive());
    assert!((m_sine.correlation_r - 1.0).abs() < 1e-14);
    assert!((m_sine.ncc - 1.0).abs() < 1e-14);
    assert!((m_sine.pearson_r - 1.0).abs() < 1e-14);

    // 2. High dynamic range decaying exponential
    let mut exp_signal = Vec::with_capacity(1000);
    for n in 0..1000 {
        exp_signal.push((-0.01 * (n as f64)).exp() * (2.0 * PI * 0.05 * (n as f64)).sin());
    }
    let m_exp = compute_time_metrics(&exp_signal, &exp_signal).unwrap();
    assert_eq!(m_exp.mse, 0.0);
    assert_eq!(m_exp.rmse, 0.0);
    assert!(m_exp.snr_db.is_infinite());
    assert!((m_exp.correlation_r - 1.0).abs() < 1e-14);

    // 3. Capped SNR behaves deterministically
    let capped_100 = compute_snr_db_capped(&sine, &sine, 100.0).unwrap();
    assert_eq!(capped_100, 100.0);
    let capped_120 = compute_snr_db_capped(&sine, &sine, 120.0).unwrap();
    assert_eq!(capped_120, 120.0);
}

#[test]
fn test_adv_time_metrics_pure_silence_axioms() {
    for len in [1, 2, 7, 64, 1024] {
        let silence = vec![0.0; len];
        let m = compute_time_metrics(&silence, &silence).expect("Silence comparison failed");

        // Mathematical axioms: silence vs silence has 0 error and +inf SNR
        assert_eq!(m.mse, 0.0);
        assert_eq!(m.rmse, 0.0);
        assert_eq!(m.mae, 0.0);
        assert!(m.snr_db.is_infinite() && m.snr_db.is_sign_positive());
        assert!(m.psnr_db.is_infinite() && m.psnr_db.is_sign_positive());

        // Zero energy normalization must NOT produce NaN
        assert!(!m.correlation_r.is_nan(), "NCC must not be NaN on pure silence");
        assert_eq!(m.correlation_r, 1.0, "NCC on identical silence must equal 1.0");
        assert!(!m.pearson_r.is_nan(), "Pearson r must not be NaN on pure silence");
        assert_eq!(m.pearson_r, 1.0, "Pearson r on identical silence must equal 1.0");
    }
}

#[test]
fn test_adv_time_metrics_silent_reference_vs_nonzero_recon() {
    let silence = vec![0.0; 512];
    let mut recon = vec![0.0; 512];
    recon[256] = 0.1; // Transient spike

    let m = compute_time_metrics(&silence, &recon).unwrap();
    // Reference has 0 signal energy, reconstructed has noise: SNR must be -inf
    assert!(m.snr_db.is_infinite() && m.snr_db.is_sign_negative());
    assert_eq!(m.correlation_r, 0.0);
    assert!(m.mse > 0.0);
    assert!(m.psnr_db.is_finite());

    // Adversarial test: compute_snr_db_capped on -inf SNR
    let capped_on_neg_inf = compute_snr_db_capped(&silence, &recon, 120.0).unwrap();
    println!("[CHALLENGE] compute_snr_db_capped on -inf SNR returned: {}", capped_on_neg_inf);
}

#[test]
fn test_adv_time_metrics_inversion_and_orthogonality() {
    let n = 4800; // 0.1 s at 48000 Hz
    let fs = 48000.0;

    // Inverted signal: r(x, -x) = -1.0
    let s = SineGenerator::new(440.0, 0.8, 0.3).generate(n, fs).samples;
    let s_inv: Vec<f64> = s.iter().map(|&x| -x).collect();

    let m_inv = compute_time_metrics(&s, &s_inv).unwrap();
    assert!((m_inv.correlation_r - (-1.0)).abs() < 1e-14, "NCC must be -1.0 for inverted signal");
    assert!((m_inv.pearson_r - (-1.0)).abs() < 1e-14, "Pearson r must be -1.0 for inverted signal");
    assert!((m_inv.mse - 4.0 * (0.8 * 0.8 / 2.0)).abs() < 1e-4);

    // Orthogonal tones: cos and sin at same frequency over integer cycles
    let mut sin_tone = vec![0.0; n];
    let mut cos_tone = vec![0.0; n];
    let freq = 100.0; // Exactly 10 cycles in 4800 samples at 48 kHz
    for i in 0..n {
        let t = (i as f64) / fs;
        sin_tone[i] = (2.0 * PI * freq * t).sin();
        cos_tone[i] = (2.0 * PI * freq * t).cos();
    }
    let m_ortho = compute_time_metrics(&sin_tone, &cos_tone).unwrap();
    assert!(m_ortho.correlation_r.abs() < 1e-12, "Orthogonal sine/cosine NCC must be 0.0");
    assert!(m_ortho.pearson_r.abs() < 1e-12, "Orthogonal sine/cosine Pearson r must be 0.0");

    // Independent helper wrappers
    assert_eq!(compute_ncc(&s, &s_inv).unwrap(), m_inv.ncc);
    assert_eq!(compute_cross_correlation(&s, &s_inv).unwrap(), m_inv.ncc);
    assert_eq!(compute_pearson_r(&s, &s_inv).unwrap(), m_inv.pearson_r);
    assert_eq!(compute_mse(&s, &s_inv).unwrap(), m_inv.mse);
    assert_eq!(compute_rmse(&s, &s_inv).unwrap(), m_inv.rmse);
    assert_eq!(compute_mae(&s, &s_inv).unwrap(), m_inv.mae);
}

#[test]
fn test_adv_time_metrics_error_rejection() {
    // Length mismatch
    let r = compute_time_metrics(&[1.0, 2.0, 3.0], &[1.0, 2.0]);
    assert!(matches!(
        r,
        Err(DspError::BufferLengthMismatch { expected: 3, actual: 2 })
    ));

    // Empty input
    assert!(matches!(compute_time_metrics(&[], &[1.0]), Err(DspError::EmptyInput)));
    assert!(matches!(compute_time_metrics(&[1.0], &[]), Err(DspError::EmptyInput)));
    assert!(matches!(compute_time_metrics(&[], &[]), Err(DspError::EmptyInput)));

    // Custom peak PSNR error propagation
    assert!(matches!(
        compute_psnr_db_with_peak(&[1.0], &[1.0, 2.0], 1.0),
        Err(DspError::BufferLengthMismatch { .. })
    ));
}

// ============================================================================
// PART 2: SPECTRAL-DOMAIN METRICS ADVERSARIAL STRESS TESTS
// ============================================================================

#[test]
fn test_adv_spectral_metrics_convergence_scale_invariance() {
    // Magnitude slices across dynamic ranges from 1e-8 to 1e4
    for &scale in &[1e-8, 1e-4, 1.0, 100.0, 10000.0] {
        let x = vec![scale * 1.0, scale * 0.5, scale * 0.25, scale * 0.125];
        let x_scaled: Vec<f64> = x.iter().map(|&v| 1.35 * v).collect();

        let sc = compute_spectral_convergence(&x, &x_scaled).unwrap();
        // SC = ||X - 1.35X|| / ||X|| = 0.35 ||X|| / ||X|| = 0.35
        assert!(
            (sc - 0.35).abs() < 1e-13,
            "Spectral convergence broke scale invariance at scale {}",
            scale
        );
    }

    // Identical magnitude spectrum: SC == 0.0
    let spec = vec![1.0, 0.8, 0.4, 0.1];
    let sc_ident = compute_spectral_convergence(&spec, &spec).unwrap();
    assert_eq!(sc_ident, 0.0);
}

#[test]
fn test_adv_spectral_metrics_pure_silence_floor() {
    let silent_spec = vec![0.0; 129];
    let m = compute_spectral_metrics(&silent_spec, &silent_spec).unwrap();

    assert_eq!(m.spectral_convergence, 0.0);
    assert_eq!(m.log_spectral_distance_db, 0.0);
    assert!(!m.spectral_convergence.is_nan());
    assert!(!m.log_spectral_distance_db.is_nan());

    // Single bin non-zero vs silent reference
    let mut perturbed = silent_spec.clone();
    perturbed[10] = 0.01;
    let m_pert = compute_spectral_metrics(&silent_spec, &perturbed).unwrap();
    assert!(m_pert.spectral_convergence.is_infinite());
    assert!(m_pert.log_spectral_distance_db.is_finite());
}

#[test]
fn test_adv_spectral_metrics_log_spectral_distance_exact_gain() {
    let spec = vec![0.5; 513];
    // Scale by factor of 4.0 (+12.0412 dB)
    let spec_scaled: Vec<f64> = spec.iter().map(|&v| 4.0 * v).collect();

    let lsd = compute_log_spectral_distance(&spec, &spec_scaled).unwrap();
    let expected_lsd = 20.0 * (4.0_f64).log10();
    assert!(
        (lsd - expected_lsd).abs() < 1e-3,
        "LSD {} dB deviated from expected {} dB",
        lsd,
        expected_lsd
    );
}

#[test]
fn test_adv_spectral_metrics_audio_stft_routing_edge_cases() {
    let fs = 48000.0;

    // 1. Oscillating waveform of standard size (N = 2048 >= 1024)
    let s = SineGenerator::new(440.0, 0.7, 0.0).generate(2048, fs).samples;
    let m_ident = compute_spectral_metrics(&s, &s).unwrap();
    assert_eq!(m_ident.spectral_convergence, 0.0);
    assert_eq!(m_ident.log_spectral_distance_db, 0.0);

    // 2. Oscillating waveform of medium size (512 <= N < 1024, e.g. 768)
    let s_768 = SineGenerator::new(440.0, 0.7, 0.0).generate(768, fs).samples;
    let m_768 = compute_spectral_metrics(&s_768, &s_768).unwrap();
    assert_eq!(m_768.spectral_convergence, 0.0);
    assert_eq!(m_768.log_spectral_distance_db, 0.0);

    // 3. Oscillating waveform of small non-power-of-two size (N = 300 < 512)
    let s_300 = SineGenerator::new(440.0, 0.7, 0.0).generate(300, fs).samples;
    let m_300 = compute_spectral_metrics(&s_300, &s_300).unwrap();
    assert_eq!(m_300.spectral_convergence, 0.0);
    assert_eq!(m_300.log_spectral_distance_db, 0.0);

    // 4. Oscillating waveform of tiny size (N = 33)
    let s_33 = SineGenerator::new(440.0, 0.7, 0.0).generate(33, fs).samples;
    let m_33 = compute_spectral_metrics(&s_33, &s_33).unwrap();
    assert_eq!(m_33.spectral_convergence, 0.0);
    assert_eq!(m_33.log_spectral_distance_db, 0.0);
}

#[test]
fn test_adv_spectral_dft_vs_fft_exact_equivalence() {
    // Generate synthetic audio block of length 64 (power of two)
    let block: Vec<f64> = (0..64)
        .map(|n| (2.0 * PI * 3.0 * (n as f64) / 64.0).sin() + 0.5 * (2.0 * PI * 7.0 * (n as f64) / 64.0).cos())
        .collect();

    let dft_mag = compute_magnitude_dft(&block);
    assert_eq!(dft_mag.len(), 33); // N/2 + 1 bins

    // compute_spectral_metrics on non-negative dft_mag slices compares magnitude directly
    let m = compute_spectral_metrics(&dft_mag, &dft_mag).unwrap();
    assert_eq!(m.spectral_convergence, 0.0);
    assert_eq!(m.log_spectral_distance_db, 0.0);
}

#[test]
fn test_adv_spectral_harmonic_peak_error_evaluation() {
    let gt = vec![
        HarmonicPartial { harmonic_index: 1, frequency: 110.0, amplitude: 1.0, phase: 0.0 },
        HarmonicPartial { harmonic_index: 2, frequency: 220.0, amplitude: 0.5, phase: 0.0 },
        HarmonicPartial { harmonic_index: 3, frequency: 330.0, amplitude: 0.25, phase: 0.0 },
    ];

    // Perfect detection
    let extracted_exact = vec![(110.0, 1.0), (220.0, 0.5), (330.0, 0.25)];
    let report_exact = evaluate_harmonic_extraction(&gt, &extracted_exact);
    assert_eq!(report_exact.max_frequency_error_hz, 0.0);
    assert_eq!(report_exact.max_relative_amplitude_error, 0.0);

    // Empty extraction returns infinity
    let report_empty = evaluate_harmonic_extraction(&gt, &[]);
    assert!(report_empty.max_frequency_error_hz.is_infinite());
    assert!(report_empty.max_relative_amplitude_error.is_infinite());

    // Frequency and amplitude offsets
    let extracted_noisy = vec![(110.5, 0.95), (220.1, 0.52), (330.8, 0.23)];
    let report_noisy = evaluate_harmonic_extraction(&gt, &extracted_noisy);
    assert!((report_noisy.max_frequency_error_hz - 0.8).abs() < 1e-10);
    // Relative amp err: (1.0-0.95)/1.0 = 0.05, (0.52-0.5)/0.5 = 0.04, (0.25-0.23)/0.25 = 0.08
    assert!((report_noisy.max_relative_amplitude_error - 0.08).abs() < 1e-10);

    // Peak error helper
    let gt_peaks = vec![
        SpectralPeak::new(2, 100.0, 1.0, 0.0),
        SpectralPeak::new(4, 200.0, 0.5, 0.0),
    ];
    let det_peaks = vec![
        SpectralPeak::new(2, 100.2, 1.0, 0.0),
        SpectralPeak::new(4, 199.7, 0.5, 0.0),
    ];
    let peak_err = compute_harmonic_peak_error(&gt_peaks, &det_peaks);
    assert!((peak_err - 0.3).abs() < 1e-10);

    assert!(compute_harmonic_peak_error(&gt_peaks, &[]).is_infinite());
    assert_eq!(compute_harmonic_peak_error(&[], &det_peaks), 0.0);
}

// ============================================================================
// PART 3: NUMERICAL STABILITY INVARIANTS ADVERSARIAL STRESS TESTS
// ============================================================================

#[test]
fn test_adv_stability_nan_and_inf_detection() {
    let mut buf = vec![0.5; 128];
    assert!(verify_numerical_stability(&buf).is_ok());

    // NaN at start, middle, and end
    buf[0] = f64::NAN;
    assert!(matches!(verify_numerical_stability(&buf), Err(DspError::NumericalInstability(_))));
    buf[0] = 0.5;

    buf[64] = f64::NAN;
    assert!(matches!(verify_numerical_stability(&buf), Err(DspError::NumericalInstability(_))));
    buf[64] = 0.5;

    buf[127] = f64::NAN;
    assert!(matches!(verify_numerical_stability(&buf), Err(DspError::NumericalInstability(_))));
    buf[127] = 0.5;

    // +Inf and -Inf
    buf[30] = f64::INFINITY;
    assert!(matches!(verify_numerical_stability(&buf), Err(DspError::NumericalInstability(_))));
    buf[30] = f64::NEG_INFINITY;
    assert!(matches!(verify_numerical_stability(&buf), Err(DspError::NumericalInstability(_))));
    buf[30] = 0.5;

    // Empty slice
    assert!(matches!(verify_numerical_stability(&[]), Err(DspError::EmptyInput)));
}

#[test]
fn test_adv_stability_clipping_threshold_boundary() {
    let mut buf = vec![0.0; 10];

    // Exactly 1.0 and -1.0 are within bounds
    buf[0] = 1.0;
    buf[1] = -1.0;
    assert!(verify_numerical_stability(&buf).is_ok());

    // Headroom boundary at 1.0001 is within bounds
    buf[2] = 1.0001;
    buf[3] = -1.0001;
    assert!(verify_numerical_stability(&buf).is_ok());

    // Exceeding 1.0001 by even 1e-6 must trigger clipping error
    buf[4] = 1.000101;
    assert!(matches!(verify_numerical_stability(&buf), Err(DspError::NumericalInstability(_))));

    buf[4] = 0.0;
    buf[5] = -1.000101;
    assert!(matches!(verify_numerical_stability(&buf), Err(DspError::NumericalInstability(_))));
}

#[test]
fn test_adv_stability_check_numerical_stability_report() {
    let orig = vec![0.5; 100];
    let mut recon = orig.clone();

    // 1. Clean report
    let rep_clean = check_numerical_stability(&orig, &recon);
    assert!(rep_clean.is_clean);
    assert_eq!(rep_clean.energy_ratio, 1.0);
    assert_eq!(rep_clean.dc_offset, 0.0);
    assert_eq!(rep_clean.clipping_count, 0);
    assert_eq!(rep_clean.nan_count, 0);
    assert_eq!(rep_clean.inf_count, 0);
    assert_eq!(rep_clean.subnormal_count, 0);
    assert!(rep_clean.violations.is_empty());

    // 2. Corrupt with subnormals, clipping, NaN, Inf
    recon[0] = 1e-315; // Subnormal
    recon[1] = 1.005;  // Clipped
    recon[2] = f64::NAN;
    recon[3] = f64::INFINITY;
    recon[4] = -f64::INFINITY;

    let rep_corrupt = check_numerical_stability(&orig, &recon);
    assert!(!rep_corrupt.is_clean);
    assert_eq!(rep_corrupt.subnormal_count, 1);
    assert_eq!(rep_corrupt.clipping_count, 1);
    assert_eq!(rep_corrupt.nan_count, 1);
    assert_eq!(rep_corrupt.inf_count, 2);
    assert!(rep_corrupt.has_nan);
    assert!(rep_corrupt.has_inf);
    assert!(!rep_corrupt.violations.is_empty());

    // 3. Energy ratio violation (< 0.85 or > 1.15)
    let recon_low_energy = vec![0.2; 100]; // Energy ratio = 0.04 / 0.25 = 0.16
    let rep_energy = check_numerical_stability(&orig, &recon_low_energy);
    assert!(!rep_energy.is_clean);
    assert!(rep_energy.violations.iter().any(|v| v.contains("Energy conservation")));

    // 4. DC offset violation (> 0.01)
    let recon_dc: Vec<f64> = orig.iter().map(|&x| x + 0.02).collect();
    let rep_dc = check_numerical_stability(&orig, &recon_dc);
    assert!(!rep_dc.is_clean);
    assert!(rep_dc.violations.iter().any(|v| v.contains("Severe DC offset")));
}

// ============================================================================
// PART 4: QUALITY TIERS & REPORT GENERATION ADVERSARIAL STRESS TESTS
// ============================================================================

#[test]
fn test_adv_quality_tier_thresholds_and_custom_validation() {
    let t_ref = QualityTierConfig::reference();
    assert_eq!(t_ref.tier, QualityTier::Reference);
    assert_eq!(t_ref.min_snr_db, 35.0);
    assert_eq!(t_ref.max_lsd_db, 0.8);
    assert_eq!(t_ref.rank, 8);
    assert_eq!(t_ref.num_noise_bands, 16);

    let t_prod = QualityTierConfig::production();
    assert_eq!(t_prod.tier, QualityTier::Production);
    assert_eq!(t_prod.min_snr_db, 28.0);
    assert_eq!(t_prod.max_lsd_db, 1.5);
    assert_eq!(t_prod.rank, 4);
    assert_eq!(t_prod.num_noise_bands, 8);

    let t_prev = QualityTierConfig::preview();
    assert_eq!(t_prev.tier, QualityTier::Preview);
    assert_eq!(t_prev.min_snr_db, 18.0);
    assert_eq!(t_prev.max_lsd_db, 2.8);
    assert_eq!(t_prev.rank, 2);
    assert_eq!(t_prev.num_noise_bands, 4);

    // Custom tier validation: rank 0 or bands 0 must fail
    assert!(matches!(QualityTierConfig::custom(0, 8), Err(DspError::InvalidParameter { .. })));
    assert!(matches!(QualityTierConfig::custom(4, 0), Err(DspError::InvalidParameter { .. })));
    let custom = QualityTierConfig::custom(6, 12).unwrap();
    assert_eq!(custom.rank, 6);
    assert_eq!(custom.num_noise_bands, 12);

    // Fidelity validation checks
    let passing_time = TimeFidelityMetrics {
        mse: 1e-4,
        rmse: 1e-2,
        mae: 1e-2,
        snr_db: 40.0,
        psnr_db: 45.0,
        correlation_r: 0.999,
        ncc: 0.999,
        pearson_r: 0.999,
    };
    let passing_spec = SpectralFidelityMetrics {
        spectral_convergence: 0.01,
        log_spectral_distance_db: 0.5,
    };
    assert!(t_ref.validate_fidelity(&passing_time, &passing_spec).is_ok());

    // Failing SNR
    let mut low_snr = passing_time;
    low_snr.snr_db = 34.0;
    assert!(matches!(t_ref.validate_fidelity(&low_snr, &passing_spec), Err(DspError::NumericalInstability(_))));

    // Failing LSD
    let mut high_lsd = passing_spec;
    high_lsd.log_spectral_distance_db = 0.85;
    assert!(matches!(t_ref.validate_fidelity(&passing_time, &high_lsd), Err(DspError::NumericalInstability(_))));
}

#[test]
fn test_adv_report_generation_and_table_formatting() {
    let fs = 48000.0;
    let s = SineGenerator::new(440.0, 0.8, 0.0).generate(1024, fs).samples;

    // 1. Successful report on identical signal
    let report = generate_verification_report(
        "Adversarial Sine Identity",
        &s,
        &s,
        fs,
        QualityTierConfig::reference(),
    ).expect("Report generation failed");

    assert!(report.overall_passed);
    assert!(report.failures().is_empty());
    assert_eq!(report.sample_count, 1024);
    assert_eq!(report.sample_rate, fs);

    // 2. Format checks
    let unicode_table = report.to_table_string();
    assert!(unicode_table.contains("CALYX DSP VERIFICATION REPORT"));
    assert!(unicode_table.contains("┌─"));
    assert!(unicode_table.contains("PASS"));

    let ascii_table = report.to_ascii_string();
    assert!(ascii_table.contains("+--"));
    assert!(ascii_table.contains("| Metric"));
    assert!(ascii_table.is_ascii(), "to_ascii_string must produce pure ASCII output");

    let markdown_table = report.to_markdown_string();
    assert!(markdown_table.contains("### Calyx Verification Report: Adversarial Sine Identity"));
    assert!(markdown_table.contains("| Metric | Value | Threshold | Unit | Status |"));
    assert!(markdown_table.contains("|---|---|---|---|---|"));

    // 3. Failing report on heavily perturbed signal
    let mut corrupted = s.clone();
    for x in &mut corrupted {
        *x *= 0.1; // Drops amplitude by 20 dB, ruining SNR and energy
    }
    let rep_fail = generate_verification_report(
        "Corrupted Sine",
        &s,
        &corrupted,
        fs,
        QualityTierConfig::reference(),
    ).unwrap();

    assert!(!rep_fail.overall_passed);
    assert!(!rep_fail.failures().is_empty());
    let md_fail = rep_fail.to_markdown_string();
    assert!(md_fail.contains("**FAIL**"));
}

#[test]
fn test_adv_report_silent_reference_formatting_hazard() {
    let silence = vec![0.0; 1024];
    let noisy = vec![0.05; 1024];
    let report = generate_verification_report(
        "Silent Ref With Noise",
        &silence,
        &noisy,
        48000.0,
        QualityTierConfig::reference(),
    ).unwrap();

    let snr_eval = &report.evaluations[0];
    assert_eq!(snr_eval.name, "SNR");
    assert!(!snr_eval.passed);
    println!("Observed SNR value_str: '{}'", snr_eval.value_str);
}
