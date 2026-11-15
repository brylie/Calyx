//! tests/adversarial_m3_stress_challenge.rs
//!
//! Empirical Adversarial Stress-Test Suite for Milestone 3:
//! Spectral Analysis, FFT & Partial Tracking Engine.
//!
//! Evaluates:
//! 1. Sub-bin parabolic interpolation accuracy on off-bin sinusoids across fractional bins (k + 0.25, k + 0.5, k + 0.75).
//! 2. Boundary conditions: DC bin (0.0 Hz), Nyquist bin (fs/2), flat spectrum plateau rejection, silence, degeneracies.
//! 3. Inharmonic partial tracking on synthetic stiff-string signals (beta = 0.0004), verifying frequency accuracy < 1.0 Hz.
//! 4. Super-Nyquist partial culling (f_k <= fs / 2.0).
//! 5. Stochastic residual extraction: perfect match, zero-energy silence, overshoot clamping, single subband.
//! 6. STFT edge cases: audio shorter than window size, non-power-of-two remainder audio, OLA reconstruction fidelity.
//! 7. Numerical stability: NaN/Inf rejection, extreme amplitudes, and absence of panics.

use calyx_dsp::error::DspError;
use calyx_dsp::math::{Complex, WindowType, generate_window};
use calyx_dsp::spectral::{
    Dft, Fft, HarmonicPartialTracker, PeakDetector, SpectralPeak,
    StochasticResidualExtractor, Stft,
};
use std::f64::consts::PI;

// ============================================================================
// Helper: Synthesize Hann-windowed tone and compute one-sided magnitude spectrum
// ============================================================================
fn compute_windowed_mag_spectrum(
    freq: f64,
    amp: f64,
    phase: f64,
    n_fft: usize,
    sample_rate: f64,
) -> Vec<f64> {
    let window = generate_window(WindowType::Hann, n_fft).unwrap();
    let mut buf = vec![Complex::zero(); n_fft];
    for n in 0..n_fft {
        let t = (n as f64) / sample_rate;
        let s = amp * (2.0 * PI * freq * t + phase).sin();
        buf[n] = Complex::new(s * window[n], 0.0);
    }
    Fft::forward_complex_radix2(&mut buf).unwrap();
    let half_bins = n_fft / 2 + 1;
    buf[0..half_bins].iter().map(|c| c.norm()).collect()
}

// ============================================================================
// 1. Sub-bin Parabolic Interpolation Accuracy on Off-Bin Sinusoids
// ============================================================================

#[test]
fn challenge_sub_bin_interpolation_accuracy_fractional_bins() {
    let fs = 48000.0;
    let n_fft = 2048;
    let df = fs / (n_fft as f64); // 23.4375 Hz per bin

    // Test across several integer bins and fractional offsets
    let test_bins = [10.0, 25.0, 50.0, 100.0, 200.0];
    let test_fractions = [0.0, 0.15, 0.25, 0.33, 0.50, 0.67, 0.75, 0.85];

    let mut max_freq_error = 0.0;
    let mut max_bin_error = 0.0;

    for &base_k in &test_bins {
        for &frac in &test_fractions {
            let true_bin = base_k + frac;
            let true_freq = true_bin * df;
            let mag_spec = compute_windowed_mag_spectrum(true_freq, 1.0, 0.0, n_fft, fs);

            let peaks = PeakDetector::detect_peaks(&mag_spec, fs);
            assert!(
                !peaks.is_empty(),
                "Failed to detect peak for bin {:.2} (freq {:.2} Hz)",
                true_bin,
                true_freq
            );

            // Find closest peak
            let best_peak = peaks
                .iter()
                .min_by(|a, b| {
                    (a.frequency_hz - true_freq)
                        .abs()
                        .partial_cmp(&(b.frequency_hz - true_freq).abs())
                        .unwrap()
                })
                .unwrap();

            let freq_error = (best_peak.frequency_hz - true_freq).abs();
            let bin_error = freq_error / df;

            if freq_error > max_freq_error {
                max_freq_error = freq_error;
            }
            if bin_error > max_bin_error {
                max_bin_error = bin_error;
            }

            // Parabolic interpolation on Hann log-magnitude has known theoretical
            // error bounded strictly below 0.06 bins for any fractional offset.
            assert!(
                bin_error < 0.08,
                "Sub-bin interpolation error too large at bin {:.2}: bin_error={:.4} bins, freq_error={:.3} Hz (df={:.3} Hz)",
                true_bin, bin_error, freq_error, df
            );

            // Estimated frequency must be very close to ground truth
            assert!(
                freq_error < 2.0, // df * 0.08 = 23.4375 * 0.08 = 1.875 Hz
                "Frequency error too large at {:.2} Hz: got {:.2} Hz (diff {:.3} Hz)",
                true_freq, best_peak.frequency_hz, freq_error
            );
        }
    }

    println!(
        "Sub-bin interpolation maximum error: bin_error = {:.4} bins, freq_error = {:.4} Hz",
        max_bin_error, max_freq_error
    );
}

#[test]
fn challenge_sub_bin_interpolation_half_bin_straddling() {
    // 0.5 fractional bin straddling is mathematically the worst case for standard FFT
    let fs = 44100.0;
    let n_fft = 1024;
    let df = fs / (n_fft as f64);

    let k = 40.5;
    let target_freq = k * df;
    let mag_spec = compute_windowed_mag_spectrum(target_freq, 2.5, PI / 4.0, n_fft, fs);

    let peaks = PeakDetector::detect_peaks(&mag_spec, fs);
    assert_eq!(peaks.len(), 1, "Should detect exactly one peak for clean sinusoid");
    let peak = &peaks[0];

    let freq_err = (peak.frequency_hz - target_freq).abs();
    assert!(
        freq_err < df * 0.06,
        "Worst-case straddling error: expected {:.2} Hz, got {:.2} Hz (err={:.2} Hz, df={:.2} Hz)",
        target_freq, peak.frequency_hz, freq_err, df
    );
    assert!(!peak.amplitude.is_nan());
    assert!(peak.amplitude > 0.0);
}

// ============================================================================
// 2. Boundary Checks: DC, Nyquist, Plateaus, and Degenerate Spectra
// ============================================================================

#[test]
fn challenge_boundary_dc_bin_isolation() {
    let fs = 48000.0;
    let mut spec = vec![0.0; 513];
    spec[0] = 42.0; // DC component

    let peaks = PeakDetector::detect_peaks(&spec, fs);
    assert_eq!(peaks.len(), 1, "DC component must be detected as 1 peak");
    assert_eq!(peaks[0].bin_index, 0);
    assert_eq!(peaks[0].frequency_hz, 0.0);
    assert!((peaks[0].amplitude - 42.0).abs() < 1e-12);
}

#[test]
fn challenge_boundary_nyquist_bin_isolation() {
    let fs = 48000.0;
    let mut spec = vec![0.0; 513];
    spec[512] = 99.0; // Nyquist bin

    let peaks = PeakDetector::detect_peaks(&spec, fs);
    assert_eq!(peaks.len(), 1, "Nyquist component must be detected as 1 peak");
    assert_eq!(peaks[0].bin_index, 512);
    assert_eq!(peaks[0].frequency_hz, fs / 2.0);
    assert!((peaks[0].amplitude - 99.0).abs() < 1e-12);
}

#[test]
fn challenge_boundary_simultaneous_dc_and_nyquist() {
    let fs = 48000.0;
    let mut spec = vec![0.0; 513];
    spec[0] = 10.0;
    spec[512] = 20.0;

    let peaks = PeakDetector::detect_peaks(&spec, fs);
    assert_eq!(peaks.len(), 2);
    assert_eq!(peaks[0].frequency_hz, 0.0);
    assert_eq!(peaks[1].frequency_hz, 24000.0);
}

#[test]
fn challenge_boundary_flat_spectrum_plateau_rejection() {
    let fs = 48000.0;

    // Flat plateau of 1.0
    let flat_ones = vec![1.0; 513];
    assert!(
        PeakDetector::detect_peaks(&flat_ones, fs).is_empty(),
        "Flat spectrum of 1.0 must produce zero peaks"
    );

    // Flat plateau of 100.0
    let flat_hundreds = vec![100.0; 1025];
    assert!(
        PeakDetector::detect_peaks(&flat_hundreds, fs).is_empty(),
        "Flat spectrum of 100.0 must produce zero peaks"
    );

    // Pure silence
    let silence = vec![0.0; 513];
    assert!(
        PeakDetector::detect_peaks(&silence, fs).is_empty(),
        "Zero-energy silence must produce zero peaks"
    );

    // Flat noise floor below threshold
    let low_noise = vec![1e-7; 513];
    assert!(
        PeakDetector::detect_peaks(&low_noise, fs).is_empty(),
        "Below-threshold spectrum must produce zero peaks"
    );
}

#[test]
fn challenge_boundary_minimal_and_degenerate_spectrum_sizes() {
    let fs = 48000.0;

    // Empty spectrum
    assert!(PeakDetector::detect_peaks(&[], fs).is_empty());

    // Single bin spectrum
    assert!(PeakDetector::detect_peaks(&[10.0], fs).is_empty());

    // 2-bin spectrum: DC > Nyquist
    let spec2_dc = vec![10.0, 1.0];
    let peaks2_dc = PeakDetector::detect_peaks(&spec2_dc, fs);
    assert_eq!(peaks2_dc.len(), 1);
    assert_eq!(peaks2_dc[0].bin_index, 0);

    // 2-bin spectrum: Nyquist > DC
    let spec2_nyq = vec![1.0, 10.0];
    let peaks2_nyq = PeakDetector::detect_peaks(&spec2_nyq, fs);
    assert_eq!(peaks2_nyq.len(), 1);
    assert_eq!(peaks2_nyq[0].bin_index, 1);

    // 2-bin spectrum: equal
    let spec2_eq = vec![10.0, 10.0];
    assert!(PeakDetector::detect_peaks(&spec2_eq, fs).is_empty());

    // 3-bin spectrum: peak in middle
    let spec3 = vec![1.0, 10.0, 1.0];
    let peaks3 = PeakDetector::detect_peaks(&spec3, fs);
    assert_eq!(peaks3.len(), 1);
    assert_eq!(peaks3[0].bin_index, 1);
    assert_eq!(peaks3[0].frequency_hz, fs / 4.0); // 3 bins -> n_fft = 2*(3-1) = 4; bin 1 = fs/4

    // Non-power-of-two length spectrum (e.g. 333 bins)
    let mut spec_odd = vec![0.0; 333];
    spec_odd[100] = 5.0;
    spec_odd[101] = 15.0;
    spec_odd[102] = 5.0;
    let peaks_odd = PeakDetector::detect_peaks(&spec_odd, fs);
    assert_eq!(peaks_odd.len(), 1);
    assert_eq!(peaks_odd[0].bin_index, 101);
}

#[test]
fn challenge_boundary_nan_and_inf_spectrum_immunity() {
    let fs = 48000.0;
    let mut spec = vec![0.0; 257];
    spec[50] = f64::NAN;
    spec[51] = 10.0;
    spec[52] = f64::INFINITY;
    spec[100] = 10.0; // Valid local peak
    spec[99] = 2.0;
    spec[101] = 2.0;

    let peaks = PeakDetector::detect_peaks(&spec, fs);
    // Must not panic, must not return NaN peak, must find valid peak at 100
    for p in &peaks {
        assert!(p.frequency_hz.is_finite());
        assert!(p.amplitude.is_finite());
    }
    assert!(peaks.iter().any(|p| p.bin_index == 100));

    // Test with non-finite sample rate
    let nan_fs_peaks = PeakDetector::detect_peaks(&spec, f64::NAN);
    // Even if sample rate is NaN, check if any panics occur
    for p in &nan_fs_peaks {
        assert!(!p.amplitude.is_nan());
    }

    // Negative sample rate must yield empty
    let neg_fs_peaks = PeakDetector::detect_peaks(&spec, -48000.0);
    assert!(neg_fs_peaks.is_empty());

    // Zero sample rate must yield empty
    let zero_fs_peaks = PeakDetector::detect_peaks(&spec, 0.0);
    assert!(zero_fs_peaks.is_empty());
}

#[test]
fn challenge_tracker_trajectory_detuned_drift_behavior() {
    // Investigate track_trajectory fallback logic when measured frequencies drift from theoretical
    let f0 = 100.0;
    let beta = 0.0;
    let tracker = HarmonicPartialTracker::new(f0, 4, beta);

    // Frame 1: peaks closely match theoretical (error = 0.2 Hz)
    let frame1_partials = vec![
        calyx_dsp::spectral::HarmonicPartial {
            harmonic_index: 1,
            frequency: 100.2,
            amplitude: 1.0,
            phase: 0.0,
        },
        calyx_dsp::spectral::HarmonicPartial {
            harmonic_index: 2,
            frequency: 200.2,
            amplitude: 1.0,
            phase: 0.0,
        },
    ];

    let traj1 = calyx_dsp::spectral::SpectralTrajectory {
        sample_rate: 48000.0,
        hop_size: 256,
        frames: vec![calyx_dsp::spectral::SpectralFrame {
            timestamp: 0.0,
            fundamental_freq: 100.0,
            partials: frame1_partials,
            residual: calyx_dsp::spectral::NoiseBandEnvelope {
                band_energies: vec![0.0; 16],
                noise_ratio: 0.0,
            },
        }],
    };

    let res1 = tracker.track_trajectory(&traj1);
    // Partials 1 and 2 are present; missing partials 3 and 4 are omitted
    assert_eq!(res1.len(), 2);
    assert!((res1[0].frequency - 100.2).abs() < 1e-10);
    assert!((res1[1].frequency - 200.2).abs() < 1e-10);

    // Frame 2: Detuned peaks with drift of 2.5 Hz (e.g. vibrato or tuning deviation)
    let frame2_partials = vec![
        calyx_dsp::spectral::HarmonicPartial {
            harmonic_index: 1,
            frequency: 102.5,
            amplitude: 1.0,
            phase: 0.0,
        },
    ];

    let traj2 = calyx_dsp::spectral::SpectralTrajectory {
        sample_rate: 48000.0,
        hop_size: 256,
        frames: vec![calyx_dsp::spectral::SpectralFrame {
            timestamp: 0.0,
            fundamental_freq: 100.0,
            partials: frame2_partials,
            residual: calyx_dsp::spectral::NoiseBandEnvelope {
                band_energies: vec![0.0; 16],
                noise_ratio: 0.0,
            },
        }],
    };

    let res2 = tracker.track_trajectory(&traj2);
    // Authentically reports measured frequency (102.5 Hz) rather than theoretical (100.0 Hz)
    assert_eq!(res2.len(), 1);
    assert!((res2[0].frequency - 102.5).abs() < 1e-10);
}

#[test]
fn challenge_residual_length_mismatch_and_asymmetric_inputs() {
    // Original spectrum longer than deterministic spectrum
    let orig = vec![2.0; 100];
    let det = vec![1.0; 40];
    let res = StochasticResidualExtractor::extract_residual(&orig, &det);
    assert_eq!(res.band_energies.len(), 40);
    assert!(res.noise_ratio > 0.0);

    // Deterministic spectrum longer than original spectrum
    let orig2 = vec![2.0; 30];
    let det2 = vec![1.0; 80];
    let res2 = StochasticResidualExtractor::extract_residual(&orig2, &det2);
    assert_eq!(res2.band_energies.len(), 30);
    assert!(res2.noise_ratio > 0.0);
}

#[test]
fn challenge_stft_ola_various_window_types() {
    let n_samples = 2048;
    let fs = 48000.0;
    let sig: Vec<f64> = (0..n_samples)
        .map(|i| (2.0 * PI * 440.0 * (i as f64) / fs).sin())
        .collect();

    // Hann window with 75% overlap (hop = win/4 = 128)
    let stft_hann = Stft::new(512, 128, WindowType::Hann).unwrap();
    let spec_hann = stft_hann.forward(&sig).unwrap();
    let recon_hann = stft_hann.inverse(&spec_hann, Some(n_samples)).unwrap();
    for i in 512..(n_samples - 512) {
        assert!((sig[i] - recon_hann[i]).abs() < 1e-6);
    }

    // Hamming window with 50% overlap (hop = win/2 = 256)
    let stft_hamm = Stft::new(512, 256, WindowType::Hamming).unwrap();
    let spec_hamm = stft_hamm.forward(&sig).unwrap();
    let recon_hamm = stft_hamm.inverse(&spec_hamm, Some(n_samples)).unwrap();
    for i in 512..(n_samples - 512) {
        assert!((sig[i] - recon_hamm[i]).abs() < 1e-5);
    }
}

// ============================================================================
// 3. Inharmonic Partial Tracking on Synthetic Stiff-String Signals
// ============================================================================

#[test]
fn challenge_inharmonic_partial_tracking_piano_wire_accuracy() {
    let fs = 48000.0;
    let duration = 0.5;
    let n_samples = (fs * duration) as usize;
    let f0 = 110.0;
    let beta = 0.0004;
    let num_partials = 24;

    // Synthesize physical stiff-string multi-tone signal
    let mut signal = vec![0.0; n_samples];
    for k in 1..=num_partials {
        let k_f = k as f64;
        let freq = k_f * f0 * (1.0 + beta * k_f * k_f).sqrt();
        let amp = 1.0 / (k_f.sqrt());
        for (i, s) in signal.iter_mut().enumerate() {
            let t = (i as f64) / fs;
            *s += amp * (2.0 * PI * freq * t).sin();
        }
    }

    // High resolution STFT
    let stft = Stft::new(2048, 512, WindowType::Hann).unwrap();
    let trajectory = stft.process_with_sample_rate(&signal, fs).unwrap();
    assert!(!trajectory.frames.is_empty());

    // Single-frame tracking directly from detected peaks
    let tracker = HarmonicPartialTracker::new(f0, num_partials, beta);
    let mid_frame = &trajectory.frames[trajectory.frames.len() / 2];

    // Convert frame partials back to spectral peaks to test track_partials
    let peaks: Vec<SpectralPeak> = mid_frame
        .partials
        .iter()
        .map(|p| SpectralPeak {
            bin_index: 0,
            frequency_hz: p.frequency,
            amplitude: p.amplitude,
            phase_rad: p.phase,
        })
        .collect();

    let tracked_frame = tracker.track_partials(&peaks, fs);
    assert_eq!(
        tracked_frame.len(),
        num_partials,
        "Should track all 24 partials in steady-state frame"
    );

    for (idx, p) in tracked_frame.iter().enumerate() {
        let k = (idx + 1) as f64;
        let expected_f = k * f0 * (1.0 + beta * k * k).sqrt();
        let error = (p.frequency - expected_f).abs();
        assert!(
            error < 1.0,
            "Frame partial {} frequency error exceeded: expected {:.3} Hz, got {:.3} Hz (err={:.3} Hz)",
            k, expected_f, p.frequency, error
        );
    }

    // Trajectory tracking
    let trajectory_partials = tracker.track_trajectory(&trajectory);
    assert_eq!(trajectory_partials.len(), num_partials);
    for (idx, p) in trajectory_partials.iter().enumerate() {
        let k = (idx + 1) as f64;
        let expected_f = k * f0 * (1.0 + beta * k * k).sqrt();
        let error = (p.frequency - expected_f).abs();
        assert!(
            error < 1.0,
            "Trajectory partial {} error exceeded: expected {:.3} Hz, got {:.3} Hz (err={:.3} Hz)",
            k, expected_f, p.frequency, error
        );
    }
}

#[test]
fn challenge_inharmonic_tracker_missing_harmonics_robustness() {
    // Square wave has only ODD harmonics (1, 3, 5, 7, ...). Even harmonics are missing.
    let fs = 48000.0;
    let f0 = 200.0;
    let beta = 0.0005;
    let tracker = HarmonicPartialTracker::new(f0, 8, beta);

    // Provide peaks only for odd harmonics 1, 3, 5, 7
    let mut peaks = Vec::new();
    for &k in &[1, 3, 5, 7] {
        let f = tracker.theoretical_frequency(k);
        peaks.push(SpectralPeak {
            bin_index: 0,
            frequency_hz: f + 0.1, // slight perturbation
            amplitude: 1.0 / (k as f64),
            phase_rad: 0.0,
        });
    }

    let tracked = tracker.track_partials(&peaks, fs);
    // Should successfully track the 4 odd partials without falsely latching missing even partials
    assert_eq!(tracked.len(), 4);
    assert_eq!(tracked[0].harmonic_index, 1);
    assert_eq!(tracked[1].harmonic_index, 3);
    assert_eq!(tracked[2].harmonic_index, 5);
    assert_eq!(tracked[3].harmonic_index, 7);
}

// ============================================================================
// 4. Super-Nyquist Partial Culling Verification
// ============================================================================

#[test]
fn challenge_super_nyquist_partial_culling_harmonic_and_inharmonic() {
    let fs = 44100.0;
    let nyquist = fs / 2.0; // 22050.0 Hz

    // Harmonic culling: f0 = 5000.0, 10 partials.
    // Partials: 5k, 10k, 15k, 20k (<= 22050), 25k (> 22050) -> max 4 partials
    let tracker_harm = HarmonicPartialTracker::new(5000.0, 10, 0.0);
    let fake_peaks: Vec<SpectralPeak> = (1..=10)
        .map(|k| SpectralPeak {
            bin_index: k * 10,
            frequency_hz: (k as f64) * 5000.0,
            amplitude: 1.0,
            phase_rad: 0.0,
        })
        .collect();

    let tracked_harm = tracker_harm.track_partials(&fake_peaks, fs);
    assert_eq!(tracked_harm.len(), 4);
    assert!(tracked_harm.iter().all(|p| p.frequency <= nyquist));

    // Inharmonic culling: beta causes partials to accelerate toward Nyquist much faster
    let f0 = 3000.0;
    let beta = 0.01;
    let tracker_inharm = HarmonicPartialTracker::new(f0, 10, beta);

    // Compute expected frequencies
    // k=1: 3000 * sqrt(1.01) = 3014.9
    // k=2: 6000 * sqrt(1.04) = 6118.8
    // k=3: 9000 * sqrt(1.09) = 9396.2
    // k=4: 12000 * sqrt(1.16) = 12924.3
    // k=5: 15000 * sqrt(1.25) = 16770.5
    // k=6: 18000 * sqrt(1.36) = 20991.4 (<= 22050)
    // k=7: 21000 * sqrt(1.49) = 25634.1 (> 22050) -> culled!
    let inharm_peaks: Vec<SpectralPeak> = (1..=10)
        .map(|k| SpectralPeak {
            bin_index: k * 10,
            frequency_hz: tracker_inharm.theoretical_frequency(k),
            amplitude: 1.0,
            phase_rad: 0.0,
        })
        .collect();

    let tracked_inharm = tracker_inharm.track_partials(&inharm_peaks, fs);
    assert_eq!(tracked_inharm.len(), 6);
    assert!(tracked_inharm.iter().all(|p| p.frequency <= nyquist));
}

// ============================================================================
// 5. Stochastic Residual Extraction Edge Cases
// ============================================================================

#[test]
fn challenge_residual_extraction_perfect_match() {
    let spec = vec![1.2345; 64];
    let res = StochasticResidualExtractor::extract_residual(&spec, &spec);
    assert_eq!(res.band_energies.len(), 64);
    for &e in &res.band_energies {
        assert_eq!(e, 0.0);
    }
    assert_eq!(res.noise_ratio, 0.0);
}

#[test]
fn challenge_residual_extraction_zero_energy_silence_division_by_zero() {
    let silence = vec![0.0; 128];
    let res = StochasticResidualExtractor::extract_residual(&silence, &silence);
    assert!(!res.noise_ratio.is_nan(), "Noise ratio must not be NaN");
    assert!(!res.noise_ratio.is_infinite(), "Noise ratio must not be Inf");
    assert_eq!(res.noise_ratio, 0.0);
    for &e in &res.band_energies {
        assert_eq!(e, 0.0);
    }
}

#[test]
fn challenge_residual_extraction_subnormal_energy_floor() {
    let subnormal = vec![1e-16; 64];
    let res = StochasticResidualExtractor::extract_residual(&subnormal, &subnormal);
    assert!(!res.noise_ratio.is_nan());
    assert_eq!(res.noise_ratio, 0.0);
}

#[test]
fn challenge_residual_extraction_deterministic_overshoot_clamping() {
    // When deterministic model over-predicts magnitude, residual energy (S^2 - D^2)
    // must strictly clamp to >= 0.0 without going negative
    let orig = vec![1.0; 32];
    let det = vec![5.0; 32]; // heavy overshoot
    let res = StochasticResidualExtractor::extract_residual(&orig, &det);
    for &e in &res.band_energies {
        assert!(e >= 0.0, "Residual energy must be non-negative");
        assert_eq!(e, 0.0);
    }
    assert_eq!(res.noise_ratio, 0.0);
}

#[test]
fn challenge_residual_extraction_single_subband() {
    let extractor = StochasticResidualExtractor::new(1).unwrap();
    let orig = vec![2.0; 16];
    let det = vec![1.0; 16];
    let res = extractor.extract(&orig, &det);

    assert_eq!(res.band_energies.len(), 1);
    // orig_e = 4.0 * 16 = 64.0; det_e = 1.0 * 16 = 16.0; res_e = 48.0
    assert!((res.band_energies[0] - 48.0).abs() < 1e-10);
    assert!((res.noise_ratio - (48.0 / 64.0)).abs() < 1e-10);
}

#[test]
fn challenge_residual_extraction_bands_greater_than_spectrum_length() {
    // num_bands = 32, but spectrum has only 8 bins
    let extractor = StochasticResidualExtractor::new(32).unwrap();
    let orig = vec![2.0; 8];
    let det = vec![1.0; 8];
    let res = extractor.extract(&orig, &det);
    assert_eq!(res.band_energies.len(), 32);
    assert!(res.noise_ratio > 0.0);
    assert!(res.band_energies.iter().all(|&e| e >= 0.0));
}

#[test]
fn challenge_residual_zero_bands_rejection() {
    assert!(matches!(
        StochasticResidualExtractor::new(0),
        Err(DspError::InvalidParameter { name: "num_bands", .. })
    ));
}

// ============================================================================
// 6. STFT Sliding Analysis & Overlap-Add Reconstruction Fidelity
// ============================================================================

#[test]
fn challenge_stft_short_audio_padding() {
    // Audio length (100) strictly smaller than window size (512)
    let short_sig = vec![0.5; 100];
    let stft = Stft::new(512, 128, WindowType::Hann).unwrap();
    let traj = stft.process(&short_sig);
    assert!(traj.is_ok());
    let frames = traj.unwrap().frames;
    assert_eq!(frames.len(), 1, "Short audio should produce exactly 1 zero-padded frame");
}

#[test]
fn challenge_stft_non_power_of_two_remainder_audio() {
    // 1337 samples with window 256, hop 128
    let sig: Vec<f64> = (0..1337).map(|i| (i as f64 * 0.1).sin()).collect();
    let stft = Stft::new(256, 128, WindowType::Hann).unwrap();
    let traj = stft.process(&sig).unwrap();
    // Verify frames span the full duration without missing samples
    let expected_frames = 1337_usize.div_ceil(128);
    assert_eq!(traj.frames.len(), expected_frames);
}

#[test]
fn challenge_stft_overlap_add_reconstruction_fidelity() {
    // Test WOLA reconstruction with Hann window at 50% (hop = N/2) and 75% (hop = N/4) overlap
    let n_samples = 4096;
    let fs = 44100.0;

    // Multi-frequency test signal
    let signal: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / fs;
            0.5 * (2.0 * PI * 220.0 * t).sin()
                + 0.3 * (2.0 * PI * 880.0 * t).sin()
                + 0.2 * (2.0 * PI * 2400.0 * t).cos()
        })
        .collect();

    for hop in &[128, 256] {
        let win_size = 512;
        let stft = Stft::new(win_size, *hop, WindowType::Hann).unwrap();
        let spec = stft.forward(&signal).unwrap();
        let recon = stft.inverse(&spec, Some(n_samples)).unwrap();

        assert_eq!(recon.len(), n_samples);

        // Discard ramp-up and ramp-down boundary transients (win_size at both ends)
        let interior_start = win_size;
        let interior_end = n_samples - win_size;

        let mut max_abs_err = 0.0;
        let mut mse = 0.0;
        let mut count = 0;

        for i in interior_start..interior_end {
            let err = (signal[i] - recon[i]).abs();
            if err > max_abs_err {
                max_abs_err = err;
            }
            mse += err * err;
            count += 1;
        }
        mse /= count as f64;

        assert!(
            max_abs_err < 1e-6,
            "WOLA reconstruction max error too large for hop {}: {:.2e}",
            hop, max_abs_err
        );
        assert!(
            mse < 1e-12,
            "WOLA reconstruction MSE too large for hop {}: {:.2e}",
            hop, mse
        );
    }
}

// ============================================================================
// 8. Transform Equivalence & Parseval Invariant: DFT vs FFT
// ============================================================================

#[test]
fn challenge_dft_fft_exact_numerical_equivalence() {
    for &size in &[4, 8, 16, 64, 256, 512] {
        let x: Vec<f64> = (0..size)
            .map(|n| (n as f64 * 0.3).sin() + (n as f64 * 0.7).cos())
            .collect();

        // Direct DFT ground truth
        let dft_spec = Dft::forward_real(&x).unwrap();

        // Fast Fourier Transform
        let mut fft_buf: Vec<Complex<f64>> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
        Fft::forward_complex_radix2(&mut fft_buf).unwrap();

        assert_eq!(dft_spec.len(), fft_buf.len());
        for k in 0..size {
            let diff_re = (dft_spec[k].re - fft_buf[k].re).abs();
            let diff_im = (dft_spec[k].im - fft_buf[k].im).abs();
            assert!(
                diff_re < 1e-10 && diff_im < 1e-10,
                "DFT vs FFT divergence at size {}, bin {}: DFT=({:.6}, {:.6}), FFT=({:.6}, {:.6})",
                size, k, dft_spec[k].re, dft_spec[k].im, fft_buf[k].re, fft_buf[k].im
            );
        }
    }
}

#[test]
fn challenge_parseval_energy_conservation_on_arbitrary_signals() {
    let test_sizes = [1, 3, 7, 13, 64, 128, 500, 1024];

    for &n in &test_sizes {
        let x: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.123).sin() * (-(i as f64) / 100.0).exp())
            .collect();

        let time_energy: f64 = x.iter().map(|&v| v * v).sum();

        let dft_spec = Dft::forward_real(&x).unwrap();
        let freq_energy: f64 = dft_spec.iter().map(|c| c.norm_sq()).sum::<f64>() / (n as f64);

        let rel_diff = (time_energy - freq_energy).abs() / time_energy.max(1e-12);
        assert!(
            rel_diff < 1e-10,
            "Parseval identity violated for size {}: time_e={:.10}, freq_e={:.10}, rel_diff={:.2e}",
            n, time_energy, freq_energy, rel_diff
        );
    }
}

#[test]
fn challenge_fft_size_65536_no_stack_overflow() {
    let size = 65536;
    let mut buf = vec![Complex::zero(); size];
    buf[0] = Complex::new(1.0, 0.0); // Kronecker delta impulse

    Fft::forward_complex_radix2(&mut buf).unwrap();

    // Frequency response of impulse must be flat 1.0 across all bins
    for (k, c) in buf.iter().enumerate() {
        assert!(
            (c.norm() - 1.0).abs() < 1e-10,
            "Bin {} magnitude was {:.6}, expected 1.0",
            k, c.norm()
        );
    }
}


#[test]
fn challenge_stft_parameter_validation_boundaries() {
    // hop_size = 0
    assert!(matches!(
        Stft::new(512, 0, WindowType::Hann),
        Err(DspError::InvalidParameter { name: "hop_size", .. })
    ));
    // hop_size > window_size
    assert!(matches!(
        Stft::new(256, 512, WindowType::Hann),
        Err(DspError::InvalidParameter { name: "hop_size", .. })
    ));
    // window_size < 2
    assert!(matches!(
        Stft::new(1, 1, WindowType::Hann),
        Err(DspError::InvalidParameter { name: "window_size", .. })
    ));
    // window_size > fft_size
    assert!(matches!(
        Stft::with_fft_size(512, 256, WindowType::Hann, 256),
        Err(DspError::InvalidParameter { name: "window_size", .. })
    ));
    // fft_size non-power-of-two (with window_size <= fft_size)
    assert!(matches!(
        Stft::with_fft_size(256, 128, WindowType::Hann, 500),
        Err(DspError::NonPowerOfTwo { size: 500 })
    ));
    // empty input
    let stft = Stft::new(256, 128, WindowType::Hann).unwrap();
    assert_eq!(stft.process(&[]), Err(DspError::EmptyInput));
    assert_eq!(stft.forward(&[]), Err(DspError::EmptyInput));
    assert_eq!(stft.inverse(&[], None), Err(DspError::EmptyInput));
}

#[test]
fn challenge_stft_non_finite_sample_rejection() {
    let stft = Stft::new(256, 128, WindowType::Hann).unwrap();
    let nan_sig = vec![0.0, 1.0, f64::NAN, 0.5];
    assert!(matches!(
        stft.forward(&nan_sig),
        Err(DspError::NumericalInstability(_))
    ));

    let inf_sig = vec![0.0, 1.0, f64::INFINITY, 0.5];
    assert!(matches!(
        stft.forward(&inf_sig),
        Err(DspError::NumericalInstability(_))
    ));
}

#[test]
fn challenge_partial_tracker_parameter_validation() {
    // f0 negative
    assert!(HarmonicPartialTracker::validate_params(-10.0, 5, 0.0).is_err());
    // f0 NaN
    assert!(HarmonicPartialTracker::validate_params(f64::NAN, 5, 0.0).is_err());
    // max_partials 0
    assert!(HarmonicPartialTracker::validate_params(440.0, 0, 0.0).is_err());
    // inharmonicity negative
    assert!(HarmonicPartialTracker::validate_params(440.0, 5, -0.01).is_err());
    // inharmonicity Inf
    assert!(HarmonicPartialTracker::validate_params(440.0, 5, f64::INFINITY).is_err());
}
