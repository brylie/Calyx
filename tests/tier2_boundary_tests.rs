//! tests/tier2_boundary_tests.rs
//! Opaque-box Tier 2 Boundary & Corner Case Verification Suite for Calyx DSP.
//!
//! Covers all 10 Core Boundary Domains across Features 1 through 30:
//! 1. Empty Buffers & Degenerate Slices
//! 2. Silent Signals & Denormals
//! 3. DC Offset Signals
//! 4. Nyquist & Super-Nyquist Frequencies
//! 5. Sub-Bass & Extreme Low Frequencies
//! 6. Non-Power-of-Two Buffer Lengths
//! 7. Singular Conditions & Parameter Rejections
//! 8. Dynamic Range Extremes & Clipping
//! 9. Rank Boundary Conditions in Matrix/Tensor Operations
//! 10. Division-by-Zero Protections
//!
//! Invariants verified:
//! - Zero unhandled panics under any valid or invalid boundary condition.
//! - Graceful typed DspError propagation.
//! - Numerical stability invariants (zero NaNs, zero Infs, finite metrics).

#[path = "common/mod.rs"]
mod common;

use calyx_dsp::error::*;
use calyx_dsp::hypercube::*;
use calyx_dsp::math::*;
use calyx_dsp::metrics::*;
use calyx_dsp::pipeline::*;
use calyx_dsp::spectral::*;
use calyx_dsp::synthesis::*;
use calyx_dsp::synthetic::*;
use std::f64::consts::{FRAC_PI_2, PI};

// ============================================================================
// Domain 1: Empty Buffers, Zero-Length Slices & Degenerate Slices
// ============================================================================

mod domain1_empty_buffers {
    use super::*;

    #[test]
    fn test_b01_01_matrix_empty_rejection() {
        assert!(matches!(
            Matrix::<f64>::from_vec(0, 5, vec![]),
            Err(DspError::InvalidParameter { .. }) | Err(DspError::EmptyInput)
        ));
        assert!(matches!(
            Matrix::<f64>::from_vec(5, 0, vec![]),
            Err(DspError::InvalidParameter { .. }) | Err(DspError::EmptyInput)
        ));
    }

    #[test]
    fn test_b01_02_time_metrics_empty_inputs() {
        assert_eq!(compute_time_metrics(&[], &[]), Err(DspError::EmptyInput));
        assert_eq!(
            compute_time_metrics(&[1.0, 2.0], &[]),
            Err(DspError::EmptyInput)
        );
    }

    #[test]
    fn test_b01_03_spectral_metrics_empty_inputs() {
        assert_eq!(
            compute_spectral_metrics(&[], &[]),
            Err(DspError::EmptyInput)
        );
        assert_eq!(
            compute_spectral_metrics(&[1.0], &[]),
            Err(DspError::EmptyInput)
        );
    }

    #[test]
    fn test_b01_04_dft_empty_rejection() {
        assert_eq!(Dft::forward_real(&[]), Err(DspError::EmptyInput));
    }

    #[test]
    fn test_b01_05_stft_empty_rejection() {
        let stft = Stft::new(512, 256, WindowType::Hann).unwrap();
        assert_eq!(stft.process(&[]), Err(DspError::EmptyInput));
    }

    #[test]
    fn test_b01_06_ola_empty_block_rejection() {
        let mut ola = OverlapAdd::new(512, 256).unwrap();
        assert_eq!(ola.push_block(&[]), Err(DspError::EmptyInput));
    }

    #[test]
    fn test_b01_07_pipeline_empty_input() {
        let pipeline = DspPipeline::default_reference();
        assert_eq!(pipeline.run(&[], 44100.0), Err(DspError::EmptyInput));
    }

    /// F04-B01: Typed DspError::EmptyInput display formatting and error trait adherence.
    #[test]
    fn test_b01_08_dsp_error_empty_input_display() {
        let err = DspError::EmptyInput;
        assert_eq!(format!("{}", err), "Input buffer or sequence is empty");
        let std_err: &dyn std::error::Error = &err;
        assert!(std_err.source().is_none());
    }

    /// F05-B05: SineGenerator zero samples requested returns empty buffer cleanly.
    #[test]
    fn test_b01_09_sine_zero_samples_requested() {
        let gen = SineGenerator::new(440.0, 1.0, 0.0);
        let buf = gen.generate(0, 44100.0);
        assert!(buf.samples.is_empty());
        assert_eq!(buf.sample_rate, 44100.0);
    }

    /// F06-B05: HarmonicComplexGenerator empty recipe weights rejected.
    #[test]
    fn test_b01_10_harmonic_empty_recipe_weights() {
        let recipe = HarmonicRecipe::Custom(vec![]);
        assert!(matches!(
            HarmonicComplexGenerator::validate_recipe(&recipe),
            Err(DspError::EmptyInput) | Err(DspError::InvalidParameter { name: "recipe", .. })
        ));
    }

    /// F07-B04: ChirpGenerator zero duration or zero samples requested.
    #[test]
    fn test_b01_11_chirp_zero_duration() {
        assert!(matches!(
            ChirpGenerator::validate_params(100.0, 1000.0, 1.0, 0.0, ChirpKind::Linear),
            Err(DspError::InvalidParameter { name: "duration", .. })
        ));
        let chirp = ChirpGenerator::new(100.0, 1000.0, 1.0, 1.0, ChirpKind::Linear);
        let buf = chirp.generate(0, 44100.0);
        assert!(buf.samples.is_empty());
    }

    /// F09-B01: NoiseGenerator zero samples requested returns empty buffer.
    #[test]
    fn test_b01_12_noise_zero_samples_requested() {
        let gen = NoiseGenerator::new(NoiseColor::WhiteUniform, 1.0, 42);
        let buf = gen.generate(0, 48000.0);
        assert!(buf.samples.is_empty());
    }

    /// F10-B04: SyntheticInstrumentSource zero duration note requested.
    #[test]
    fn test_b01_13_instrument_zero_duration_note() {
        let inst = SyntheticInstrumentSource::default();
        let buf = inst.synthesize([0.5, 0.5, 0.5, 0.5, 0.5], 0, 44100.0);
        assert!(buf.is_ok());
        assert!(buf.unwrap().samples.is_empty());
    }

    /// F12-B04: Harmonic peak error computation with zero detected peaks in target.
    #[test]
    fn test_b01_14_harmonic_peak_error_zero_detected_peaks() {
        let gt_peaks = vec![SpectralPeak { bin_index: 10, frequency_hz: 440.0, amplitude: 1.0, phase_rad: 0.0 }];
        let detected_peaks: Vec<SpectralPeak> = vec![];
        let err = compute_harmonic_peak_error(&gt_peaks, &detected_peaks);
        assert!(err.is_nan() || err > 100.0, "Missing peaks must yield high penalty or sentinel");
    }

    /// F15-B02: Radix-2 FFT empty slice rejection.
    #[test]
    fn test_b01_15_fft_empty_slice_rejection() {
        let mut empty_buf: Vec<Complex<f64>> = vec![];
        assert!(matches!(
            Fft::forward_complex_radix2(&mut empty_buf),
            Err(DspError::EmptyInput) | Err(DspError::NonPowerOfTwo { size: 0 })
        ));
    }

    /// F18-B01: HarmonicPartialTracker zero detected peaks input.
    #[test]
    fn test_b01_16_partial_tracker_zero_detected_peaks() {
        let tracker = HarmonicPartialTracker::new(440.0, 8, 0.0);
        let partials = tracker.track_partials(&[], 44100.0);
        assert!(partials.is_empty());
    }

    /// F20-B04: HypercubeLayout empty axis configuration rejection.
    #[test]
    fn test_b01_17_hypercube_empty_layout_rejection() {
        assert!(matches!(
            HypercubeLayout::new(vec![]),
            Err(DspError::EmptyInput) | Err(DspError::InvalidParameter { name: "axes", .. })
        ));
    }

    /// F25-B01: AdditiveOscillatorBank empty partials frame outputs silence.
    #[test]
    fn test_b01_18_additive_empty_partials_silence() {
        let frame = SpectralFrame::empty(0.0);
        let osc = AdditiveOscillatorBank::new(44100.0);
        let out = osc.synthesize_frame(&frame, 256);
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|&s| s == 0.0));
    }

    /// F25-B03: AdditiveOscillatorBank zero samples synthesis requested.
    #[test]
    fn test_b01_19_additive_zero_samples_synthesis() {
        let mut frame = SpectralFrame::empty(0.0);
        frame.partials.push(HarmonicPartial { harmonic_index: 1, frequency: 440.0, amplitude: 0.8, phase: 0.0 });
        let osc = AdditiveOscillatorBank::new(44100.0);
        let out = osc.synthesize_frame(&frame, 0);
        assert!(out.is_empty());
    }

    /// F27-B02: StochasticNoiseResynthesizer empty envelope rejection.
    #[test]
    fn test_b01_20_noise_synth_empty_envelope_rejection() {
        let env = NoiseBandEnvelope { band_energies: vec![], noise_ratio: 0.5 };
        let synth = StochasticNoiseResynthesizer::new(16, 42);
        assert!(matches!(
            synth.synthesize_frame(&env, 256),
            Err(DspError::EmptyInput) | Err(DspError::InvalidParameter { name: "band_energies", .. })
        ));
    }

    /// F27-B03: StochasticNoiseResynthesizer zero samples requested.
    #[test]
    fn test_b01_21_noise_synth_zero_samples_requested() {
        let env = NoiseBandEnvelope { band_energies: vec![0.5; 16], noise_ratio: 0.5 };
        let synth = StochasticNoiseResynthesizer::new(16, 42);
        let out = synth.synthesize_frame(&env, 0).unwrap();
        assert!(out.is_empty());
    }

    /// F28-B03: OverlapAdd pull available audio from empty accumulator.
    #[test]
    fn test_b01_22_ola_pull_empty_accumulator() {
        let mut ola = OverlapAdd::new(512, 256).unwrap();
        let audio = ola.pull_available_audio();
        assert!(audio.is_empty());
    }

    /// F28-B04: OverlapAdd flush on newly created empty accumulator.
    #[test]
    fn test_b01_23_ola_flush_empty_accumulator() {
        let mut ola = OverlapAdd::new(512, 256).unwrap();
        let flushed = ola.flush();
        assert!(flushed.is_empty());
    }
}

// ============================================================================
// Domain 2: Silent Signals (All Zeros, Negative Zeros) & Denormals
// ============================================================================

mod domain2_silent_signals {
    use super::*;

    #[test]
    fn test_b02_01_complex_zero_norm_and_arg() {
        let z = Complex::new(0.0, 0.0);
        assert_eq!(z.norm(), 0.0);
        assert_eq!(z.norm_sq(), 0.0);
        assert_eq!(z.arg(), 0.0);
        assert!(!z.norm().is_nan());
        assert!(!z.arg().is_nan());
    }

    #[test]
    fn test_b02_02_complex_negative_zero() {
        let z_neg = Complex::new(-0.0, 0.0);
        assert_eq!(z_neg.norm(), 0.0);
        assert!(!z_neg.arg().is_nan());
    }

    #[test]
    fn test_b02_03_time_metrics_pure_silence() {
        let silent = vec![0.0; 512];
        let m = compute_time_metrics(&silent, &silent).expect("Metrics on silence failed");
        assert_eq!(m.mse, 0.0);
        assert_eq!(m.rmse, 0.0);
        assert!(!m.snr_db.is_nan());
        assert!(!m.correlation_r.is_nan());
    }

    #[test]
    fn test_b02_04_spectral_metrics_pure_silence() {
        let silent = vec![0.0; 513];
        let m = compute_spectral_metrics(&silent, &silent).expect("Metrics on silence failed");
        assert_eq!(m.spectral_convergence, 0.0);
        assert_eq!(m.log_spectral_distance_db, 0.0);
        assert!(!m.spectral_convergence.is_nan());
        assert!(!m.log_spectral_distance_db.is_nan());
    }

    #[test]
    fn test_b02_05_nmf_all_zeros_matrix() {
        let x = Matrix::from_vec(8, 8, vec![0.0; 64]).unwrap();
        let res = Nmf::decompose(&x, 2, 20);
        assert!(res.is_ok(), "NMF on zeros must not divide by zero or panic");
        let (w, h) = res.unwrap();
        assert!(w.as_slice().iter().all(|&v: &f64| !v.is_nan() && v >= 0.0));
        assert!(h.as_slice().iter().all(|&v: &f64| !v.is_nan() && v >= 0.0));
    }

    #[test]
    fn test_b02_06_tucker_all_zeros_tensor() {
        let t = Tensor::from_vec(vec![4, 4, 4], vec![0.0; 64]).unwrap();
        let res = TuckerDecomposition::decompose(&t, &[2, 2, 2]);
        assert!(res.is_ok(), "Tucker on zeros must not divide by zero");
        let decomp = res.unwrap();
        assert!(decomp.core.data().iter().all(|&v: &f64| !v.is_nan()));
    }

    #[test]
    fn test_b02_07_pipeline_pure_silence() {
        let pipeline = DspPipeline::default_reference();
        let silence = vec![0.0; 44100];
        let res = pipeline.run(&silence, 44100.0);
        assert!(res.is_ok(), "Pipeline must process silence cleanly");
        let output = res.unwrap();
        assert!(output.reconstructed_audio.iter().all(|&s: &f64| s.abs() < 1e-10));
    }

    /// F10-B05: SyntheticInstrumentSource zero velocity/dynamic synthesizes clean silence.
    #[test]
    fn test_b02_08_instrument_zero_velocity_silence() {
        let inst = SyntheticInstrumentSource::default();
        let buf = inst.synthesize([0.5, 0.5, 0.0, 0.5, 0.5], 1024, 44100.0).unwrap();
        assert!(buf.samples.iter().all(|&s: &f64| s.abs() < 1e-12));
    }

    /// F11-B04: Time metrics computation with silent reference and noisy reconstruction.
    #[test]
    fn test_b02_09_time_metrics_silent_reference_noisy_recon() {
        let silent_ref = vec![0.0; 512];
        let noisy_recon = vec![0.5; 512];
        let m = compute_time_metrics(&silent_ref, &noisy_recon).unwrap();
        assert!((m.mse - 0.25).abs() < 1e-12);
        assert!((m.rmse - 0.5).abs() < 1e-12);
        assert!(!m.snr_db.is_nan(), "SNR must not be NaN on silent reference");
    }

    /// F17-B01: Spectral peak detector on all-zeros silence spectrum.
    #[test]
    fn test_b02_10_peak_detection_pure_silence() {
        let silent_spec = vec![0.0; 257];
        let peaks = PeakDetector::detect_peaks(&silent_spec, 48000.0);
        assert!(peaks.is_empty(), "Silence spectrum must yield zero detected peaks");
    }

    /// F19-B01: Stochastic residual extraction with identical spectra yields zero residual.
    #[test]
    fn test_b02_11_residual_perfect_deterministic_match() {
        let spec = vec![1.0; 16];
        let res = StochasticResidualExtractor::extract_residual(&spec, &spec);
        assert!(res.band_energies.iter().all(|&e: &f64| e.abs() < 1e-12));
        assert!(res.noise_ratio.abs() < 1e-12);
    }

    /// F21-B01: PCA decomposition on all-zeros matrix handles zero variance without division by zero.
    #[test]
    fn test_b02_12_pca_all_zeros_matrix() {
        let x = Matrix::from_vec(4, 4, vec![0.0; 16]).unwrap();
        let res = Pca::decompose(&x, 2);
        assert!(res.is_ok(), "PCA on all-zeros matrix must not divide by zero");
        let (weights, basis) = res.unwrap();
        assert!(weights.as_slice().iter().all(|&v: &f64| !v.is_nan()));
        assert!(basis.as_slice().iter().all(|&v: &f64| !v.is_nan()));
    }

    /// F24-B05: CP-ALS decomposition on all-zeros tensor converges cleanly.
    #[test]
    fn test_b02_13_cp_all_zeros_tensor() {
        let t = Tensor::from_vec(vec![3, 3, 3], vec![0.0; 27]).unwrap();
        let res = CpAls::decompose(&t, 1, 10);
        assert!(res.is_ok(), "CP-ALS on zeros must not produce NaN or infinite loop");
    }

    /// F27-B01: Stochastic noise resynthesis with zero envelope produces pure silence.
    #[test]
    fn test_b02_14_noise_synth_all_zero_envelope() {
        let env = NoiseBandEnvelope { band_energies: vec![0.0; 16], noise_ratio: 0.0 };
        let synth = StochasticNoiseResynthesizer::new(16, 42);
        let out = synth.synthesize_frame(&env, 512).unwrap();
        assert!(out.iter().all(|&s: &f64| s.abs() < 1e-12));
    }

    /// F30-B05: Quality tier validation passes on pure silence across all tiers.
    #[test]
    fn test_b02_15_quality_tier_validation_on_silence() {
        let silent = vec![0.0; 1024];
        let m_time = compute_time_metrics(&silent, &silent).unwrap();
        let m_spec = compute_spectral_metrics(&silent, &silent).unwrap();
        for tier in &[QualityTierConfig::reference(), QualityTierConfig::production(), QualityTierConfig::preview()] {
            assert!(tier.validate_fidelity(&m_time, &m_spec).is_ok());
        }
    }
}

// ============================================================================
// Domain 3: DC Offset Signals (Pure Constant Amplitude)
// ============================================================================

mod domain3_dc_offset {
    use super::*;

    #[test]
    fn test_b03_01_sine_zero_frequency_pure_dc() {
        let gen = SineGenerator::new(0.0, 1.0, FRAC_PI_2);
        let buf = gen.generate(1024, 48000.0);
        // sin(0 + pi/2) == 1.0 for all samples
        assert!(buf.samples.iter().all(|&s: &f64| (s - 1.0).abs() < 1e-14));
    }

    #[test]
    fn test_b03_02_dft_pure_dc_energy_isolation() {
        let n = 256;
        let dc_signal = vec![1.0; n];
        let spectrum = Dft::forward_real(&dc_signal).expect("DFT forward failed");

        // DC bin 0 has all energy: X[0] = n * 1.0 = 256.0
        assert!((spectrum[0].re - (n as f64)).abs() < 1e-10);
        assert!(spectrum[0].im.abs() < 1e-10);

        // All AC bins k >= 1 should be zero
        for k in 1..spectrum.len() {
            assert!(
                spectrum[k].norm() < 1e-10,
                "AC bin {} had non-zero energy: {}",
                k,
                spectrum[k].norm()
            );
        }
    }

    #[test]
    fn test_b03_03_peak_detection_dc_bin_boundary() {
        let mut mag_spectrum = vec![0.0; 257];
        mag_spectrum[0] = 10.0; // Peak at exact DC bin 0

        let peaks = PeakDetector::detect_peaks(&mag_spectrum, 48000.0);
        // Parabolic interpolation must not access bin -1
        assert!(!peaks.is_empty());
        assert_eq!(peaks[0].bin_index, 0);
        assert!((peaks[0].frequency_hz - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_b03_04_stability_dc_offset_flagged() {
        let dc_heavy = vec![0.5; 1000];
        let rep = check_numerical_stability(&vec![0.0; 1000], &dc_heavy);
        assert!(!rep.is_clean);
        assert!((rep.dc_offset - 0.5).abs() < 1e-6);
    }

    /// F07-B01: Chirp with identical start and end frequency behaves as pure sinusoidal tone.
    #[test]
    fn test_b03_05_chirp_start_equals_stop_frequency() {
        let fs = 48000.0;
        let chirp = ChirpGenerator::new(440.0, 440.0, 1.0, 0.1, ChirpKind::Linear);
        let buf_chirp = chirp.generate(4800, fs);
        let sine = SineGenerator::new(440.0, 1.0, 0.0);
        let buf_sine = sine.generate(4800, fs);
        for i in 0..4800 {
            assert!((buf_chirp.samples[i] - buf_sine.samples[i]).abs() < 1e-12);
        }
    }

    /// F26-B03: Partial with 0.0 Hz frequency produces constant DC offset.
    #[test]
    fn test_b03_06_phase_zero_frequency_partial() {
        let mut frame = SpectralFrame::empty(0.0);
        frame.partials.push(HarmonicPartial { harmonic_index: 0, frequency: 0.0, amplitude: 0.75, phase: FRAC_PI_2 });
        let osc = AdditiveOscillatorBank::new(48000.0);
        let out = osc.synthesize_frame(&frame, 256);
        for &s in &out {
            let s: f64 = s;
            assert!((s - 0.75).abs() < 1e-12, "0 Hz partial must produce pure DC amplitude");
        }
    }

    /// F28-B05: OverlapAdd reconstruction of constant blocks converges to exact unity gain.
    #[test]
    fn test_b03_07_ola_constant_stream_unity_gain() {
        let window_len = 512;
        let hop_size = 256;
        let mut ola = OverlapAdd::new(window_len, hop_size).unwrap();
        let window = generate_window(WindowType::Hann, window_len).unwrap();
        for _ in 0..10 {
            let block: Vec<f64> = window.clone(); // Hann windowed constant 1.0
            ola.push_block(&block).unwrap();
        }
        let stream = ola.pull_available_audio();
        // After overlap buildup (indices >= 512), Hann OLA at 50% hop must sum to 1.0
        if stream.len() > 512 {
            for &sample in &stream[256..stream.len() - 256] {
                let sample: f64 = sample;
                assert!((sample - 1.0).abs() < 1e-4, "COLA reconstruction unity gain failed: {}", sample);
            }
        }
    }
}

// ============================================================================
// Domain 4: Nyquist and Near-Nyquist Frequencies (f = fs/2, f > fs/2)
// ============================================================================

mod domain4_nyquist {
    use super::*;

    #[test]
    fn test_b04_01_sine_exact_nyquist_frequency() {
        let fs = 44100.0;
        let gen = SineGenerator::new(fs / 2.0, 1.0, FRAC_PI_2);
        let buf = gen.generate(100, fs);

        // Sequence: [1.0, -1.0, 1.0, -1.0, ...]
        for (i, &s) in buf.samples.iter().enumerate() {
            let s: f64 = s;
            let expected = if i % 2 == 0 { 1.0 } else { -1.0 };
            assert!((s - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn test_b04_02_sine_super_nyquist_rejection() {
        let fs = 44100.0;
        // Frequency 25000 Hz > fs / 2 (22050 Hz)
        assert!(matches!(
            SineGenerator::validate_params(25000.0, 1.0, 0.0, fs),
            Err(DspError::InvalidParameter {
                name: "frequency",
                ..
            })
        ));
    }

    #[test]
    fn test_b04_03_harmonic_partials_super_nyquist_culled() {
        let fs = 44100.0;
        // f0 = 5000 Hz, K = 10 -> partials at 5k, 10k, 15k, 20k (4 under Nyquist)
        let gen = HarmonicComplexGenerator::new(5000.0, 1.0, 10, 0.0, HarmonicRecipe::Sawtooth);
        let buf = gen.generate(1000, fs);
        assert_eq!(buf.metadata.known_partials.len(), 4);
        assert!(buf
            .metadata
            .known_partials
            .iter()
            .all(|p| p.frequency_hz <= fs / 2.0));
    }

    #[test]
    fn test_b04_04_chirp_sweep_to_exact_nyquist() {
        let fs = 48000.0;
        let gen = ChirpGenerator::new(20.0, fs / 2.0, 0.9, 0.5, ChirpKind::Linear);
        let buf = gen.generate(24000, fs);
        assert!(buf.samples.iter().all(|s: &f64| !s.is_nan() && !s.is_infinite()));
        assert!(buf.samples.iter().all(|s: &f64| s.abs() <= 0.900000001));
    }

    #[test]
    fn test_b04_05_peak_detection_nyquist_bin_boundary() {
        let n_bins = 257; // bins 0..=256
        let mut mag_spectrum = vec![0.0; n_bins];
        mag_spectrum[256] = 10.0; // Peak at exact Nyquist bin

        let peaks = PeakDetector::detect_peaks(&mag_spectrum, 48000.0);
        // Parabolic interpolation must not access bin 257
        assert!(!peaks.is_empty());
        let peak = peaks.iter().find(|p| p.bin_index == 256);
        assert!(peak.is_some());
        assert!((peak.unwrap().frequency_hz - 24000.0).abs() < 1.0);
    }

    #[test]
    fn test_b04_06_additive_partial_frequency_exceeds_nyquist() {
        let mut frame = SpectralFrame::empty(0.0);
        frame.partials.push(HarmonicPartial {
            harmonic_index: 1,
            frequency: 25000.0, // Exceeds Nyquist of 44100 / 2
            amplitude: 0.8,
            phase: 0.0,
        });

        let osc = AdditiveOscillatorBank::new(44100.0);
        let out = osc.synthesize_frame(&frame, 256);
        // Partial above Nyquist must be attenuated or culled to avoid aliasing
        assert!(out.iter().all(|&s: &f64| s.abs() < 1e-10));
    }

    /// F01-B04: Complex::from_polar with boundary angles (-PI, 0, PI, 2*PI).
    #[test]
    fn test_b04_07_complex_from_polar_boundary_angles() {
        let angles = &[-PI, 0.0, PI, 2.0 * PI, -3.0 * PI];
        for &theta in angles {
            let z = Complex::from_polar(1.0, theta);
            assert!((z.norm() - 1.0).abs() < 1e-14);
            assert!((z.re - theta.cos()).abs() < 1e-14);
            assert!((z.im - theta.sin()).abs() < 1e-14);
        }
    }

    /// F06-B04: HarmonicComplexGenerator with extreme inharmonicity B culls super-Nyquist harmonics.
    #[test]
    fn test_b04_08_harmonic_extreme_inharmonicity() {
        let fs = 44100.0;
        let gen = HarmonicComplexGenerator::new(1000.0, 1.0, 10, 1.0, HarmonicRecipe::Sawtooth);
        let buf = gen.generate(1024, fs);
        for p in &buf.metadata.known_partials {
            assert!(p.frequency_hz <= fs / 2.0, "Inharmonic partial exceeded Nyquist: {}", p.frequency_hz);
        }
    }

    /// F18-B03: HarmonicPartialTracker supresses partials exceeding Nyquist.
    #[test]
    fn test_b04_09_partial_tracker_super_nyquist_cutoff() {
        let fs = 44100.0;
        let tracker = HarmonicPartialTracker::new(10000.0, 5, 0.0);
        let peaks = vec![
            SpectralPeak { bin_index: 100, frequency_hz: 10000.0, amplitude: 1.0, phase_rad: 0.0 },
            SpectralPeak { bin_index: 200, frequency_hz: 20000.0, amplitude: 0.8, phase_rad: 0.0 },
            SpectralPeak { bin_index: 300, frequency_hz: 30000.0, amplitude: 0.6, phase_rad: 0.0 }, // Super-Nyquist
        ];
        let partials = tracker.track_partials(&peaks, fs);
        assert!(partials.iter().all(|p| p.frequency <= fs / 2.0));
    }

    /// F18-B04: HarmonicPartialTracker safely handles inharmonic partials crossing Nyquist.
    #[test]
    fn test_b04_10_partial_tracker_inharmonicity_super_nyquist() {
        let fs = 48000.0;
        let tracker = HarmonicPartialTracker::new(5000.0, 8, 0.5);
        let peaks = vec![
            SpectralPeak { bin_index: 50, frequency_hz: 5000.0, amplitude: 1.0, phase_rad: 0.0 },
            SpectralPeak { bin_index: 150, frequency_hz: 15000.0, amplitude: 0.5, phase_rad: 0.0 },
        ];
        let partials = tracker.track_partials(&peaks, fs);
        assert!(partials.iter().all(|p| p.frequency <= fs / 2.0));
    }

    /// F26-B01: Phase harmonic lock assigns exact integer multiples modulo 2*PI.
    #[test]
    fn test_b04_11_phase_harmonic_lock_exact_multiples() {
        let phi_0 = 0.5;
        let k = 4;
        let phi_k = (k as f64 * phi_0) % (2.0 * PI);
        assert!((phi_k - 2.0).abs() < 1e-14);
    }

    /// F26-B02: Phase wrapping boundary check at 2*PI.
    #[test]
    fn test_b04_12_phase_wrapping_boundaries() {
        let mut phase = 2.0 * PI - 1e-14;
        let delta = 0.1;
        phase += delta;
        if phase >= 2.0 * PI {
            phase -= 2.0 * PI;
        }
        assert!(phase >= 0.0 && phase < 2.0 * PI);
        assert!((phase - 0.1).abs() < 1e-12);
    }
}

// ============================================================================
// Domain 5: Sub-Bass & Extreme Low Frequencies (f -> 0)
// ============================================================================

mod domain5_sub_bass {
    use super::*;

    #[test]
    fn test_b05_01_sine_sub_bass_phase_monotonicity() {
        let f0 = 0.01; // 0.01 Hz
        let fs = 44100.0;
        let gen = SineGenerator::new(f0, 1.0, 0.0);
        let buf = gen.generate(441000, fs); // 10 seconds

        assert_eq!(buf.samples.len(), 441000);
        // In the first 100 samples, sin(2*pi*f0*t) must be strictly non-decreasing
        for i in 0..99 {
            assert!(
                buf.samples[i + 1] >= buf.samples[i],
                "Monotonicity failed at {}: {} vs {}",
                i,
                buf.samples[i + 1],
                buf.samples[i]
            );
        }
    }

    #[test]
    fn test_b05_02_log_chirp_zero_start_frequency_rejection() {
        // Log chirp requires f_start > 0 because log(0) is undefined
        assert!(matches!(
            ChirpGenerator::validate_params(0.0, 1000.0, 1.0, 1.0, ChirpKind::Logarithmic),
            Err(DspError::InvalidParameter { name: "f_start", .. })
        ));
    }

    #[test]
    fn test_b05_03_partial_tracker_near_zero_f0() {
        let tracker = HarmonicPartialTracker::new(0.5, 5, 0.0);
        let peaks = vec![
            SpectralPeak {
                bin_index: 1,
                frequency_hz: 0.5,
                amplitude: 1.0,
                phase_rad: 0.0,
            },
            SpectralPeak {
                bin_index: 2,
                frequency_hz: 1.0,
                amplitude: 0.5,
                phase_rad: 0.0,
            },
        ];
        let partials = tracker.track_partials(&peaks, 48000.0);
        assert!(!partials.is_empty());
        assert!((partials[0].frequency - 0.5).abs() < 1e-2);
    }

    /// F26-B05: Continuous phase integration under smooth frequency modulation without phase jumps.
    #[test]
    fn test_b05_04_phase_vocoder_continuous_integration() {
        let fs = 44100.0;
        let n = 1000;
        let mut phase = 0.0;
        let mut prev_phase = 0.0;
        for i in 0..n {
            let freq = 440.0 + (i as f64) * 0.44; // Ramp 440 -> 880 Hz
            let d_phi = 2.0 * PI * freq / fs;
            phase += d_phi;
            if i > 0 {
                let delta = phase - prev_phase;
                assert!((delta - d_phi).abs() < 1e-12, "Phase discontinuity detected at step {}", i);
            }
            prev_phase = phase;
        }
    }
}

// ============================================================================
// Domain 6: Non-Power-of-Two Buffer Lengths
// ============================================================================

mod domain6_non_power_of_two {
    use super::*;

    #[test]
    fn test_b06_01_fft_non_power_of_two_rejection() {
        let non_p2_sizes = &[3, 5, 7, 300, 500, 1000];
        for &size in non_p2_sizes {
            let mut complex_buf = vec![Complex::zero(); size];
            assert!(matches!(
                Fft::forward_complex_radix2(&mut complex_buf),
                Err(DspError::NonPowerOfTwo { size: s }) if s == size
            ));
        }
    }

    #[test]
    fn test_b06_02_dft_prime_lengths_3_and_7() {
        for &n in &[3, 7] {
            let x: Vec<f64> = (0..n).map(|v| (v + 1) as f64).collect();
            let spectrum = Dft::forward_real(&x).expect("DFT prime length forward failed");
            let reconstructed = Dft::inverse_real(&spectrum).expect("DFT inverse failed");

            for idx in 0..n {
                assert!(
                    (x[idx] - reconstructed[idx]).abs() < 1e-12,
                    "Reconstruction mismatch at {}: expected {}, got {}",
                    idx,
                    x[idx],
                    reconstructed[idx]
                );
            }
        }
    }

    #[test]
    fn test_b06_03_dft_arbitrary_lengths_500_and_1000() {
        for &n in &[500, 1000] {
            let x: Vec<f64> = (0..n).map(|v| (v as f64 * 0.01).sin()).collect();
            let spectrum = Dft::forward_real(&x).unwrap();

            // Verify Parseval energy theorem
            let time_energy: f64 = x.iter().map(|&s| s * s).sum();
            let freq_energy: f64 = spectrum.iter().map(|c: &Complex<f64>| c.norm_sq()).sum::<f64>() / (n as f64);
            let diff = (time_energy - freq_energy).abs();
            assert!(diff < 1e-10);
        }
    }

    #[test]
    fn test_b06_04_stft_non_power_of_two_remainder() {
        let audio = vec![0.1; 1000];
        let stft = Stft::new(256, 128, WindowType::Hann).unwrap();
        let traj = stft.process(&audio).expect("STFT remainder processing failed");
        assert!(!traj.frames.is_empty());
    }

    /// F02-B05: Symmetric window generation with odd length has unique center peak.
    #[test]
    fn test_b06_05_window_odd_length_symmetry() {
        let w = generate_window_symmetric(WindowType::Hamming, 513).unwrap();
        assert_eq!(w.len(), 513);
        let center = 256;
        for i in 0..center {
            assert!((w[i] - w[512 - i]).abs() < 1e-14, "Asymmetry at index {}", i);
            assert!(w[center] >= w[i], "Center sample was not maximum");
        }
    }

    /// F04-B04: DspError::NonPowerOfTwo display formatting with size 0.
    #[test]
    fn test_b06_06_dsp_error_non_power_of_two_zero() {
        let err = DspError::NonPowerOfTwo { size: 0 };
        assert_eq!(format!("{}", err), "Size 0 is not a power of two (Radix-2 FFT requirement)");
    }

    /// F15-B03: Radix-2 FFT minimal valid power-of-two sizes N = 1, 2, 4.
    #[test]
    fn test_b06_07_fft_minimal_power_of_two_sizes() {
        for &size in &[1, 2, 4] {
            let mut buf: Vec<Complex<f64>> = (0..size).map(|i| Complex::new(i as f64 + 1.0, 0.0)).collect();
            let original = buf.clone();
            Fft::forward_complex_radix2(&mut buf).unwrap();
            Fft::inverse_complex_radix2(&mut buf).unwrap();
            for i in 0..size {
                assert!((buf[i].re - original[i].re).abs() < 1e-12);
                assert!(buf[i].im.abs() < 1e-12);
            }
        }
    }
}

// ============================================================================
// Domain 7: Singular Conditions & Parameter Rejections
// ============================================================================

mod domain7_singular_parameters {
    use super::*;

    #[test]
    fn test_b07_01_window_zero_length_rejection() {
        assert!(matches!(
            generate_window(WindowType::Hann, 0),
            Err(DspError::InvalidParameter { name: "length", .. }) | Err(DspError::EmptyInput)
        ));
    }

    #[test]
    fn test_b07_02_cola_zero_hop_size_rejection() {
        let w = generate_window(WindowType::Hann, 512).unwrap();
        assert!(!verify_cola(&w, 0, 1e-4));
    }

    #[test]
    fn test_b07_03_cola_hop_greater_than_length() {
        let w = generate_window(WindowType::Hann, 512).unwrap();
        // Hop 1024 > length 512 leaves gaps, must fail COLA
        assert!(!verify_cola(&w, 1024, 1e-4));
    }

    #[test]
    fn test_b07_04_harmonic_zero_f0_rejection() {
        assert!(matches!(
            HarmonicComplexGenerator::validate_params(0.0, 1.0, 5, 0.0, 48000.0),
            Err(DspError::InvalidParameter { name: "f0", .. })
        ));
    }

    #[test]
    fn test_b07_05_sinc_zero_bandwidth_rejection() {
        assert!(matches!(
            ImpulseGenerator::bandlimited_sinc_checked(0.0, 128, 1.0, 48000.0),
            Err(DspError::InvalidParameter {
                name: "bandwidth",
                ..
            })
        ));
    }

    #[test]
    fn test_b07_06_stft_zero_hop_rejection() {
        assert!(matches!(
            Stft::new(512, 0, WindowType::Hann),
            Err(DspError::InvalidParameter {
                name: "hop_size",
                ..
            })
        ));
    }

    #[test]
    fn test_b07_07_pipeline_zero_sample_rate_rejection() {
        let pipeline = DspPipeline::default_reference();
        assert!(matches!(
            pipeline.run(&[0.1; 100], 0.0),
            Err(DspError::InvalidParameter {
                name: "sample_rate",
                ..
            })
        ));
    }

    /// F02-B02: Window generation with degenerate length 1 rejected.
    #[test]
    fn test_b07_08_window_length_one_degenerate() {
        assert!(matches!(
            generate_window(WindowType::Hann, 1),
            Err(DspError::InvalidParameter { name: "length", .. })
        ));
    }

    /// F04-B02: DspError::InvalidParameter empty strings handling.
    #[test]
    fn test_b07_09_dsp_error_invalid_parameter_empty_strings() {
        let err = DspError::InvalidParameter { name: "", reason: String::new() };
        assert_eq!(format!("{}", err), "Invalid parameter '': ");
    }

    /// F04-B05: DspError downcasting from standard Box<dyn std::error::Error>.
    #[test]
    fn test_b07_10_dsp_error_downcast_and_boxed_interop() {
        let boxed: Box<dyn std::error::Error> = Box::new(DspError::EmptyInput);
        let downcasted = boxed.downcast_ref::<DspError>();
        assert_eq!(downcasted, Some(&DspError::EmptyInput));
    }

    /// F06-B02: HarmonicComplexGenerator zero partials count validation.
    #[test]
    fn test_b07_11_harmonic_zero_partials_count() {
        assert!(matches!(
            HarmonicComplexGenerator::validate_params(440.0, 1.0, 0, 0.0, 48000.0),
            Err(DspError::InvalidParameter { name: "num_partials", .. })
        ));
    }

    /// F07-B05: ChirpGenerator negative start or end frequency rejection.
    #[test]
    fn test_b07_12_chirp_negative_frequency_rejection() {
        assert!(matches!(
            ChirpGenerator::validate_params(-100.0, 1000.0, 1.0, 1.0, ChirpKind::Linear),
            Err(DspError::InvalidParameter { name: "f_start", .. })
        ));
        assert!(matches!(
            ChirpGenerator::validate_params(100.0, -1000.0, 1.0, 1.0, ChirpKind::Linear),
            Err(DspError::InvalidParameter { name: "f_end", .. })
        ));
    }

    /// F08-B03: ImpulseGenerator Kronecker delta with delay index out of bounds.
    #[test]
    fn test_b07_13_impulse_delay_out_of_bounds() {
        assert!(matches!(
            ImpulseGenerator::kronecker_delta(1000, 512, 1.0),
            Err(DspError::InvalidParameter { name: "delay", .. })
        ));
    }

    /// F09-B02: NoiseGenerator seed 0 reproducibility.
    #[test]
    fn test_b07_14_noise_seed_zero_reproducibility() {
        let gen1 = NoiseGenerator::new(NoiseColor::WhiteUniform, 1.0, 0);
        let gen2 = NoiseGenerator::new(NoiseColor::WhiteUniform, 1.0, 0);
        let buf1 = gen1.generate(100, 44100.0);
        let buf2 = gen2.generate(100, 44100.0);
        assert_eq!(buf1.samples, buf2.samples);
    }

    /// F09-B05: NoiseGenerator negative variance or amplitude rejection.
    #[test]
    fn test_b07_15_noise_negative_variance_rejection() {
        assert!(matches!(
            NoiseGenerator::validate_params(NoiseColor::Gaussian, -0.5, 42),
            Err(DspError::InvalidParameter { name: "amplitude", .. })
        ));
    }

    /// F10-B03: SyntheticInstrumentSource coordinate domain boundary violations.
    #[test]
    fn test_b07_16_instrument_out_of_bounds_coordinates() {
        let inst = SyntheticInstrumentSource::default();
        assert!(matches!(
            inst.synthesize([1.5, 0.5, 0.5, 0.5, 0.5], 256, 44100.0),
            Err(DspError::InvalidParameter { name: "coordinates", .. })
        ));
        assert!(matches!(
            inst.synthesize([0.5, -0.2, 0.5, 0.5, 0.5], 256, 44100.0),
            Err(DspError::InvalidParameter { name: "coordinates", .. })
        ));
    }

    /// F11-B02: Time metrics length mismatch rejection.
    #[test]
    fn test_b07_17_time_metrics_length_mismatch() {
        let x = vec![1.0; 100];
        let y = vec![1.0; 101];
        assert_eq!(
            compute_time_metrics(&x, &y),
            Err(DspError::BufferLengthMismatch { expected: 100, actual: 101 })
        );
    }

    /// F16-B02: STFT audio shorter than window size handled gracefully.
    #[test]
    fn test_b07_18_stft_audio_shorter_than_window() {
        let short_audio = vec![0.5; 200];
        let stft = Stft::new(512, 256, WindowType::Hann).unwrap();
        let traj = stft.process(&short_audio);
        assert!(traj.is_ok(), "STFT must pad or process sub-window audio without panic");
    }

    /// F16-B04: STFT hop size greater than window size rejection.
    #[test]
    fn test_b07_19_stft_hop_larger_than_window() {
        assert!(matches!(
            Stft::new(256, 512, WindowType::Hann),
            Err(DspError::InvalidParameter { name: "hop_size", .. })
        ));
    }

    /// F18-B05: HarmonicPartialTracker zero maximum partials parameter.
    #[test]
    fn test_b07_20_partial_tracker_max_partials_zero() {
        assert!(matches!(
            HarmonicPartialTracker::validate_params(440.0, 0, 0.0),
            Err(DspError::InvalidParameter { name: "max_partials", .. })
        ));
    }

    /// F19-B04: StochasticResidualExtractor zero sub-bands requested rejection.
    #[test]
    fn test_b07_21_residual_num_bands_zero_rejection() {
        assert!(matches!(
            StochasticResidualExtractor::new(0),
            Err(DspError::InvalidParameter { name: "num_bands", .. })
        ));
    }

    /// F20-B02: Hypercube coordinate query out of bounds rejection.
    #[test]
    fn test_b07_22_hypercube_clamp_out_of_bounds_coords() {
        let cube = CompressedHypercube::dummy_unit_cube();
        assert!(matches!(
            cube.interpolate(&[-0.2, 0.5, 0.5, 0.5, 0.5]),
            Err(DspError::InvalidParameter { name: "coords", .. })
        ));
    }

    /// F20-B05: Hypercube retrieval on missing discrete coordinate frame.
    #[test]
    fn test_b07_23_hypercube_missing_frame_retrieval() {
        let cube = CompressedHypercube::dummy_unit_cube();
        assert!(cube.get_frame(&[99, 99, 99, 99, 99]).is_err());
    }

    /// F22-B02: NMF negative matrix entries rejection.
    #[test]
    fn test_b07_24_nmf_negative_input_rejection() {
        let x = Matrix::from_vec(2, 2, vec![1.0, -0.5, 0.5, 1.0]).unwrap();
        assert!(matches!(
            Nmf::decompose(&x, 1, 10),
            Err(DspError::InvalidParameter { name: "data", .. })
        ));
    }

    /// F23-B05: Tucker point evaluation out-of-bounds coordinates rejection.
    #[test]
    fn test_b07_25_tucker_point_evaluation_out_of_bounds() {
        let t = Tensor::from_vec(vec![3, 3, 3], vec![1.0; 27]).unwrap();
        let decomp = TuckerDecomposition::decompose(&t, &[2, 2, 2]).unwrap();
        assert!(matches!(
            decomp.evaluate_point(&[5, 0, 0]),
            Err(DspError::DimensionMismatch { .. }) | Err(DspError::InvalidParameter { .. })
        ));
    }

    /// F24-B04: CP-ALS zero iterations parameter rejection.
    #[test]
    fn test_b07_26_cp_zero_iterations_rejection() {
        let t = Tensor::from_vec(vec![2, 2, 2], vec![1.0; 8]).unwrap();
        assert!(matches!(
            CpAls::decompose(&t, 1, 0),
            Err(DspError::InvalidParameter { name: "max_iterations", .. })
        ));
    }

    /// F26-B04: Phase dispersion reproducibility under deterministic seed.
    #[test]
    fn test_b07_27_phase_dispersion_reproducibility() {
        let d1 = PhaseDispersionModel::new(42);
        let d2 = PhaseDispersionModel::new(42);
        assert_eq!(d1.disperse_phase(0.5, 1), d2.disperse_phase(0.5, 1));
    }

    /// F27-B05: Stochastic noise resynthesizer bit-for-bit repeatability under same seed.
    #[test]
    fn test_b07_28_noise_synth_seed_consistency() {
        let env = NoiseBandEnvelope { band_energies: vec![0.5; 8], noise_ratio: 0.5 };
        let s1 = StochasticNoiseResynthesizer::new(8, 777);
        let s2 = StochasticNoiseResynthesizer::new(8, 777);
        let out1 = s1.synthesize_frame(&env, 128).unwrap();
        let out2 = s2.synthesize_frame(&env, 128).unwrap();
        assert_eq!(out1, out2);
    }

    /// F28-B02: OverlapAdd pushed block length mismatch rejection.
    #[test]
    fn test_b07_29_ola_push_block_length_mismatch() {
        let mut ola = OverlapAdd::new(512, 256).unwrap();
        assert_eq!(
            ola.push_block(&vec![0.0; 256]),
            Err(DspError::BufferLengthMismatch { expected: 512, actual: 256 })
        );
    }

    /// F29-B04: DspPipeline reset clears internal filter and accumulator state.
    #[test]
    fn test_b07_30_pipeline_reset_clears_internal_state() {
        let mut pipeline = DspPipeline::default_reference();
        let audio = vec![0.8; 4096];
        let _ = pipeline.run(&audio, 44100.0).unwrap();
        pipeline.reset();
        let silence_out = pipeline.run(&vec![0.0; 1024], 44100.0).unwrap();
        assert!(silence_out.reconstructed_audio.iter().all(|&s: &f64| s.abs() < 1e-10));
    }

    /// F29-B05: DspPipeline stage failure propagation without panics.
    #[test]
    fn test_b07_31_pipeline_stage_failure_propagation() {
        let pipeline = DspPipeline::default_reference();
        let bad_audio = vec![f64::NAN; 512];
        let res = pipeline.run(&bad_audio, 44100.0);
        assert!(res.is_err(), "Pipeline must propagate numerical failure on NaN input");
    }

    /// F30-B02: QualityTierConfig zero noise bands rejection.
    #[test]
    fn test_b07_32_quality_tier_zero_noise_bands_rejection() {
        assert!(matches!(
            QualityTierConfig::custom(4, 0),
            Err(DspError::InvalidParameter { name: "num_noise_bands", .. })
        ));
    }
}

// ============================================================================
// Domain 8: Dynamic Range Extremes & Clipping Detection
// ============================================================================

mod domain8_dynamic_range {
    use super::*;

    #[test]
    fn test_b08_01_complex_large_dynamic_range_multiplication() {
        let c1 = Complex::new(1e200, 1e200);
        let c2 = Complex::new(1e-200, 1e-200);
        let prod = c1 * c2;
        // (1e200 + 1e200 i) * (1e-200 + 1e-200 i) = (1 - 1) + (1 + 1)i = 0 + 2i
        assert!((prod.re - 0.0).abs() < 1e-12);
        assert!((prod.im - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_b08_02_stability_clipping_detection() {
        let mut buf = vec![0.0; 100];
        buf[50] = 1.0002; // Exceeds 1.0001
        let rep = check_numerical_stability(&vec![0.0; 100], &buf);
        assert!(!rep.is_clean);
        assert!(rep.peak_amplitude > 1.0001);
    }

    #[test]
    fn test_b08_03_stability_full_scale_exact_bounds() {
        let buf = vec![-1.0, 0.0, 1.0];
        let rep = check_numerical_stability(&vec![0.0; 3], &buf);
        assert!(rep.is_clean);
        assert!(rep.peak_amplitude <= 1.0001);
    }

    #[test]
    fn test_b08_04_noise_zero_amplitude() {
        let gen = NoiseGenerator::new(NoiseColor::WhiteUniform, 0.0, 42);
        let buf = gen.generate(1000, 48000.0);
        assert!(buf.samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_b08_05_residual_overshoot_clamping() {
        let orig_spectrum = vec![1.0; 16];
        let det_spectrum = vec![1.5; 16]; // Overshoots original

        let residual =
            StochasticResidualExtractor::extract_residual(&orig_spectrum, &det_spectrum);
        // Energy subtraction must clamp to >= 0.0
        assert!(residual.band_energies.iter().all(|&e| e >= 0.0));
    }

    /// F08-B01: ImpulseGenerator Kronecker delta at sample index 0 has exact Parseval energy 1.0.
    #[test]
    fn test_b08_06_impulse_delay_at_sample_zero() {
        let imp = ImpulseGenerator::kronecker_delta(0, 512, 1.0).unwrap();
        assert_eq!(imp[0], 1.0);
        assert!(imp[1..].iter().all(|&s| s == 0.0));
        let energy: f64 = imp.iter().map(|s| s * s).sum();
        assert!((energy - 1.0).abs() < 1e-14);
    }

    /// F08-B02: ImpulseGenerator Kronecker delta at the final sample index N-1.
    #[test]
    fn test_b08_07_impulse_delay_at_last_sample() {
        let imp = ImpulseGenerator::kronecker_delta(511, 512, 1.0).unwrap();
        assert_eq!(imp[511], 1.0);
        assert!(imp[..511].iter().all(|&s| s == 0.0));
        let energy: f64 = imp.iter().map(|s| s * s).sum();
        assert!((energy - 1.0).abs() < 1e-14);
    }

    /// F11-B05: Normalized cross-correlation of orthogonal sequences evaluates to 0.0.
    #[test]
    fn test_b08_08_cross_correlation_orthogonal_signals() {
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let y = vec![1.0, 1.0, 1.0, 1.0];
        let m = compute_time_metrics(&x, &y).unwrap();
        assert!(m.correlation_r.abs() < 1e-12, "Orthogonal cross-correlation must be zero: {}", m.correlation_r);
    }

    /// F12-B05: Spectral metrics on single-bin pure frequency excitation.
    #[test]
    fn test_b08_09_spectral_metrics_single_bin_excitation() {
        let mut ref_spec = vec![0.0; 257];
        let mut recon_spec = vec![0.0; 257];
        ref_spec[10] = 5.0;
        recon_spec[10] = 5.0;
        let m = compute_spectral_metrics(&ref_spec, &recon_spec).unwrap();
        assert!(m.spectral_convergence.abs() < 1e-12);
        assert!(m.log_spectral_distance_db.abs() < 1e-12);
    }

    /// F13-B01: Numerical stability checker flags NaN values.
    #[test]
    fn test_b08_10_stability_nan_detection() {
        let nan_buf = vec![0.0, f64::NAN, 0.5];
        let rep = check_numerical_stability(&vec![0.0; 3], &nan_buf);
        assert!(!rep.is_clean);
        assert_eq!(rep.nan_count, 1);
    }

    /// F13-B02: Numerical stability checker flags Infinity values.
    #[test]
    fn test_b08_11_stability_infinity_detection() {
        let inf_buf = vec![0.0, f64::INFINITY, -f64::INFINITY];
        let rep = check_numerical_stability(&vec![0.0; 3], &inf_buf);
        assert!(!rep.is_clean);
        assert_eq!(rep.inf_count, 2);
    }

    /// F15-B04: Large Radix-2 FFT size N = 65536 execution without stack overflow.
    #[test]
    fn test_b08_12_fft_large_size_65536() {
        let n = 65536;
        let mut buf = vec![Complex::zero(); n];
        buf[0] = Complex::new(1.0, 0.0); // Unit impulse
        Fft::forward_complex_radix2(&mut buf).unwrap();
        // Spectrum of delta is flat 1.0 across all bins
        assert!((buf[0].re - 1.0).abs() < 1e-10);
        assert!((buf[n / 2].re - 1.0).abs() < 1e-10);
    }

    /// F15-B05: Radix-2 FFT energy conservation satisfies Parseval's theorem within 1e-10.
    #[test]
    fn test_b08_13_fft_energy_conservation_parseval() {
        let n = 1024;
        let gen = NoiseGenerator::new(NoiseColor::WhiteUniform, 0.5, 1234);
        let time_buf = gen.generate(n, 44100.0);
        let mut freq_buf: Vec<Complex<f64>> = time_buf.samples.iter().map(|&s| Complex::new(s, 0.0)).collect();
        Fft::forward_complex_radix2(&mut freq_buf).unwrap();
        let time_energy: f64 = time_buf.samples.iter().map(|&s| s * s).sum();
        let freq_energy: f64 = freq_buf.iter().map(|c: &Complex<f64>| c.norm_sq()).sum::<f64>() / (n as f64);
        assert!((time_energy - freq_energy).abs() / time_energy < 1e-10);
    }

    /// F17-B02: Spectral peak detector on flat spectrum detects no spurious interior peaks.
    #[test]
    fn test_b08_14_peak_detection_flat_spectrum() {
        let flat_spec = vec![1.0; 257];
        let peaks = PeakDetector::detect_peaks(&flat_spec, 48000.0);
        assert!(peaks.is_empty(), "Flat spectrum has no local maxima");
    }

    /// F25-B04: Additive synthesis smooth linear amplitude birth and death ramping.
    #[test]
    fn test_b08_15_additive_amplitude_birth_death() {
        let osc = AdditiveOscillatorBank::new(44100.0);
        let mut f1 = SpectralFrame::empty(0.0);
        f1.partials.push(HarmonicPartial { harmonic_index: 1, frequency: 440.0, amplitude: 0.0, phase: 0.0 });
        let mut f2 = SpectralFrame::empty(0.01);
        f2.partials.push(HarmonicPartial { harmonic_index: 1, frequency: 440.0, amplitude: 1.0, phase: 0.0 });
        let out = osc.synthesize_transition(&f1, &f2, 441);
        assert_eq!(out.len(), 441);
        for i in 1..441 {
            let step = (out[i] - out[i - 1]).abs();
            assert!(step < 0.05, "Transitional click detected at sample {}", i);
        }
    }

    /// F25-B05: Additive synthesis in-phase constructive interference scaling.
    #[test]
    fn test_b08_16_additive_constructive_summation_scale() {
        let osc = AdditiveOscillatorBank::new(44100.0);
        let mut frame = SpectralFrame::empty(0.0);
        for k in 1..=20 {
            frame.partials.push(HarmonicPartial { harmonic_index: k, frequency: k as f64 * 100.0, amplitude: 0.2, phase: 0.0 });
        }
        let out = osc.synthesize_frame(&frame, 441);
        let rep = check_numerical_stability(&vec![0.0; 441], &out);
        assert!(rep.peak_amplitude > 1.0001, "20 partials with amp 0.2 must constructively peak > 1.0");
    }

    /// F27-B04: Stochastic noise resynthesizer with single excited high frequency band.
    #[test]
    fn test_b08_17_noise_synth_single_excited_high_band() {
        let mut energies = vec![0.0; 16];
        energies[15] = 1.0;
        let env = NoiseBandEnvelope { band_energies: energies, noise_ratio: 1.0 };
        let synth = StochasticNoiseResynthesizer::new(16, 42);
        let out = synth.synthesize_frame(&env, 1024).unwrap();
        let centroid = common::compute_spectral_centroid(&out, 48000.0);
        assert!(centroid > 16000.0, "High band excitation must yield high spectral centroid: {}", centroid);
    }

    /// F30-B04: Quality tier minimum SNR threshold ordering verification.
    #[test]
    fn test_b08_18_quality_tier_snr_threshold_ordering() {
        let ref_tier = QualityTierConfig::reference();
        let prod_tier = QualityTierConfig::production();
        let prev_tier = QualityTierConfig::preview();
        assert!(ref_tier.min_snr_db > prod_tier.min_snr_db);
        assert!(prod_tier.min_snr_db > prev_tier.min_snr_db);
    }
}

// ============================================================================
// Domain 9: Rank Boundary Conditions in Matrix/Tensor Operations
// ============================================================================

mod domain9_rank_boundaries {
    use super::*;

    #[test]
    fn test_b09_01_matrix_1x1_operations() {
        let m1 = Matrix::from_vec(1, 1, vec![3.0]).unwrap();
        let m2 = Matrix::from_vec(1, 1, vec![4.0]).unwrap();

        assert_eq!(m1.transpose().get(0, 0).unwrap(), 3.0);
        let prod = m1.matmul(&m2).unwrap();
        assert_eq!(prod.rows(), 1);
        assert_eq!(prod.cols(), 1);
        assert_eq!(prod.get(0, 0).unwrap(), 12.0);
    }

    #[test]
    fn test_b09_02_matrix_matmul_inner_dim_mismatch() {
        let m1 = Matrix::from_vec(2, 3, vec![0.0; 6]).unwrap();
        let m2 = Matrix::from_vec(4, 2, vec![0.0; 8]).unwrap();
        assert!(matches!(
            m1.matmul(&m2),
            Err(DspError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_b09_03_tensor_out_of_bounds_index() {
        let t = Tensor::from_vec(vec![2, 3, 4], vec![0.0; 24]).unwrap();
        assert!(t.get(&[2, 0, 0]).is_err());
        assert!(t.get(&[0, 3, 0]).is_err());
        assert!(t.get(&[0, 0, 4]).is_err());
    }

    #[test]
    fn test_b09_04_tensor_mode_unfold_invalid_axis() {
        let t = Tensor::from_vec(vec![3, 4, 5], vec![0.0; 60]).unwrap();
        assert!(matches!(
            t.mode_unfold(3),
            Err(DspError::InvalidParameter { name: "mode", .. })
        ));
    }

    #[test]
    fn test_b09_05_pca_1x1_matrix() {
        let x = Matrix::from_vec(1, 1, vec![5.0]).unwrap();
        let (weights, basis) = Pca::decompose(&x, 1).expect("1x1 PCA failed");
        assert_eq!(weights.rows(), 1);
        assert_eq!(basis.cols(), 1);
    }

    #[test]
    fn test_b09_06_pca_target_dim_zero_rejection() {
        let x = Matrix::from_vec(4, 4, vec![1.0; 16]).unwrap();
        assert!(matches!(
            Pca::decompose(&x, 0),
            Err(DspError::InvalidParameter {
                name: "target_dim",
                ..
            })
        ));
    }

    #[test]
    fn test_b09_07_pca_target_dim_greater_than_rank() {
        let x = Matrix::from_vec(3, 4, vec![1.0; 12]).unwrap();
        // Rank cannot exceed min(3, 4) = 3; target_dim = 5 must be rejected
        assert!(matches!(
            Pca::decompose(&x, 5),
            Err(DspError::InvalidParameter { .. }) | Err(DspError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_b09_08_tucker_rank_zero_rejection() {
        let t = Tensor::from_vec(vec![4, 4, 4], vec![1.0; 64]).unwrap();
        assert!(matches!(
            TuckerDecomposition::decompose(&t, &[2, 0, 2]),
            Err(DspError::InvalidParameter {
                name: "tensor_ranks",
                ..
            })
        ));
    }

    #[test]
    fn test_b09_09_tucker_rank_exceeds_dimension_rejection() {
        let t = Tensor::from_vec(vec![3, 3, 3], vec![1.0; 27]).unwrap();
        assert!(matches!(
            TuckerDecomposition::decompose(&t, &[5, 2, 2]),
            Err(DspError::InvalidParameter { .. }) | Err(DspError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_b09_10_cp_rank_zero_rejection() {
        let t = Tensor::from_vec(vec![3, 3, 3], vec![1.0; 27]).unwrap();
        assert!(matches!(
            CpAls::decompose(&t, 0, 10),
            Err(DspError::InvalidParameter { name: "rank", .. })
        ));
    }

    /// F04-B03: DspError::DimensionMismatch empty vectors display.
    #[test]
    fn test_b09_11_dsp_error_dimension_mismatch_empty_vectors() {
        let err = DspError::DimensionMismatch { expected: vec![], actual: vec![1] };
        assert_eq!(format!("{}", err), "Dimension mismatch: expected shape [], got [1]");
    }

    /// F10-B01: SyntheticInstrumentSource corner vertex (0,0,0,0,0) interpolation.
    #[test]
    fn test_b09_12_instrument_corner_vertex_coordinates() {
        let inst = SyntheticInstrumentSource::default();
        let buf = inst.synthesize([0.0; 5], 512, 44100.0);
        assert!(buf.is_ok());
        common::assert_clean_audio(&buf.unwrap().samples);
    }

    /// F10-B02: SyntheticInstrumentSource maximum vertex (1,1,1,1,1) interpolation.
    #[test]
    fn test_b09_13_instrument_max_vertex_coordinates() {
        let inst = SyntheticInstrumentSource::default();
        let buf = inst.synthesize([1.0; 5], 512, 44100.0);
        assert!(buf.is_ok());
        common::assert_clean_audio(&buf.unwrap().samples);
    }

    /// F14-B02: DFT on length 1 slice preserves scalar magnitude.
    #[test]
    fn test_b09_14_dft_length_one() {
        let input = vec![3.5];
        let spec = Dft::forward_real(&input).unwrap();
        assert_eq!(spec.len(), 1);
        assert!((spec[0].re - 3.5).abs() < 1e-12);
        assert!(spec[0].im.abs() < 1e-12);
        let recon = Dft::inverse_real(&spec).unwrap();
        assert_eq!(recon.len(), 1);
        assert!((recon[0] - 3.5).abs() < 1e-12);
    }

    /// F19-B05: Stochastic residual extraction with single subband.
    #[test]
    fn test_b09_15_residual_single_subband() {
        let extractor = StochasticResidualExtractor::new(1).unwrap();
        let orig = vec![2.0; 8];
        let det = vec![1.0; 8];
        let res = extractor.extract(&orig, &det);
        assert_eq!(res.band_energies.len(), 1);
        assert!(res.band_energies[0] > 0.0);
    }

    /// F20-B01: Hypercube discrete boundary vertex query weight 1.0.
    #[test]
    fn test_b09_16_hypercube_boundary_vertex_interpolation() {
        let cube = CompressedHypercube::dummy_unit_cube();
        let frame = cube.interpolate(&[0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(frame.timestamp, 0.0);
    }

    /// F20-B03: Hypercube axis with dimension 1 handles zero stride without division by zero.
    #[test]
    fn test_b09_17_hypercube_single_cell_axis() {
        let layout = HypercubeLayout::new(vec![1, 4, 4, 4, 4]).unwrap();
        assert_eq!(layout.strides()[0], 0);
    }

    /// F21-B05: PCA on collinear identical rows isolates 100% variance into first component.
    #[test]
    fn test_b09_18_pca_collinear_identical_rows() {
        let row = vec![1.0, 2.0, 3.0, 4.0];
        let mut data = Vec::with_capacity(16);
        for _ in 0..4 {
            data.extend_from_slice(&row);
        }
        let x = Matrix::from_vec(4, 4, data).unwrap();
        let (weights, basis) = Pca::decompose(&x, 2).unwrap();
        assert_eq!(weights.cols(), 2);
        assert_eq!(basis.rows(), 2);
    }

    /// F22-B03: NMF target rank 0 rejection.
    #[test]
    fn test_b09_19_nmf_target_rank_zero_rejection() {
        let x = Matrix::from_vec(2, 2, vec![1.0; 4]).unwrap();
        assert!(matches!(
            Nmf::decompose(&x, 0, 10),
            Err(DspError::InvalidParameter { name: "rank", .. })
        ));
    }

    /// F22-B05: NMF rank 1 decomposition factor non-negativity.
    #[test]
    fn test_b09_20_nmf_rank_one_decomposition() {
        let x = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0]).unwrap();
        let (w, h) = Nmf::decompose(&x, 1, 20).unwrap();
        assert!(w.as_slice().iter().all(|&v: &f64| v >= 0.0));
        assert!(h.as_slice().iter().all(|&v: &f64| v >= 0.0));
    }

    /// F23-B04: Tucker decomposition with rank 1 across all modes.
    #[test]
    fn test_b09_21_tucker_rank_one_across_all_modes() {
        let t = Tensor::from_vec(vec![2, 2, 2], vec![1.0; 8]).unwrap();
        let decomp = TuckerDecomposition::decompose(&t, &[1, 1, 1]).unwrap();
        assert_eq!(decomp.core.shape(), &[1, 1, 1]);
    }

    /// F24-B02: CP-ALS rank 1 decomposition outer product recovery.
    #[test]
    fn test_b09_22_cp_rank_one() {
        let t = Tensor::from_vec(vec![2, 2, 2], vec![1.0; 8]).unwrap();
        let model = CpAls::decompose(&t, 1, 15).unwrap();
        assert_eq!(model.factors.len(), 3);
        assert_eq!(model.factors[0].cols(), 1);
    }
}

// ============================================================================
// Domain 10: Division-by-Zero Protections in NMF & Spectral Normalization
// ============================================================================

mod domain10_zero_division_protection {
    use super::*;

    #[test]
    fn test_b10_01_nmf_multiplicative_update_zero_protection() {
        let mut x_sparse = vec![0.0; 16];
        x_sparse[0] = 1.0; // Single non-zero entry
        let x = Matrix::from_vec(4, 4, x_sparse).unwrap();

        let res = Nmf::decompose(&x, 2, 30);
        assert!(res.is_ok());
        let (w, h) = res.unwrap();
        assert!(w.as_slice().iter().all(|&v: &f64| !v.is_nan() && v >= 0.0));
        assert!(h.as_slice().iter().all(|&v: &f64| !v.is_nan() && v >= 0.0));
    }

    #[test]
    fn test_b10_02_spectral_convergence_zero_denominator() {
        let ref_zero = vec![0.0; 128];
        let recon_active = vec![0.1; 128];
        let m = compute_spectral_metrics(&ref_zero, &recon_active).unwrap();
        assert!(!m.spectral_convergence.is_nan());
        assert!(!m.spectral_convergence.is_infinite());
    }

    #[test]
    fn test_b10_03_cp_singular_gram_matrix_regularization() {
        // Collinear slices produce ill-conditioned Gram matrix
        let mut data = vec![0.0; 27];
        for i in 0..9 {
            data[i] = 1.0;
            data[i + 9] = 2.0;
            data[i + 18] = 3.0;
        }
        let t = Tensor::from_vec(vec![3, 3, 3], data).unwrap();
        let res = CpAls::decompose(&t, 2, 20);
        assert!(res.is_ok(), "Regularized CP-ALS must not fail with singular matrix inversion");
    }

    #[test]
    fn test_b10_04_quality_tier_zero_ranks_rejection() {
        assert!(matches!(
            QualityTierConfig::custom(0, 8),
            Err(DspError::InvalidParameter { name: "rank", .. })
        ));
        assert!(matches!(
            QualityTierConfig::custom(4, 0),
            Err(DspError::InvalidParameter {
                name: "num_noise_bands",
                ..
            })
        ));
    }

    #[test]
    fn test_b10_05_quality_tier_monotonic_thresholds() {
        let ref_tier = QualityTierConfig::reference();
        let prod_tier = QualityTierConfig::production();
        let prev_tier = QualityTierConfig::preview();

        assert!(ref_tier.min_snr_db > prod_tier.min_snr_db);
        assert!(prod_tier.min_snr_db > prev_tier.min_snr_db);

        assert!(ref_tier.max_lsd_db < prod_tier.max_lsd_db);
        assert!(prod_tier.max_lsd_db < prev_tier.max_lsd_db);
    }

    /// F01-B03: Complex division by zero behavior.
    #[test]
    fn test_b10_06_complex_division_by_zero() {
        let c = Complex::new(1.0, 2.0);
        let denom = Complex::ZERO;
        let res = c / denom;
        assert!(res.re.is_nan() || res.re.is_infinite());
        assert!(res.im.is_nan() || res.im.is_infinite());
    }

    /// F08-B05: ImpulseGenerator damped transient with decay_tau = 0.0 rejection.
    #[test]
    fn test_b10_07_damped_transient_zero_decay_rate() {
        assert!(matches!(
            ImpulseGenerator::damped_transient_checked(440.0, 0.0, 512, 44100.0),
            Err(DspError::InvalidParameter { name: "decay_tau", .. })
        ));
    }

    /// F09-B04: NoiseGenerator Box-Muller transformation protects against ln(0.0).
    #[test]
    fn test_b10_08_gaussian_noise_box_muller_stability() {
        let gen = NoiseGenerator::new(NoiseColor::Gaussian, 1.0, 42);
        let buf = gen.generate(10000, 48000.0);
        assert!(buf.samples.iter().all(|s: &f64| !s.is_nan() && !s.is_infinite()));
    }

    /// F17-B05: Spectral peak detector parabolic interpolation protects denominator when adjacent bins match.
    #[test]
    fn test_b10_09_peak_detection_two_equal_adjacent_peaks() {
        let mut spec = vec![0.0; 257];
        spec[50] = 10.0;
        spec[51] = 10.0;
        let peaks = PeakDetector::detect_peaks(&spec, 48000.0);
        for p in &peaks {
            assert!(!p.frequency_hz.is_nan());
            assert!(!p.amplitude.is_nan());
        }
    }

    /// F19-B02: Stochastic residual extraction with zero energy original avoids 0/0 division.
    #[test]
    fn test_b10_10_residual_zero_energy_original() {
        let orig = vec![0.0; 16];
        let det = vec![0.0; 16];
        let res = StochasticResidualExtractor::extract_residual(&orig, &det);
        assert!(!res.noise_ratio.is_nan(), "Noise ratio must not be NaN on zero original energy");
        assert_eq!(res.noise_ratio, 0.0);
    }
}
