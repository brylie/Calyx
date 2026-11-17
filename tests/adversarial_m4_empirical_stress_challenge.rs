//! tests/adversarial_m4_empirical_stress_challenge.rs
//! Adversarial M4 Empirical Stress & Robustness Challenge Harness.
//!
//! Stress-tests:
//! 1. Numerical Invariants (Zero NaNs, Zero Infs, clipping bounds, energy conservation)
//! 2. Pathological & Edge-Case Signals (Silence, Pure DC, Kronecker Impulse, Sinc pulse, Sub-bass, Near-Nyquist, Heavy Noise)
//! 3. Arbitrary & Prime Buffer Lengths
//! 4. Extreme Decomposition Dimensions (1x1, 1xK, Mx1, Rank 1, High-rank, Collinear)
//! 5. Concurrency, Deadlock & Throughput Stress (Multi-threaded parallel pipeline execution)
//! 6. Long-duration Phase & Energy Continuity (Multi-second signal reconstruction)

// use calyx_dsp::error::*;
use calyx_dsp::hypercube::*;
use calyx_dsp::math::*;
use calyx_dsp::metrics::*;
use calyx_dsp::pipeline::*;
use calyx_dsp::synthetic::*;

use std::thread;

// ============================================================================
// 1. Pathological Signal Stress
// ============================================================================

#[test]
fn challenge_pipeline_pure_silence_all_tiers() {
    let fs = 48000.0;
    let n_samples = 4800; // 0.1s
    let silence = vec![0.0; n_samples];

    let tiers = [
        QualityTierConfig::reference(),
        QualityTierConfig::production(),
        QualityTierConfig::preview(),
    ];

    for config in tiers {
        let pipeline = DspPipeline::from_config(config);
        let result = pipeline.run(&silence, fs).expect("Pipeline should not fail on silence");

        assert_eq!(result.reconstructed_audio.len(), n_samples);
        for &s in &result.reconstructed_audio {
            assert!(!s.is_nan(), "Sample must not be NaN");
            assert!(!s.is_infinite(), "Sample must not be Infinite");
            assert!(s.abs() < 1e-6, "Silent signal reconstruction must be near zero, got {}", s);
        }
    }
}

#[test]
fn challenge_pipeline_pure_dc_signal() {
    let fs = 44100.0;
    let n_samples = 4410;
    let dc_signal = vec![0.5; n_samples];

    let pipeline = DspPipeline::from_config(QualityTierConfig::production());
    let result = pipeline.run(&dc_signal, fs).expect("Pipeline should not fail on DC");

    assert_eq!(result.reconstructed_audio.len(), n_samples);
    for &s in &result.reconstructed_audio {
        assert!(!s.is_nan(), "Sample must not be NaN");
        assert!(!s.is_infinite(), "Sample must not be Infinite");
    }
}

#[test]
fn challenge_pipeline_kronecker_impulse() {
    let fs = 48000.0;
    let n_samples = 4800;
    let mut impulse = vec![0.0; n_samples];
    impulse[0] = 0.99; // Sharp delta at sample 0

    let pipeline = DspPipeline::from_config(QualityTierConfig::reference());
    let result = pipeline.run(&impulse, fs).expect("Pipeline should handle Kronecker impulse");

    assert_eq!(result.reconstructed_audio.len(), n_samples);
    verify_numerical_stability(&result.reconstructed_audio).expect("Impulse output must be stable");
}

#[test]
fn challenge_pipeline_bandlimited_sinc_pulse() {
    let fs = 48000.0;
    let n_samples = 4800;
    let sinc = ImpulseGenerator::bandlimited_sinc(4000.0, 2400, 0.9).generate(n_samples, fs).samples;

    let pipeline = DspPipeline::from_config(QualityTierConfig::reference());
    let result = pipeline.run(&sinc, fs).expect("Pipeline should handle sinc pulse");

    assert_eq!(result.reconstructed_audio.len(), n_samples);
    verify_numerical_stability(&result.reconstructed_audio).expect("Sinc output must be stable");
}

#[test]
fn challenge_pipeline_sub_bass_and_near_nyquist() {
    let fs = 48000.0;
    let n_samples = 9600; // 0.2s

    // Sub-bass 5 Hz
    let sub_bass = SineGenerator::new(5.0, 0.7, 0.0).generate(n_samples, fs).samples;
    let pipeline = DspPipeline::from_config(QualityTierConfig::production());
    let res_bass = pipeline.run(&sub_bass, fs).expect("Pipeline should handle sub-bass");
    verify_numerical_stability(&res_bass.reconstructed_audio).expect("Sub-bass output must be stable");

    // Near-Nyquist: 23,500 Hz (Nyquist is 24,000 Hz)
    let near_nyquist = SineGenerator::new(23500.0, 0.7, 0.0).generate(n_samples, fs).samples;
    let res_nyq = pipeline.run(&near_nyquist, fs).expect("Pipeline should handle near-Nyquist");
    verify_numerical_stability(&res_nyq.reconstructed_audio).expect("Near-Nyquist output must be stable");
}

#[test]
fn challenge_pipeline_extreme_noise_floor() {
    let fs = 48000.0;
    let n_samples = 9600;
    // Heavy pink noise (0.8 amplitude) with no distinct harmonics
    let heavy_noise = NoiseGenerator::new(NoiseColor::Pink, 0.8, 99999).generate(n_samples, fs).samples;

    let pipeline = DspPipeline::from_config(QualityTierConfig::production());
    let result = pipeline.run(&heavy_noise, fs).expect("Pipeline should handle heavy noise without crashing");

    assert_eq!(result.reconstructed_audio.len(), n_samples);
    verify_numerical_stability(&result.reconstructed_audio).expect("Heavy noise reconstruction must be stable");
}

// ============================================================================
// 2. Arbitrary & Non-Power-of-Two Buffer Length Stress
// ============================================================================

#[test]
fn challenge_pipeline_prime_and_boundary_buffer_lengths() {
    let fs = 44100.0;
    let pipeline = DspPipeline::from_config(QualityTierConfig::production());

    // Test a variety of prime and unaligned buffer sizes with adequate duration:
    let test_lengths = [
        2053,  // Prime (~46ms)
        3333,  // Odd (~75ms)
        7777,  // Large prime (~176ms)
        12345, // Arbitrary (~280ms)
        22050, // Half second (standard benchmark length)
    ];

    for &len in &test_lengths {
        let sig = SineGenerator::new(440.0, 0.7, 0.0).generate(len, fs).samples;
        let res = pipeline.run(&sig, fs).unwrap_or_else(|e| panic!("Failed on buffer length {}: {:?}", len, e));

        assert_eq!(res.reconstructed_audio.len(), len, "Output length mismatch for input length {}", len);
        verify_numerical_stability(&res.reconstructed_audio).unwrap_or_else(|e| panic!("Stability failed on len {}: {:?}", len, e));

        let tm = compute_time_metrics(&sig, &res.reconstructed_audio).unwrap();
        eprintln!("Length {} -> SNR: {:.2} dB", len, tm.snr_db);
        assert!(tm.snr_db > 25.0, "SNR too low ({:.2} dB) for length {}", tm.snr_db, len);
    }
}

#[test]
fn challenge_pipeline_sub_window_audio_handling() {
    let fs = 48000.0;
    let pipeline = DspPipeline::from_config(QualityTierConfig::production());
    // Production window is 512 samples. Test audio shorter than window (200 samples).
    let short_sig = SineGenerator::new(440.0, 0.7, 0.0).generate(200, fs).samples;
    let result = pipeline.run(&short_sig, fs).expect("Sub-window audio should be zero-padded and processed cleanly");
    assert_eq!(result.reconstructed_audio.len(), 200);
    verify_numerical_stability(&result.reconstructed_audio).expect("Reconstructed sub-window audio must be stable");
}

// ============================================================================
// 3. Extreme Decomposition Dimensions & Degenerate Matrices
// ============================================================================

#[test]
fn challenge_decompositions_degenerate_matrices() {
    // 1. All-zeros matrix
    let zeros = Matrix::zeros(10, 10).unwrap();
    let (w_pca, b_pca) = Pca::decompose(&zeros, 3).unwrap();
    assert_eq!(w_pca.rows(), 10);
    assert_eq!(b_pca.cols(), 10);
    for &v in w_pca.as_slice() { assert_eq!(v, 0.0); }
    for &v in b_pca.as_slice() { assert_eq!(v, 0.0); }

    let (w_nmf, h_nmf) = Nmf::decompose(&zeros, 3, 20).unwrap();
    for &v in w_nmf.as_slice() { assert!(!v.is_nan() && !v.is_infinite()); }
    for &v in h_nmf.as_slice() { assert!(!v.is_nan() && !v.is_infinite()); }

    // 2. Collinear (identical) rows (Rank 1 matrix)
    let mut collinear = Matrix::zeros(20, 15).unwrap();
    let base_row: Vec<f64> = (1..=15).map(|i| i as f64).collect();
    for r in 0..20 {
        let scale = (r + 1) as f64;
        for c in 0..15 {
            collinear[(r, c)] = base_row[c] * scale;
        }
    }

    let (w_pca_col, b_pca_col) = Pca::decompose(&collinear, 4).unwrap();
    let recon_pca = w_pca_col.matmul(&b_pca_col).unwrap();
    for r in 0..20 {
        for c in 0..15 {
            let diff = (collinear[(r, c)] - recon_pca[(r, c)]).abs();
            assert!(diff < 1e-4, "Collinear PCA error at ({}, {}): diff={}", r, c, diff);
        }
    }

    // 3. 1x1 Matrix
    let m_1x1 = Matrix::from_vec(1, 1, vec![42.0]).unwrap();
    let (w_1, b_1) = Pca::decompose(&m_1x1, 1).unwrap();
    let recon_1 = w_1.matmul(&b_1).unwrap();
    assert!((recon_1[(0, 0)] - 42.0).abs() < 1e-10);

    // 4. Large Matrix Performance (100 x 50)
    let mut large_m = Matrix::zeros(100, 50).unwrap();
    for r in 0..100 {
        for c in 0..50 {
            large_m[(r, c)] = ((r * 3 + c * 7) % 23) as f64 * 0.1;
        }
    }
    let start = std::time::Instant::now();
    let (w_lg, b_lg) = Pca::decompose(&large_m, 8).unwrap();
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() < 500, "100x50 PCA took too long: {:?}", elapsed);
    assert_eq!(w_lg.cols(), 8);
    assert_eq!(b_lg.rows(), 8);
}

#[test]
fn challenge_tucker_and_cp_decompositions_robustness() {
    // 3D Tensor Tucker test with rank 1
    let shape = vec![4, 5, 6];
    let data = vec![1.5; 4 * 5 * 6];
    let tensor = Tensor::new(shape.clone(), data).unwrap();

    let ranks = vec![1, 1, 1];
    let tucker = TuckerDecomposition::decompose(&tensor, &ranks).expect("Tucker rank-1 should succeed");
    assert_eq!(tucker.core.shape(), &[1, 1, 1]);
    for f in &tucker.factors {
        assert_eq!(f.cols(), 1);
    }

    // Point evaluation
    let eval = tucker.evaluate_point(&[0, 0, 0]).unwrap();
    assert!(!eval.is_nan() && !eval.is_infinite());

    // CP-ALS with rank 2
    let cp = CpAls::decompose(&tensor, 2, 20).expect("CP-ALS should succeed on uniform tensor");
    assert_eq!(cp.weights.len(), 2);
    for factor in &cp.factors {
        for &val in factor.as_slice() {
            assert!(!val.is_nan() && !val.is_infinite(), "CP factor contains NaN/Inf: {}", val);
        }
    }
}

// ============================================================================
// 4. Long Duration & Phase Continuity Stress
// ============================================================================

#[test]
fn challenge_long_duration_harmonic_signal() {
    let fs = 48000.0;
    let n_samples = 48000 * 3; // 3 full seconds (144,000 samples)
    // Use base_amplitude 0.5 so that the sum of partial amplitudes stays safely within [-1.0, 1.0]
    let harm = HarmonicComplexGenerator::new(220.0, 0.5, 8, 0.0, HarmonicRecipe::Sawtooth)
        .generate(n_samples, fs).samples;

    // Check that input signal itself adheres to dynamic range [-1.0, 1.0]
    let in_peak = harm.iter().copied().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(in_peak <= 1.0, "Input signal peak was {}", in_peak);

    let pipeline = DspPipeline::from_config(QualityTierConfig::reference());
    let start = std::time::Instant::now();
    let out = pipeline.run(&harm, fs).expect("3-second pipeline should run without error");
    let elapsed = start.elapsed();

    eprintln!("3-second signal execution time: {:.2}s", elapsed.as_secs_f64());
    assert!(elapsed.as_secs_f64() < 5.0, "Execution took longer than 5 seconds: {:?}", elapsed);

    assert_eq!(out.reconstructed_audio.len(), n_samples);
    verify_numerical_stability(&out.reconstructed_audio).expect("Reconstructed audio must be numerically stable");

    let tm = compute_time_metrics(&harm, &out.reconstructed_audio).unwrap();
    let sm = compute_spectral_metrics(&harm, &out.reconstructed_audio).unwrap();

    eprintln!("3-second Sawtooth: SNR = {:.2} dB, LSD = {:.2} dB", tm.snr_db, sm.log_spectral_distance_db);
    assert!(tm.snr_db > 35.0, "Long signal SNR must be > 35 dB, got {:.2} dB", tm.snr_db);
    assert!(sm.log_spectral_distance_db < 0.8, "Long signal LSD must be < 0.8 dB, got {:.2} dB", sm.log_spectral_distance_db);
}

// ============================================================================
// 5. Multi-Threaded Concurrency & Deadlock Stress
// ============================================================================

#[test]
fn challenge_parallel_pipeline_concurrency_no_deadlocks() {
    let fs = 48000.0;
    let n_samples = 9600;

    let num_threads = 8;
    let mut handles = Vec::with_capacity(num_threads);

    for thread_idx in 0..num_threads {
        let handle = thread::spawn(move || {
            let freq = 110.0 * (thread_idx + 1) as f64;
            let sig = HarmonicComplexGenerator::new(freq, 0.5, 6, 0.0, HarmonicRecipe::Sawtooth)
                .generate(n_samples, fs).samples;

            let tier_cfg = match thread_idx % 3 {
                0 => QualityTierConfig::reference(),
                1 => QualityTierConfig::production(),
                _ => QualityTierConfig::preview(),
            };

            let pipe = DspPipeline::from_config(tier_cfg);
            let out = pipe.run(&sig, fs).unwrap();

            assert_eq!(out.reconstructed_audio.len(), n_samples);
            verify_numerical_stability(&out.reconstructed_audio).unwrap();

            let tm = compute_time_metrics(&sig, &out.reconstructed_audio).unwrap();
            assert!(tm.snr_db > 20.0);
        });
        handles.push(handle);
    }

    for (i, h) in handles.into_iter().enumerate() {
        h.join().unwrap_or_else(|_| panic!("Thread {} panicked or deadlocked!", i));
    }
}

// ============================================================================
// 6. Expressive Instrument 5D Hypercube Query Stability
// ============================================================================

#[test]
fn challenge_instrument_hypercube_extreme_coordinates() {
    let fs = 48000.0;
    let inst = SyntheticInstrumentSource::default();

    // Test extreme corner coordinates:
    let coords_to_test = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [0.5, 0.5, 0.5, 0.5, 0.5],
    ];

    for (idx, coord) in coords_to_test.iter().enumerate() {
        let buffer = inst.synthesize(*coord, 4800, fs).expect("Instrument note generation failed");
        assert!(!buffer.samples.is_empty());
        verify_numerical_stability(&buffer.samples).unwrap_or_else(|e| {
            panic!("Coordinate set {} failed stability: {:?}", idx, e);
        });
    }
}
