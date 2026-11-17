//! tests/adversarial_m5_stress_challenge.rs
//! Milestone 5 Adversarial Full Suite Empirical Stress Harness.
//!
//! Authored by Challenger 1 (Milestone 5 Final Full Suite Empirical Stress Verifier).
//! Stress Dimensions:
//! 1. Denormal float resistance (IEEE 754 denormals: f64::MIN_POSITIVE * 0.25).
//! 2. Pathological & Edge Waveforms:
//!    - Pure Nyquist square wave alternating [-0.8, 0.8, -0.8, 0.8].
//!    - Sparse Poisson-distributed impulsive spikes.
//!    - Near-zero amplitude pianissimo signals (1e-7).
//! 3. Prime & Unaligned Buffer Lengths (1009, 2053, 4099, 8191).
//! 4. Multithreaded High-Concurrency Pipeline Execution (16 threads).
//! 5. High-Order Tensor Tucker & CP-ALS Decomposition Robustness.
//! 6. Rapid consecutive pipeline reusability & state isolation.

use calyx_dsp::hypercube::{CpAls, TuckerDecomposition};
use calyx_dsp::math::Tensor;
use calyx_dsp::metrics::{compute_time_metrics, verify_numerical_stability};
use calyx_dsp::pipeline::{DspPipeline, QualityTierConfig};
use calyx_dsp::synthetic::{
    HarmonicComplexGenerator, HarmonicRecipe, SignalGenerator, SineGenerator,
};
use std::thread;

// ============================================================================
// 1. Denormal Float & Micro-Amplitude Audio Stress
// ============================================================================

#[test]
fn challenge_m5_denormal_and_subnormal_signals() {
    let fs = 48000.0;
    let n_samples = 4800;

    // Construct a signal containing subnormal floats (1e-315 to 1e-308)
    let denorm_val = f64::MIN_POSITIVE * 0.25;
    assert!(denorm_val > 0.0 && denorm_val < f64::MIN_POSITIVE);
    let denorm_signal = vec![denorm_val; n_samples];

    let tiers = [
        QualityTierConfig::reference(),
        QualityTierConfig::production(),
        QualityTierConfig::preview(),
    ];

    for tier in tiers {
        let pipeline = DspPipeline::from_config(tier);
        let result = pipeline
            .run(&denorm_signal, fs)
            .expect("Pipeline must handle denormal audio gracefully without panic");

        assert_eq!(result.reconstructed_audio.len(), n_samples);
        for &sample in &result.reconstructed_audio {
            assert!(!sample.is_nan(), "Sample must not be NaN");
            assert!(!sample.is_infinite(), "Sample must not be Infinite");
            assert!(
                sample.abs() < 1e-4,
                "Denormal reconstruction must remain negligible: got {}",
                sample
            );
        }
    }
}

#[test]
fn challenge_m5_micro_amplitude_pianissimo_signal() {
    let fs = 48000.0;
    let n_samples = 4800;
    // Micro-amplitude 1e-7 tone
    let sine = SineGenerator::new(440.0, 1e-7, 0.0).generate(n_samples, fs).samples;

    let pipeline = DspPipeline::from_config(QualityTierConfig::production());
    let res = pipeline.run(&sine, fs).expect("Micro-amplitude signal failed");

    assert_eq!(res.reconstructed_audio.len(), n_samples);
    verify_numerical_stability(&res.reconstructed_audio)
        .expect("Micro-amplitude reconstruction must pass stability check");
}

// ============================================================================
// 2. Pathological & Extreme Waveforms
// ============================================================================

#[test]
fn challenge_m5_alternating_nyquist_square_wave() {
    let fs = 48000.0;
    let n_samples = 4800;
    // Alternating +- 0.8 at Nyquist frequency (fs/2 = 24000 Hz)
    let mut nyq_wave = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        nyq_wave.push(if i % 2 == 0 { 0.8 } else { -0.8 });
    }

    let pipeline = DspPipeline::from_config(QualityTierConfig::production());
    let res = pipeline
        .run(&nyq_wave, fs)
        .expect("Pipeline must handle exact Nyquist square wave");

    assert_eq!(res.reconstructed_audio.len(), n_samples);
    verify_numerical_stability(&res.reconstructed_audio)
        .expect("Nyquist wave reconstruction must remain numerically stable");
}

#[test]
fn challenge_m5_sparse_poisson_impulsive_train() {
    let fs = 48000.0;
    let n_samples = 9600;
    let mut signal = vec![0.0; n_samples];

    // Seeded pseudo-random sparse impulsive train
    let mut state: u64 = 0xDEADBEEFCAFE;
    for i in 0..n_samples {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        if (state >> 56) < 4 {
            // ~1.5% impulse density
            let sign = if (state & 1) == 0 { 1.0 } else { -1.0 };
            signal[i] = sign * 0.95;
        }
    }

    let pipeline = DspPipeline::from_config(QualityTierConfig::reference());
    let res = pipeline
        .run(&signal, fs)
        .expect("Pipeline must process sparse impulsive audio without failure");

    assert_eq!(res.reconstructed_audio.len(), n_samples);
    verify_numerical_stability(&res.reconstructed_audio)
        .expect("Impulsive reconstruction must pass stability invariants");
}

// ============================================================================
// 3. Prime & Unaligned Buffer Lengths
// ============================================================================

#[test]
fn challenge_m5_prime_buffer_lengths_all_tiers() {
    let fs = 44100.0;
    let prime_lengths = [1009, 2053, 4099, 8191];

    for &len in &prime_lengths {
        let sig = SineGenerator::new(880.0, 0.6, 0.0).generate(len, fs).samples;

        for tier in [
            QualityTierConfig::reference(),
            QualityTierConfig::production(),
            QualityTierConfig::preview(),
        ] {
            let pipeline = DspPipeline::from_config(tier);
            let res = pipeline
                .run(&sig, fs)
                .unwrap_or_else(|e| panic!("Failed on prime len {} tier {:?}: {:?}", len, tier.tier, e));

            assert_eq!(res.reconstructed_audio.len(), len);
            verify_numerical_stability(&res.reconstructed_audio).unwrap();
        }
    }
}

// ============================================================================
// 4. Multi-Threaded Concurrency (16 Threads)
// ============================================================================

#[test]
fn challenge_m5_heavy_concurrency_16_threads() {
    let fs = 48000.0;
    let n_samples = 4800;
    let num_threads = 16;
    let mut handles = Vec::with_capacity(num_threads);

    for t in 0..num_threads {
        let handle = thread::spawn(move || {
            let freq = 100.0 + (t as f64) * 75.0;
            let sig = HarmonicComplexGenerator::new(freq, 0.45, 8, 0.0001, HarmonicRecipe::Sawtooth)
                .generate(n_samples, fs)
                .samples;

            let tier_cfg = match t % 3 {
                0 => QualityTierConfig::reference(),
                1 => QualityTierConfig::production(),
                _ => QualityTierConfig::preview(),
            };

            let pipeline = DspPipeline::from_config(tier_cfg);
            let out = pipeline.run(&sig, fs).expect("Concurrent pipeline run failed");

            assert_eq!(out.reconstructed_audio.len(), n_samples);
            verify_numerical_stability(&out.reconstructed_audio)
                .expect("Concurrent output must be stable");

            let tm = compute_time_metrics(&sig, &out.reconstructed_audio).unwrap();
            assert!(
                tm.snr_db > 18.0,
                "Thread {} SNR too low: {:.2} dB",
                t,
                tm.snr_db
            );
        });
        handles.push(handle);
    }

    for (t, h) in handles.into_iter().enumerate() {
        h.join()
            .unwrap_or_else(|_| panic!("Thread {} panicked or deadlocked!", t));
    }
}

// ============================================================================
// 5. High-Order Tensor Tucker & CP-ALS Decomposition Robustness
// ============================================================================

#[test]
fn challenge_m5_tucker_unbalanced_high_order_tensor() {
    // 4D Tensor with unbalanced dimensions [3, 4, 2, 5]
    let dims = vec![3, 4, 2, 5];
    let total_elements: usize = dims.iter().product();
    let mut data = Vec::with_capacity(total_elements);

    for i in 0..total_elements {
        data.push(((i * 17) % 31) as f64 * 0.1);
    }

    let tensor = Tensor::new(dims.clone(), data).unwrap();
    let ranks = vec![2, 2, 1, 3];

    let tucker = TuckerDecomposition::decompose(&tensor, &ranks)
        .expect("Tucker decomposition failed on 4D unbalanced tensor");

    assert_eq!(tucker.core.shape(), &[2, 2, 1, 3]);
    assert_eq!(tucker.factors.len(), 4);
    assert_eq!(tucker.factors[0].cols(), 2);
    assert_eq!(tucker.factors[1].cols(), 2);
    assert_eq!(tucker.factors[2].cols(), 1);
    assert_eq!(tucker.factors[3].cols(), 3);

    // Evaluate point
    let p_val = tucker.evaluate_point(&[1, 2, 0, 3]).unwrap();
    assert!(!p_val.is_nan() && !p_val.is_infinite());
}

#[test]
fn challenge_m5_cp_als_rank_deficient_matrix_tensor() {
    // 3D Tensor where one slice is all zeros
    let shape = vec![3, 3, 3];
    let mut data = vec![1.0; 27];
    // Zero out the first mode-0 slice
    for i in 0..9 {
        data[i] = 0.0;
    }

    let tensor = Tensor::new(shape, data).unwrap();
    let cp = CpAls::decompose(&tensor, 2, 25).expect("CP-ALS should succeed with regularizer");

    assert_eq!(cp.weights.len(), 2);
    for factor in &cp.factors {
        for &v in factor.as_slice() {
            assert!(!v.is_nan() && !v.is_infinite(), "CP factor contained NaN: {}", v);
        }
    }
}

// ============================================================================
// 6. Rapid Consecutive Pipeline Execution & State Reusability
// ============================================================================

#[test]
fn challenge_m5_rapid_pipeline_reusability() {
    let fs = 48000.0;
    let n_samples = 4800;
    let pipeline = DspPipeline::from_config(QualityTierConfig::production());

    for i in 0..10 {
        let freq = 200.0 + (i as f64) * 50.0;
        let sig = SineGenerator::new(freq, 0.7, 0.0).generate(n_samples, fs).samples;

        let out = pipeline.run(&sig, fs).unwrap_or_else(|e| {
            panic!("Iteration {} failed with error: {:?}", i, e);
        });

        assert_eq!(out.reconstructed_audio.len(), n_samples);
        verify_numerical_stability(&out.reconstructed_audio).unwrap();
        let tm = compute_time_metrics(&sig, &out.reconstructed_audio).unwrap();
        assert!(tm.snr_db > 28.0, "Iteration {} SNR was {:.2} dB", i, tm.snr_db);
    }
}
