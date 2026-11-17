//! tests/tier4_application_tests.rs
//! Opaque-box Tier 4 Real-World Application Scenario Verification Suite for Calyx DSP.
//!
//! Validates authentic sampled acoustic instrument workflows, physics-based string modeling,
//! expressive performance gestures, and multi-tier quality trade-offs:
//!
//! 1. Scenario 1: 5D Synthetic Acoustic Instrument Hypercube (t, p, d, a, v) with Tucker Decomposition
//! 2. Scenario 2: Inharmonic Stiff-String Complex with Multi-Rate Decay & Stochastic Residual
//! 3. Scenario 3: Expressive Gesture (Glissando Pitch Sweep + Tremolo AM + Filter Modulation)
//! 4. Scenario 4: Multi-Tier Quality Trade-Off & Graceful Degradation (Reference, Production, Preview)

#[path = "common/mod.rs"]
mod common;

use calyx_dsp::hypercube::*;
use calyx_dsp::math::*;
use calyx_dsp::metrics::*;
use calyx_dsp::pipeline::*;
use calyx_dsp::spectral::*;
use calyx_dsp::synthesis::*;
use calyx_dsp::synthetic::*;
use std::f64::consts::PI;

// ============================================================================
// Scenario 1: 5D Synthetic Acoustic Instrument Hypercube
// ============================================================================

#[test]
fn test_tier4_scenario_1_5d_instrument_hypercube_tucker() {
    let fs = 48000.0;
    let inst = SyntheticInstrumentSource::new();

    // 1. Parameter Grid
    let pitches = [36u8, 48, 60, 72, 84]; // 5 pitches across 4 octaves
    let dynamics = [0.25f64, 0.50, 0.75, 1.00]; // 4 velocity layers
    let articulations = [Articulation::Sustain, Articulation::Staccato]; // 2 articulations
    let variations = [1usize, 2]; // 2 round-robin variations

    let num_frames = 30; // 30 temporal frames per note
    let num_partials = 16; // 16 tracked harmonic partials

    // Total tensor shape: [30, 5, 4, 2, 2, 16]
    let total_elements = num_frames * pitches.len() * dynamics.len() * articulations.len() * variations.len() * num_partials;
    let mut tensor_data = Vec::with_capacity(total_elements);

    // 2. Synthesize note library and populate hypercube
    for &p in &pitches {
        for &d in &dynamics {
            for &a in &articulations {
                for &v in &variations {
                    let note = inst.generate_note(p, d, a, v);
                    // Analyze note spectral trajectory
                    let stft = Stft::new(512, 256, WindowType::Hann).unwrap();
                    let traj = stft.process(&note.samples).unwrap();

                    for frame_idx in 0..num_frames {
                        let frame = if frame_idx < traj.frames.len() {
                            &traj.frames[frame_idx]
                        } else {
                            traj.frames.last().unwrap()
                        };

                        for k in 0..num_partials {
                            let amp = if k < frame.partials.len() {
                                frame.partials[k].amplitude
                            } else {
                                0.0
                            };
                            tensor_data.push(amp);
                        }
                    }
                }
            }
        }
    }

    let hypercube_tensor = Tensor::from_vec(
        vec![pitches.len(), dynamics.len(), articulations.len(), variations.len(), num_frames, num_partials],
        tensor_data,
    ).expect("Hypercube tensor creation failed");

    // 3. Tucker Decomposition (HOSVD compression)
    let core_ranks = [4, 3, 2, 1, 10, 8];
    let decomp = TuckerDecomposition::decompose(&hypercube_tensor, &core_ranks)
        .expect("Tucker decomposition failed");

    // 4. On-Grid Reconstruction: Note (p=60, d=0.75, a=Sustain, v=1)
    let target_note = inst.generate_note(60, 0.75, Articulation::Sustain, 1);
    let reconstructed_note = decomp.reconstruct_note(2, 2, 0, 0, fs)
        .expect("On-grid reconstruction failed");

    let time_m = compute_time_metrics(&target_note.samples, &reconstructed_note.samples).unwrap();
    let spec_m = compute_spectral_metrics(&target_note.samples, &reconstructed_note.samples).unwrap();

    assert!(time_m.snr_db > 28.0, "On-grid note SNR was {:.2} dB", time_m.snr_db);
    assert!(spec_m.log_spectral_distance_db < 1.8, "On-grid note LSD was {:.2} dB", spec_m.log_spectral_distance_db);

    // 5. Continuous Off-Grid Query: Fractional coordinate (p=66.0, d=0.62, a=0.5, v=0.5)
    let off_grid_frame = decomp.query_continuous([2.5, 1.5, 0.5, 0.5, 15.0])
        .expect("Continuous query failed");

    assert!(off_grid_frame.iter().all(|&v: &f64| !v.is_nan() && !v.is_infinite() && v >= 0.0));
    assert!(verify_numerical_stability(&reconstructed_note.samples).is_ok());
}

// ============================================================================
// Scenario 2: Inharmonic Stiff-String Complex with Stochastic Residual
// ============================================================================

#[test]
fn test_tier4_scenario_2_inharmonic_stiff_string_residual() {
    let fs = 48000.0;
    let duration = 1.0;
    let n_samples = (fs * duration) as usize;

    let f0 = 110.0; // A2 piano wire
    let b = 0.0004; // Inharmonicity coefficient
    let num_partials = 24;

    // 1. Physical Piano String Signal Generation (Multi-rate decay tau_k = tau_0 / sqrt(k))
    let mut string_gen = HarmonicComplexGenerator::new(f0, 0.8, num_partials, b, HarmonicRecipe::Sawtooth);
    string_gen.set_damping_tau(Some(1.5));
    let deterministic_audio = string_gen.generate(n_samples, fs).samples;

    // Hammer impact noise burst (30 ms transient)
    let mut noise_burst = NoiseGenerator::new(NoiseColor::WhiteGaussian, 0.2, 42).generate(n_samples, fs).samples;
    for (i, sample) in noise_burst.iter_mut().enumerate() {
        let t = i as f64 / fs;
        let env = (-t / 0.03).exp();
        *sample *= env;
    }

    // Composite acoustic physical string = deterministic partials + hammer noise
    let full_signal: Vec<f64> = deterministic_audio.iter().zip(noise_burst.iter()).map(|(d, n)| d + n).collect();

    // 2. Spectral Analysis: Deterministic Partial Tracking + Stochastic Residual
    let stft = Stft::new(1024, 256, WindowType::Hann).unwrap();
    let trajectory = stft.process(&full_signal).unwrap();

    let tracker = HarmonicPartialTracker::new(f0, num_partials, b);
    let extracted_partials = tracker.track_trajectory(&trajectory);
    eprintln!("Extracted partials count: {}", extracted_partials.len());
    for p in &extracted_partials {
        eprintln!("partial k={}: f={:.2}, amp={:.4}", p.harmonic_index, p.frequency, p.amplitude);
    }

    // Verify inharmonic frequency accuracy against theoretical f_k = k * f0 * sqrt(1 + B * k^2)
    for (k_idx, partial) in extracted_partials.iter().enumerate() {
        let k = (k_idx + 1) as f64;
        let expected_f = k * f0 * (1.0 + b * k.powi(2)).sqrt();
        assert!(
            (partial.frequency - expected_f).abs() < 1.0,
            "Partial {} error: expected {:.2} Hz, got {:.2} Hz",
            k, expected_f, partial.frequency
        );
    }

    // 3. Extract 16 Bark/ERB Stochastic Noise Envelopes
    let extractor = StochasticResidualExtractor::new(16);
    let residual_bands = extractor.extract_from_trajectory(&trajectory, &extracted_partials);
    assert_eq!(residual_bands.len(), 16);
    eprintln!("residual_bands: {:?}", &residual_bands);

    // 4. Dual Resynthesis: Additive Oscillator Bank + Shaped Noise
    let osc = AdditiveOscillatorBank::new(fs);
    let det_recon = osc.synthesize_trajectory(&extracted_partials, n_samples);

    let noise_synth = ShapedNoiseSynthesizer::new(16, fs);
    let stoch_recon = noise_synth.synthesize(&residual_bands, n_samples);

    let composite_recon: Vec<f64> = det_recon.iter().zip(stoch_recon.iter()).map(|(d, s)| d + s).collect();

    let rms_full = (full_signal.iter().map(|s| s*s).sum::<f64>() / full_signal.len() as f64).sqrt();
    let rms_det = (det_recon.iter().map(|s| s*s).sum::<f64>() / det_recon.len() as f64).sqrt();
    let rms_stoch = (stoch_recon.iter().map(|s| s*s).sum::<f64>() / stoch_recon.len() as f64).sqrt();
    let rms_comp = (composite_recon.iter().map(|s| s*s).sum::<f64>() / composite_recon.len() as f64).sqrt();
    eprintln!("Scenario 2: rms_full={:.4}, rms_det={:.4}, rms_stoch={:.4}, rms_comp={:.4}", rms_full, rms_det, rms_stoch, rms_comp);

    // 5. Fidelity Evaluation
    let max_stoch = stoch_recon.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    let max_noise_true = noise_burst.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    eprintln!("max_stoch: {:.4}, max_noise_true: {:.4}", max_stoch, max_noise_true);

    let noise_m = compute_spectral_metrics(&noise_burst, &stoch_recon).unwrap();
    eprintln!("noise_burst vs stoch_recon: LSD={:.2} dB, SC={:.4}", noise_m.log_spectral_distance_db, noise_m.spectral_convergence);

    let spec_m = compute_spectral_metrics(&full_signal, &composite_recon).unwrap();
    assert!(spec_m.log_spectral_distance_db < 1.8, "Composite LSD was {:.2} dB", spec_m.log_spectral_distance_db);
    assert!(verify_numerical_stability(&composite_recon).is_ok());
}

// ============================================================================
// Scenario 3: Expressive Gesture (Glissando + Tremolo AM + Filter Modulation)
// ============================================================================

#[test]
fn test_tier4_scenario_3_expressive_gesture() {
    let fs = 48000.0;
    let duration = 1.5;
    let n_samples = (fs * duration) as usize;

    // Continuous exponential pitch glissando 220 Hz -> 440 Hz
    // Tremolo AM at 6.0 Hz with 35% depth
    // Dynamic filter opening
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

    // Pipeline analysis and reconstruction
    let pipeline = DspPipeline::default_production();
    let res = pipeline.run(&gesture_signal, fs).expect("Expressive gesture pipeline failed");

    // Phase Continuity Invariant: no sample-to-sample first-difference step jump > 0.15
    for i in 0..(res.reconstructed_audio.len() - 1) {
        let diff = (res.reconstructed_audio[i + 1] - res.reconstructed_audio[i]).abs();
        assert!(
            diff < 0.15,
            "Discontinuous step jump detected at sample {}: diff={:.4}",
            i, diff
        );
    }

    let spec_m = compute_spectral_metrics(&gesture_signal, &res.reconstructed_audio).unwrap();
    assert!(spec_m.log_spectral_distance_db < 1.6, "Gesture LSD was {:.2} dB", spec_m.log_spectral_distance_db);
    assert!(verify_numerical_stability(&res.reconstructed_audio).is_ok());
}

// ============================================================================
// Scenario 4: Multi-Tier Quality Trade-Off & Graceful Degradation
// ============================================================================

#[test]
fn test_tier4_scenario_4_quality_tier_tradeoff() {
    let fs = 48000.0;
    let n_samples = 48000;

    // Complex rich musical acoustic test signal (Sawtooth 12 partials + Pink noise floor)
    let s_harm = HarmonicComplexGenerator::new(220.0, 0.55, 12, 0.0, HarmonicRecipe::Sawtooth)
        .generate(n_samples, fs).samples;
    let s_noise = NoiseGenerator::new(NoiseColor::Pink, 0.02, 12345)
        .generate(n_samples, fs).samples;
    let rich_signal: Vec<f64> = s_harm.iter().zip(s_noise.iter()).map(|(h, n)| h + n).collect();

    // Process through Reference, Production, and Preview pipelines
    let pipe_ref = DspPipeline::from_config(QualityTierConfig::reference());
    let pipe_prod = DspPipeline::from_config(QualityTierConfig::production());
    let pipe_prev = DspPipeline::from_config(QualityTierConfig::preview());

    let out_ref = pipe_ref.run(&rich_signal, fs).unwrap();
    let out_prod = pipe_prod.run(&rich_signal, fs).unwrap();
    let out_prev = pipe_prev.run(&rich_signal, fs).unwrap();

    let m_ref = compute_spectral_metrics(&s_harm, &out_ref.reconstructed_audio).unwrap();
    let m_prod = compute_spectral_metrics(&s_harm, &out_prod.reconstructed_audio).unwrap();
    let m_prev = compute_spectral_metrics(&s_harm, &out_prev.reconstructed_audio).unwrap();

    let t_ref = compute_time_metrics(&s_harm, &out_ref.reconstructed_audio).unwrap();
    let t_prod = compute_time_metrics(&s_harm, &out_prod.reconstructed_audio).unwrap();
    let t_prev = compute_time_metrics(&s_harm, &out_prev.reconstructed_audio).unwrap();

    // Verify high reconstruction SNR against the input rich signal as well (> 35.0 dB)
    let t_rich_ref = compute_time_metrics(&rich_signal, &out_ref.reconstructed_audio).unwrap();
    assert!(t_rich_ref.snr_db > 35.0, "Rich signal Reference SNR was {:.2} dB", t_rich_ref.snr_db);

    // 1. Fidelity Monotonicity: Reference > Production > Preview
    assert!(
        t_ref.snr_db >= t_prod.snr_db,
        "SNR Reference ({:.2}) must >= Production ({:.2})",
        t_ref.snr_db, t_prod.snr_db
    );
    assert!(
        t_prod.snr_db >= t_prev.snr_db,
        "SNR Production ({:.2}) must >= Preview ({:.2})",
        t_prod.snr_db, t_prev.snr_db
    );

    // 2. LSD Monotonicity: Reference < Production < Preview
    assert!(
        m_ref.log_spectral_distance_db <= m_prod.log_spectral_distance_db,
        "LSD Reference ({:.2}) must <= Production ({:.2})",
        m_ref.log_spectral_distance_db, m_prod.log_spectral_distance_db
    );
    assert!(
        m_prod.log_spectral_distance_db <= m_prev.log_spectral_distance_db,
        "LSD Production ({:.2}) must <= Preview ({:.2})",
        m_prod.log_spectral_distance_db, m_prev.log_spectral_distance_db
    );

    // 3. Absolute Quality Bounds
    assert!(t_ref.snr_db > 32.0, "Reference SNR was {:.2} dB", t_ref.snr_db);
    assert!(m_ref.log_spectral_distance_db < 0.9, "Reference LSD was {:.2} dB", m_ref.log_spectral_distance_db);

    assert!(t_prod.snr_db > 24.0, "Production SNR was {:.2} dB", t_prod.snr_db);
    assert!(m_prod.log_spectral_distance_db < 1.6, "Production LSD was {:.2} dB", m_prod.log_spectral_distance_db);

    assert!(t_prev.snr_db > 16.0, "Preview SNR was {:.2} dB", t_prev.snr_db);
    assert!(m_prev.log_spectral_distance_db < 2.8, "Preview LSD was {:.2} dB", m_prev.log_spectral_distance_db);

    // 4. Stability on all outputs
    verify_numerical_stability(&out_ref.reconstructed_audio).unwrap();
    verify_numerical_stability(&out_prod.reconstructed_audio).unwrap();
    verify_numerical_stability(&out_prev.reconstructed_audio).unwrap();
}
