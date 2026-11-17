//! tests/adversarial_challenger2_fidelity_challenge.rs
//! Adversarial Verification Harness for Milestone 4 Reconstruction Fidelity & Quality Presets.
//! Authored by Challenger 2 (Milestone 4 Fidelity & Presets Stress Verifier).

use calyx_dsp::metrics::*;
use calyx_dsp::pipeline::*;
use calyx_dsp::synthetic::*;

#[test]
fn test_scenario_4_empirical_metrics_and_strict_monotonicity() {
    let fs = 48000.0;
    let n_samples = 48000;

    let s_harm = HarmonicComplexGenerator::new(220.0, 0.55, 12, 0.0, HarmonicRecipe::Sawtooth)
        .generate(n_samples, fs).samples;
    let s_noise = NoiseGenerator::new(NoiseColor::Pink, 0.02, 12345)
        .generate(n_samples, fs).samples;
    let rich_signal: Vec<f64> = s_harm.iter().zip(s_noise.iter()).map(|(h, n)| h + n).collect();

    let pipe_ref = DspPipeline::from_config(QualityTierConfig::reference());
    let pipe_prod = DspPipeline::from_config(QualityTierConfig::production());
    let pipe_prev = DspPipeline::from_config(QualityTierConfig::preview());

    let out_ref = pipe_ref.run(&rich_signal, fs).expect("Reference pipeline failed");
    let out_prod = pipe_prod.run(&rich_signal, fs).expect("Production pipeline failed");
    let out_prev = pipe_prev.run(&rich_signal, fs).expect("Preview pipeline failed");

    // Metrics against harmonic ground truth s_harm
    let t_ref_harm = compute_time_metrics(&s_harm, &out_ref.reconstructed_audio).unwrap();
    let m_ref_harm = compute_spectral_metrics(&s_harm, &out_ref.reconstructed_audio).unwrap();

    let t_prod_harm = compute_time_metrics(&s_harm, &out_prod.reconstructed_audio).unwrap();
    let m_prod_harm = compute_spectral_metrics(&s_harm, &out_prod.reconstructed_audio).unwrap();

    let t_prev_harm = compute_time_metrics(&s_harm, &out_prev.reconstructed_audio).unwrap();
    let m_prev_harm = compute_spectral_metrics(&s_harm, &out_prev.reconstructed_audio).unwrap();

    // Metrics against noisy input rich_signal
    let t_ref_rich = compute_time_metrics(&rich_signal, &out_ref.reconstructed_audio).unwrap();
    let m_ref_rich = compute_spectral_metrics(&rich_signal, &out_ref.reconstructed_audio).unwrap();

    let t_prod_rich = compute_time_metrics(&rich_signal, &out_prod.reconstructed_audio).unwrap();
    let m_prod_rich = compute_spectral_metrics(&rich_signal, &out_prod.reconstructed_audio).unwrap();

    let t_prev_rich = compute_time_metrics(&rich_signal, &out_prev.reconstructed_audio).unwrap();
    let m_prev_rich = compute_spectral_metrics(&rich_signal, &out_prev.reconstructed_audio).unwrap();

    eprintln!("\n=== SCENARIO 4 FIDELITY METRICS (VS S_HARM GROUND TRUTH) ===");
    eprintln!("Reference Tier:  SNR = {:.4} dB, LSD = {:.4} dB, SC = {:.6}",
        t_ref_harm.snr_db, m_ref_harm.log_spectral_distance_db, m_ref_harm.spectral_convergence);
    eprintln!("Production Tier: SNR = {:.4} dB, LSD = {:.4} dB, SC = {:.6}",
        t_prod_harm.snr_db, m_prod_harm.log_spectral_distance_db, m_prod_harm.spectral_convergence);
    eprintln!("Preview Tier:    SNR = {:.4} dB, LSD = {:.4} dB, SC = {:.6}",
        t_prev_harm.snr_db, m_prev_harm.log_spectral_distance_db, m_prev_harm.spectral_convergence);

    eprintln!("\n=== SCENARIO 4 METRICS (VS RICH_SIGNAL INPUT) ===");
    eprintln!("Reference Tier:  SNR = {:.4} dB, LSD = {:.4} dB, SC = {:.6}",
        t_ref_rich.snr_db, m_ref_rich.log_spectral_distance_db, m_ref_rich.spectral_convergence);
    eprintln!("Production Tier: SNR = {:.4} dB, LSD = {:.4} dB, SC = {:.6}",
        t_prod_rich.snr_db, m_prod_rich.log_spectral_distance_db, m_prod_rich.spectral_convergence);
    eprintln!("Preview Tier:    SNR = {:.4} dB, LSD = {:.4} dB, SC = {:.6}",
        t_prev_rich.snr_db, m_prev_rich.log_spectral_distance_db, m_prev_rich.spectral_convergence);

    // 1. Verify Absolute Thresholds against s_harm
    assert!(t_ref_harm.snr_db > 32.0, "Reference SNR must exceed 32.0 dB, got {:.2}", t_ref_harm.snr_db);
    assert!(m_ref_harm.log_spectral_distance_db < 0.9, "Reference LSD must be below 0.9 dB, got {:.4}", m_ref_harm.log_spectral_distance_db);

    assert!(t_prod_harm.snr_db > 24.0, "Production SNR must exceed 24.0 dB, got {:.2}", t_prod_harm.snr_db);
    assert!(m_prod_harm.log_spectral_distance_db < 1.6, "Production LSD must be below 1.6 dB, got {:.4}", m_prod_harm.log_spectral_distance_db);

    assert!(t_prev_harm.snr_db > 16.0, "Preview SNR must exceed 16.0 dB, got {:.2}", t_prev_harm.snr_db);
    assert!(m_prev_harm.log_spectral_distance_db < 2.8, "Preview LSD must be below 2.8 dB, got {:.4}", m_prev_harm.log_spectral_distance_db);

    // 2. Verify Strict Monotonicity against harmonic ground truth
    assert!(
        t_ref_harm.snr_db > t_prod_harm.snr_db,
        "Strict SNR Monotonicity failed: Reference ({:.4}) <= Production ({:.4})",
        t_ref_harm.snr_db, t_prod_harm.snr_db
    );
    assert!(
        t_prod_harm.snr_db > t_prev_harm.snr_db,
        "Strict SNR Monotonicity failed: Production ({:.4}) <= Preview ({:.4})",
        t_prod_harm.snr_db, t_prev_harm.snr_db
    );

    assert!(
        m_ref_harm.log_spectral_distance_db < m_prod_harm.log_spectral_distance_db,
        "Strict LSD Monotonicity failed: Reference ({:.4}) >= Production ({:.4})",
        m_ref_harm.log_spectral_distance_db, m_prod_harm.log_spectral_distance_db
    );
    assert!(
        m_prod_harm.log_spectral_distance_db < m_prev_harm.log_spectral_distance_db,
        "Strict LSD Monotonicity failed: Production ({:.4}) >= Preview ({:.4})",
        m_prod_harm.log_spectral_distance_db, m_prev_harm.log_spectral_distance_db
    );

    // 3. Rich signal baseline SNR
    assert!(t_ref_rich.snr_db > 35.0, "Reference SNR vs rich signal must exceed 35.0 dB, got {:.2}", t_ref_rich.snr_db);
}

#[test]
fn test_quality_tiers_on_clean_harmonic_signal() {
    let fs = 48000.0;
    let n_samples = 24000; // 0.5s

    let clean_saw = HarmonicComplexGenerator::new(220.0, 0.50, 12, 0.0, HarmonicRecipe::Sawtooth)
        .generate(n_samples, fs).samples;

    let in_peak = clean_saw.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    eprintln!("clean_saw in_peak = {:.4}", in_peak);

    let pipe_ref = DspPipeline::from_config(QualityTierConfig::reference());
    let pipe_prod = DspPipeline::from_config(QualityTierConfig::production());
    let pipe_prev = DspPipeline::from_config(QualityTierConfig::preview());

    let out_ref = pipe_ref.run(&clean_saw, fs).unwrap();
    let out_prod = pipe_prod.run(&clean_saw, fs).unwrap();
    let out_prev = pipe_prev.run(&clean_saw, fs).unwrap();

    let out_peak_ref = out_ref.reconstructed_audio.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    let out_peak_prod = out_prod.reconstructed_audio.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    let out_peak_prev = out_prev.reconstructed_audio.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    eprintln!("out_peaks: ref={:.4}, prod={:.4}, prev={:.4}", out_peak_ref, out_peak_prod, out_peak_prev);

    let t_ref = compute_time_metrics(&clean_saw, &out_ref.reconstructed_audio).unwrap();
    let m_ref = compute_spectral_metrics(&clean_saw, &out_ref.reconstructed_audio).unwrap();

    let t_prod = compute_time_metrics(&clean_saw, &out_prod.reconstructed_audio).unwrap();
    let m_prod = compute_spectral_metrics(&clean_saw, &out_prod.reconstructed_audio).unwrap();

    let t_prev = compute_time_metrics(&clean_saw, &out_prev.reconstructed_audio).unwrap();
    let m_prev = compute_spectral_metrics(&clean_saw, &out_prev.reconstructed_audio).unwrap();

    eprintln!("\n=== CLEAN HARMONIC SIGNAL (NO NOISE) ===");
    eprintln!("Reference:  SNR = {:.2} dB, LSD = {:.4} dB", t_ref.snr_db, m_ref.log_spectral_distance_db);
    eprintln!("Production: SNR = {:.2} dB, LSD = {:.4} dB", t_prod.snr_db, m_prod.log_spectral_distance_db);
    eprintln!("Preview:    SNR = {:.2} dB, LSD = {:.4} dB", t_prev.snr_db, m_prev.log_spectral_distance_db);

    assert!(t_ref.snr_db > 35.0);
    assert!(t_prod.snr_db > 30.0);
    assert!(t_prev.snr_db > 25.0);

    verify_numerical_stability(&out_ref.reconstructed_audio).unwrap();
    verify_numerical_stability(&out_prod.reconstructed_audio).unwrap();
    verify_numerical_stability(&out_prev.reconstructed_audio).unwrap();
}

#[test]
fn test_quality_tiers_across_different_pitches() {
    let fs = 48000.0;
    let n_samples = 24000;
    let pitches = [110.0, 220.0, 440.0, 880.0];

    for &f0 in &pitches {
        let s_harm = HarmonicComplexGenerator::new(f0, 0.45, 8, 0.0, HarmonicRecipe::Sawtooth)
            .generate(n_samples, fs).samples;

        let pipe_ref = DspPipeline::from_config(QualityTierConfig::reference());
        let pipe_prod = DspPipeline::from_config(QualityTierConfig::production());
        let pipe_prev = DspPipeline::from_config(QualityTierConfig::preview());

        let out_ref = pipe_ref.run(&s_harm, fs).unwrap();
        let out_prod = pipe_prod.run(&s_harm, fs).unwrap();
        let out_prev = pipe_prev.run(&s_harm, fs).unwrap();

        let t_ref = compute_time_metrics(&s_harm, &out_ref.reconstructed_audio).unwrap();
        let t_prod = compute_time_metrics(&s_harm, &out_prod.reconstructed_audio).unwrap();
        let t_prev = compute_time_metrics(&s_harm, &out_prev.reconstructed_audio).unwrap();

        eprintln!("Pitch f0={:.0} Hz: Ref SNR={:.2} dB, Prod SNR={:.2} dB, Prev SNR={:.2} dB",
            f0, t_ref.snr_db, t_prod.snr_db, t_prev.snr_db);

        assert!(t_ref.snr_db > 32.0, "f0={}: Ref SNR was {:.2}", f0, t_ref.snr_db);
        assert!(t_prod.snr_db > 24.0, "f0={}: Prod SNR was {:.2}", f0, t_prod.snr_db);
        assert!(t_prev.snr_db > 16.0, "f0={}: Prev SNR was {:.2}", f0, t_prev.snr_db);

        verify_numerical_stability(&out_ref.reconstructed_audio).unwrap();
        verify_numerical_stability(&out_prod.reconstructed_audio).unwrap();
        verify_numerical_stability(&out_prev.reconstructed_audio).unwrap();
    }
}

#[test]
fn test_quality_tiers_inharmonic_stiff_string() {
    let fs = 48000.0;
    let n_samples = 24000;
    let f0 = 110.0;
    let b = 0.0004;

    let inharm_string = HarmonicComplexGenerator::new(f0, 0.35, 12, b, HarmonicRecipe::Sawtooth)
        .generate(n_samples, fs).samples;

    let in_peak = inharm_string.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    eprintln!("inharm_string in_peak = {:.4}", in_peak);

    let pipe_ref = DspPipeline::from_config(QualityTierConfig::reference());
    let pipe_prod = DspPipeline::from_config(QualityTierConfig::production());
    let pipe_prev = DspPipeline::from_config(QualityTierConfig::preview());

    let out_ref = pipe_ref.run(&inharm_string, fs).unwrap();
    let out_prod = pipe_prod.run(&inharm_string, fs).unwrap();
    let out_prev = pipe_prev.run(&inharm_string, fs).unwrap();

    let out_peak_ref = out_ref.reconstructed_audio.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    let out_peak_prod = out_prod.reconstructed_audio.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    let out_peak_prev = out_prev.reconstructed_audio.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    eprintln!("inharm out_peaks: ref={:.4}, prod={:.4}, prev={:.4}", out_peak_ref, out_peak_prod, out_peak_prev);

    let t_ref = compute_time_metrics(&inharm_string, &out_ref.reconstructed_audio).unwrap();
    let m_ref = compute_spectral_metrics(&inharm_string, &out_ref.reconstructed_audio).unwrap();

    let t_prod = compute_time_metrics(&inharm_string, &out_prod.reconstructed_audio).unwrap();
    let m_prod = compute_spectral_metrics(&inharm_string, &out_prod.reconstructed_audio).unwrap();

    let t_prev = compute_time_metrics(&inharm_string, &out_prev.reconstructed_audio).unwrap();
    let m_prev = compute_spectral_metrics(&inharm_string, &out_prev.reconstructed_audio).unwrap();

    eprintln!("\n=== INHARMONIC STIFF STRING ===");
    eprintln!("Reference:  SNR = {:.2} dB, LSD = {:.4} dB", t_ref.snr_db, m_ref.log_spectral_distance_db);
    eprintln!("Production: SNR = {:.2} dB, LSD = {:.4} dB", t_prod.snr_db, m_prod.log_spectral_distance_db);
    eprintln!("Preview:    SNR = {:.2} dB, LSD = {:.4} dB", t_prev.snr_db, m_prev.log_spectral_distance_db);

    assert!(t_ref.snr_db > 32.0);
    assert!(m_ref.log_spectral_distance_db < 0.9);
    assert!(t_prod.snr_db > 24.0);
    assert!(m_prod.log_spectral_distance_db < 1.6);
    assert!(t_prev.snr_db > 16.0);
    assert!(m_prev.log_spectral_distance_db < 2.8);

    verify_numerical_stability(&out_ref.reconstructed_audio).unwrap();
    verify_numerical_stability(&out_prod.reconstructed_audio).unwrap();
    verify_numerical_stability(&out_prev.reconstructed_audio).unwrap();
}

#[test]
fn test_quality_tier_compression_data_reduction() {
    let fs = 48000.0;
    let n_samples = 24000;

    let signal = HarmonicComplexGenerator::new(220.0, 0.6, 12, 0.0, HarmonicRecipe::Sawtooth)
        .generate(n_samples, fs).samples;

    let pipe_ref = DspPipeline::from_config(QualityTierConfig::reference());
    let pipe_prod = DspPipeline::from_config(QualityTierConfig::production());
    let pipe_prev = DspPipeline::from_config(QualityTierConfig::preview());

    let res_ref = pipe_ref.run(&signal, fs).unwrap();
    let res_prod = pipe_prod.run(&signal, fs).unwrap();
    let res_prev = pipe_prev.run(&signal, fs).unwrap();

    let ref_frames = res_ref.analyzed_trajectory.as_ref().unwrap().frames.len();
    let prod_frames = res_prod.analyzed_trajectory.as_ref().unwrap().frames.len();
    let prev_frames = res_prev.analyzed_trajectory.as_ref().unwrap().frames.len();

    eprintln!("\n=== COMPRESSION / FRAME DATA RATIOS ===");
    eprintln!("Reference:  hop=256 -> frames={}, rank=8, noise_bands=16", ref_frames);
    eprintln!("Production: hop=128 -> frames={}, rank=4, noise_bands=8", prod_frames);
    eprintln!("Preview:    hop=256 -> frames={}, rank=2, noise_bands=4", prev_frames);

    // Verify config settings
    assert_eq!(pipe_ref.config.rank, 8);
    assert_eq!(pipe_ref.config.num_noise_bands, 16);
    assert_eq!(pipe_prod.config.rank, 4);
    assert_eq!(pipe_prod.config.num_noise_bands, 8);
    assert_eq!(pipe_prev.config.rank, 2);
    assert_eq!(pipe_prev.config.num_noise_bands, 4);
}
