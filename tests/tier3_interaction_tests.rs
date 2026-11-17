//! tests/tier3_interaction_tests.rs
//! Opaque-box Tier 3 Pairwise Combinatorial Subsystem Interaction Suite for Calyx DSP.
//!
//! Validates the full signal lifecycle:
//! Synthetic Generation -> Spectral Analysis -> Hypercube Decomposition -> Resynthesis -> Fidelity Assessment
//!
//! Executes the 16 Canonical Orthogonal Covering Array configurations (T3.01 to T3.16)
//! covering all pairwise parameter interactions between:
//! - Sample Rates: 44.1 kHz, 48.0 kHz
//! - Window Sizes: 512, 1024
//! - Hop Ratios: N/4 (75% overlap), N/2 (50% overlap)
//! - Decomposition Algorithms: PCA, NMF, Tucker
//! - Decomposition Ranks: 2, 4, 8
//! - Signal Complexities: Pure Sine, Sawtooth Harmonic, Inharmonic Stiff String, Linear Chirp

#[path = "common/mod.rs"]
mod common;

use calyx_dsp::error::*;
use calyx_dsp::hypercube::*;
use calyx_dsp::math::*;
use calyx_dsp::metrics::*;
use calyx_dsp::spectral::*;
use calyx_dsp::synthesis::*;
use calyx_dsp::synthetic::*;

/// Helper running an end-to-end analysis, decomposition, and resynthesis chain.
fn run_interaction_pipeline(
    input_signal: &[f64],
    sample_rate: f64,
    window_size: usize,
    hop_size: usize,
    decomposition_kind: &str,
    rank: usize,
) -> DspResult<(Vec<f64>, TimeFidelityMetrics, SpectralFidelityMetrics)> {
    // Stage 1: STFT Analysis
    let stft = Stft::new(window_size, hop_size, WindowType::Hann)?.with_sample_rate(sample_rate);
    let trajectory = stft.process(input_signal)?;

    // Stage 2: Decomposition (PCA, NMF, or Tucker)
    let reconstructed_trajectory = match decomposition_kind {
        "PCA" => {
            let pca = PcaDecomposer::new(rank);
            let compressed = pca.decompose(&trajectory)?;
            pca.reconstruct(&compressed)?
        }
        "NMF" => {
            let nmf = NmfDecomposer::new(rank, 40);
            let compressed = nmf.decompose(&trajectory)?;
            nmf.reconstruct(&compressed)?
        }
        "Tucker" => {
            let tucker = TuckerDecomposer::new(&[rank, rank.min(trajectory.frames[0].partials.len().max(1))]);
            let compressed = tucker.decompose(&trajectory)?;
            tucker.reconstruct(&compressed)?
        }
        _ => return Err(DspError::invalid_parameter("decomposition_kind", "unknown algorithm")),
    };

    // Stage 3: Resynthesis
    let reconstructor = AdditiveOlaReconstructor::new(window_size, hop_size);
    let output_audio = reconstructor.reconstruct(&reconstructed_trajectory, input_signal.len(), sample_rate)?;

    let time_metrics = compute_time_metrics(input_signal, &output_audio)?;
    let spectral_metrics = compute_spectral_metrics(input_signal, &output_audio)?;

    // Verify stability invariants on output
    if input_signal.iter().all(|&s| s.abs() <= 1.0001) {
        verify_numerical_stability(&output_audio)?;
    }

    let rms_in = (input_signal.iter().map(|s| s*s).sum::<f64>() / input_signal.len() as f64).sqrt();
    let rms_out = (output_audio.iter().map(|s| s*s).sum::<f64>() / output_audio.len() as f64).sqrt();
    eprintln!(
        "[{}] SNR: {:.2} dB, LSD: {:.2} dB, SC: {:.4}, rms_in={:.4}, rms_out={:.4}",
        decomposition_kind, time_metrics.snr_db, spectral_metrics.log_spectral_distance_db, spectral_metrics.spectral_convergence,
        rms_in, rms_out
    );

    Ok((output_audio, time_metrics, spectral_metrics))
}

// ============================================================================
// Canonical 16 Orthogonal Covering Array Tests (T3.01 .. T3.16)
// ============================================================================

#[test]
fn test_t3_01_sr441_w512_h128_pca_r2_sine() {
    let fs = 44100.0;
    let input = SineGenerator::new(440.0, 0.8, 0.0).generate((fs * 0.5) as usize, fs).samples;
    let (_out, time_m, spec_m) = run_interaction_pipeline(&input, fs, 512, 128, "PCA", 2).unwrap();

    assert!(time_m.snr_db > 35.0, "SNR was {} dB", time_m.snr_db);
    assert!(spec_m.log_spectral_distance_db < 1.0, "LSD was {} dB", spec_m.log_spectral_distance_db);
    assert!(spec_m.spectral_convergence < 0.05);
}

#[test]
fn test_t3_02_sr441_w1024_h512_nmf_r4_sawtooth() {
    let fs = 44100.0;
    let input = HarmonicComplexGenerator::new(220.0, 0.8, 10, 0.0, HarmonicRecipe::Sawtooth)
        .generate((fs * 0.5) as usize, fs)
        .samples;
    let (_out, time_m, spec_m) = run_interaction_pipeline(&input, fs, 1024, 512, "NMF", 4).unwrap();

    assert!(time_m.snr_db > 25.0, "SNR was {} dB", time_m.snr_db);
    assert!(spec_m.log_spectral_distance_db < 1.8, "LSD was {} dB", spec_m.log_spectral_distance_db);
}

#[test]
fn test_t3_03_sr441_w512_h256_tucker_r8_inharmonic() {
    let fs = 44100.0;
    let input = HarmonicComplexGenerator::new(110.0, 0.8, 12, 0.0004, HarmonicRecipe::Sawtooth)
        .generate((fs * 0.5) as usize, fs)
        .samples;
    let (_out, time_m, spec_m) = run_interaction_pipeline(&input, fs, 512, 256, "Tucker", 8).unwrap();

    assert!(time_m.snr_db > 24.0, "SNR was {} dB", time_m.snr_db);
    assert!(spec_m.log_spectral_distance_db < 2.0, "LSD was {} dB", spec_m.log_spectral_distance_db);
}

#[test]
fn test_t3_04_sr441_w1024_h256_pca_r4_chirp() {
    let fs = 44100.0;
    let input = ChirpGenerator::new(100.0, 3000.0, 0.8, 0.5, ChirpKind::Linear)
        .generate((fs * 0.5) as usize, fs)
        .samples;
    let (_out, _time_m, spec_m) = run_interaction_pipeline(&input, fs, 1024, 256, "PCA", 4).unwrap();

    assert!(spec_m.log_spectral_distance_db < 2.2, "LSD was {} dB", spec_m.log_spectral_distance_db);
    assert!(spec_m.spectral_convergence < 0.15);
}

#[test]
fn test_t3_05_sr48_w1024_h256_tucker_r2_sine() {
    let fs = 48000.0;
    let input = SineGenerator::new(440.0, 0.8, 0.0).generate((fs * 0.5) as usize, fs).samples;
    let (_out, time_m, spec_m) = run_interaction_pipeline(&input, fs, 1024, 256, "Tucker", 2).unwrap();

    assert!(time_m.snr_db > 35.0, "SNR was {} dB", time_m.snr_db);
    assert!(spec_m.log_spectral_distance_db < 0.8);
}

#[test]
fn test_t3_06_sr48_w512_h256_pca_r8_sawtooth() {
    let fs = 48000.0;
    let input = HarmonicComplexGenerator::new(220.0, 0.8, 10, 0.0, HarmonicRecipe::Sawtooth)
        .generate((fs * 0.5) as usize, fs)
        .samples;
    let (_out, time_m, spec_m) = run_interaction_pipeline(&input, fs, 512, 256, "PCA", 8).unwrap();

    assert!(time_m.snr_db > 30.0, "SNR was {} dB", time_m.snr_db);
    assert!(spec_m.log_spectral_distance_db < 1.2);
}

#[test]
fn test_t3_07_sr48_w1024_h512_nmf_r2_inharmonic() {
    let fs = 48000.0;
    let input = HarmonicComplexGenerator::new(110.0, 0.8, 12, 0.0004, HarmonicRecipe::Sawtooth)
        .generate((fs * 0.5) as usize, fs)
        .samples;
    let (_out, _time_m, spec_m) = run_interaction_pipeline(&input, fs, 1024, 512, "NMF", 2).unwrap();

    assert!(spec_m.log_spectral_distance_db < 2.5);
}

#[test]
fn test_t3_08_sr48_w512_h128_tucker_r4_chirp() {
    let fs = 48000.0;
    let input = ChirpGenerator::new(100.0, 3000.0, 0.8, 0.5, ChirpKind::Linear)
        .generate((fs * 0.5) as usize, fs)
        .samples;
    let (_out, _time_m, spec_m) = run_interaction_pipeline(&input, fs, 512, 128, "Tucker", 4).unwrap();

    assert!(spec_m.log_spectral_distance_db < 2.2);
}

#[test]
fn test_t3_09_sr441_w1024_h256_nmf_r8_sine() {
    let fs = 44100.0;
    let input = SineGenerator::new(440.0, 0.8, 0.0).generate((fs * 0.5) as usize, fs).samples;
    let (_out, time_m, spec_m) = run_interaction_pipeline(&input, fs, 1024, 256, "NMF", 8).unwrap();

    assert!(time_m.snr_db > 30.0);
    assert!(spec_m.log_spectral_distance_db < 1.0);
}

#[test]
fn test_t3_10_sr441_w512_h256_pca_r4_inharmonic() {
    let fs = 44100.0;
    let input = HarmonicComplexGenerator::new(110.0, 0.8, 10, 0.0004, HarmonicRecipe::Sawtooth)
        .generate((fs * 0.5) as usize, fs)
        .samples;
    let (_out, time_m, spec_m) = run_interaction_pipeline(&input, fs, 512, 256, "PCA", 4).unwrap();

    assert!(time_m.snr_db > 24.0);
    assert!(spec_m.log_spectral_distance_db < 2.0);
}

#[test]
fn test_t3_11_sr48_w512_h128_nmf_r4_sine() {
    let fs = 48000.0;
    let input = SineGenerator::new(440.0, 0.8, 0.0).generate((fs * 0.5) as usize, fs).samples;
    let (_out, time_m, spec_m) = run_interaction_pipeline(&input, fs, 512, 128, "NMF", 4).unwrap();

    assert!(time_m.snr_db > 30.0);
    assert!(spec_m.log_spectral_distance_db < 1.0);
}

#[test]
fn test_t3_12_sr48_w1024_h256_pca_r2_sawtooth() {
    let fs = 48000.0;
    let input = HarmonicComplexGenerator::new(220.0, 0.8, 10, 0.0, HarmonicRecipe::Sawtooth)
        .generate((fs * 0.5) as usize, fs)
        .samples;
    let (_out, time_m, spec_m) = run_interaction_pipeline(&input, fs, 1024, 256, "PCA", 2).unwrap();

    assert!(time_m.snr_db > 20.0);
    assert!(spec_m.log_spectral_distance_db < 2.5);
}

#[test]
fn test_t3_13_sr441_w512_h128_tucker_r4_sawtooth() {
    let fs = 44100.0;
    let input = HarmonicComplexGenerator::new(220.0, 0.8, 8, 0.0, HarmonicRecipe::Sawtooth)
        .generate((fs * 0.5) as usize, fs)
        .samples;
    let (_out, time_m, spec_m) = run_interaction_pipeline(&input, fs, 512, 128, "Tucker", 4).unwrap();

    assert!(time_m.snr_db > 24.0);
    assert!(spec_m.log_spectral_distance_db < 2.0);
}

#[test]
fn test_t3_14_sr48_w1024_h512_tucker_r8_inharmonic() {
    let fs = 48000.0;
    let input = HarmonicComplexGenerator::new(110.0, 0.8, 12, 0.0004, HarmonicRecipe::Sawtooth)
        .generate((fs * 0.5) as usize, fs)
        .samples;
    let (_out, time_m, spec_m) = run_interaction_pipeline(&input, fs, 1024, 512, "Tucker", 8).unwrap();

    assert!(time_m.snr_db > 24.0);
    assert!(spec_m.log_spectral_distance_db < 1.8);
}

#[test]
fn test_t3_15_sr441_w1024_h512_pca_r8_chirp() {
    let fs = 44100.0;
    let input = ChirpGenerator::new(100.0, 3000.0, 0.8, 0.5, ChirpKind::Linear)
        .generate((fs * 0.5) as usize, fs)
        .samples;
    let (_out, _time_m, spec_m) = run_interaction_pipeline(&input, fs, 1024, 512, "PCA", 8).unwrap();

    assert!(spec_m.log_spectral_distance_db < 2.0);
    assert!(spec_m.spectral_convergence < 0.12);
}

#[test]
fn test_t3_16_sr48_w512_h256_nmf_r8_chirp() {
    let fs = 48000.0;
    let input = ChirpGenerator::new(100.0, 3000.0, 0.8, 0.5, ChirpKind::Linear)
        .generate((fs * 0.5) as usize, fs)
        .samples;
    let (_out, _time_m, spec_m) = run_interaction_pipeline(&input, fs, 512, 256, "NMF", 8).unwrap();

    assert!(spec_m.log_spectral_distance_db < 2.2);
}
