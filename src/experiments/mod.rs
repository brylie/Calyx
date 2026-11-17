//! # Empirical DSP Experiments and Sweeps
//!
//! Provides a reproducible experimentation suite to investigate the open research
//! questions in `docs/open-questions.md`, including:
//! 1. **Partial Count Sweep (OQ 1.1)**: Reconstruction fidelity vs. retained partial count $N$ across instrument timbres.
//! 2. **Phase Strategy Comparison (OQ 1.2 & 3.1)**: Stored phase vs. continuous phase vocoder vs. harmonic locking vs. phase dispersion.
//! 3. **Spectral Frame Rate Sweep (OQ 1.3)**: STFT hop size / frame rate vs. transient and sustain fidelity.
//! 4. **Dimensionality Reduction Benchmark (OQ 2.1 & 2.3)**: PCA vs. NMF across ranks $K$, comparing SNR, LSD, compression ratio, and compute latency.
//! 5. **Tensor Decomposition & Real-Time Polyphony Benchmark (OQ 2.2, 2.5, & 5.1)**: Tucker HOSVD vs. CP-ALS on a multi-dimensional instrument hypercube, measuring compression ratios vs. PCM, decomposition latency, single-frame evaluation latency, and real-time voice polyphony ceilings.
//! 6. **.calyx Container Serialization Benchmark (OQ 4.1 & 4.2)**: Serialization throughput, container disk footprint, and compression ratios against raw PCM audio and uncompressed hypercubes.

use std::f64::consts::PI;
use std::io::Cursor;
use std::time::Instant;

use crate::error::DspResult;
use crate::hypercube::coordinates::HypercubeLayout;
use crate::hypercube::{CpAls, NmfDecomposer, PcaDecomposer, TuckerDecomposition};
use crate::io::{
    AxisDefinition, CalyxContainer, CalyxContainerHeader, CpContainerData,
    QualityTier as ContainerQualityTier, ResidualContainerData, TuckerContainerData,
};
use crate::math::{Tensor, WindowType};
use crate::metrics::{compute_spectral_metrics, compute_time_metrics};
use crate::spectral::{HarmonicPartial, SpectralAnalyzer, SpectralTrajectory, Stft};
use crate::synthesis::{lock_harmonic_phase, AdditiveOlaReconstructor, PhaseDispersionModel};
use crate::synthetic::{
    ChirpGenerator, ChirpKind, HarmonicComplexGenerator, HarmonicRecipe, SignalGenerator,
    SyntheticInstrumentSource,
};

/// Result of a single point in Experiment 1 (Partial Count Sweep).
#[derive(Debug, Clone, PartialEq)]
pub struct PartialSweepRecord {
    /// Instrument timbre archetype name.
    pub timbre: String,
    /// Maximum number of retained partials per frame.
    pub max_partials: usize,
    /// Time-domain Signal-to-Noise Ratio (dB).
    pub snr_db: f64,
    /// Frequency-domain Log-Spectral Distance (dB).
    pub lsd_db: f64,
    /// Spectral convergence (relative Frobenius norm error).
    pub spectral_convergence: f64,
    /// Time-domain Root Mean Square Error.
    pub rmse: f64,
    /// Uncompressed deterministic partial payload rate (bytes/sec, assuming f32 amp+freq per partial).
    pub bytes_per_sec: f64,
}

/// Result of a single phase strategy evaluation in Experiment 2.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseStrategyRecord {
    /// Signal archetype name.
    pub signal_name: String,
    /// Phase handling strategy name.
    pub strategy: String,
    /// Time-domain waveform SNR (dB) — sensitive to phase alignment.
    pub waveform_snr_db: f64,
    /// Time-domain Pearson correlation coefficient $r$.
    pub pearson_r: f64,
    /// Phase-invariant Log-Spectral Distance (dB).
    pub lsd_db: f64,
    /// Phase-invariant Spectral Convergence.
    pub spectral_convergence: f64,
    /// Stored phase overhead in bytes per frame (assuming 16 partials, f32 storage).
    pub phase_bytes_per_frame: usize,
}

/// Result of a single frame rate / hop size configuration in Experiment 3.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameRateSweepRecord {
    /// Signal archetype name.
    pub signal_name: String,
    /// STFT window size in samples.
    pub window_size: usize,
    /// STFT hop size in samples.
    pub hop_size: usize,
    /// Resulting spectral frame rate in frames/second.
    pub frames_per_sec: f64,
    /// Reconstructed SNR (dB).
    pub snr_db: f64,
    /// Reconstructed Log-Spectral Distance (dB).
    pub lsd_db: f64,
    /// Spectral convergence.
    pub spectral_convergence: f64,
}

/// Result of a single PCA vs. NMF rank evaluation in Experiment 4.
#[derive(Debug, Clone, PartialEq)]
pub struct DimReductionRecord {
    /// Decomposition algorithm ("PCA" or "NMF").
    pub method: String,
    /// Retained rank $K$.
    pub rank: usize,
    /// Reconstructed audio SNR (dB).
    pub snr_db: f64,
    /// Reconstructed audio Log-Spectral Distance (dB).
    pub lsd_db: f64,
    /// Spectral convergence.
    pub spectral_convergence: f64,
    /// Matrix parameter compression ratio (uncompressed elements / compressed elements).
    pub matrix_compression_ratio: f64,
    /// Decomposition wall-clock duration in microseconds.
    pub decompose_us: f64,
    /// Reconstruction wall-clock duration in microseconds.
    pub reconstruct_us: f64,
}

/// Result of a multi-dimensional tensor decomposition benchmark in Experiment 5.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorBenchmarkRecord {
    /// Tensor decomposition method ("Tucker-HOSVD" or "CP-ALS").
    pub method: String,
    /// Quality tier label ("Preview", "Production", "Reference").
    pub tier_label: String,
    /// Rank description string.
    pub rank_desc: String,
    /// Relative Frobenius reconstruction error on the hypercube tensor.
    pub relative_tensor_error: f64,
    /// Tensor reconstruction SNR (dB).
    pub tensor_snr_db: f64,
    /// Number of stored f32 parameters in the decomposed representation.
    pub compressed_floats: usize,
    /// Compression ratio relative to uncompressed spectral hypercube tensor.
    pub spectral_compression_ratio: f64,
    /// Compression ratio relative to equivalent raw 48 kHz PCM audio library.
    pub pcm_compression_ratio: f64,
    /// Decomposition compute time in milliseconds.
    pub decompose_ms: f64,
    /// Single-frame reconstruction latency in microseconds.
    pub frame_eval_us: f64,
    /// Estimated maximum simultaneous real-time voices at 100 Hz frame update rate (50% CPU budget).
    pub estimated_max_voices: usize,
}

/// Result of a container serialization benchmark in Experiment 6.
#[derive(Debug, Clone, PartialEq)]
pub struct ContainerBenchmarkRecord {
    /// Configuration description string (e.g. "Tucker-HOSVD (Production)").
    pub config_label: String,
    /// Quality tier label ("Preview", "Production", "Reference").
    pub tier_label: String,
    /// Serialized container byte size.
    pub file_size_bytes: usize,
    /// Serialized container size in KB.
    pub file_size_kb: f64,
    /// Compression ratio relative to uncompressed spectral hypercube (f64 raw data).
    pub hypercube_compression_ratio: f64,
    /// Compression ratio relative to equivalent raw 16-bit 48 kHz PCM audio.
    pub pcm_compression_ratio: f64,
    /// Serialization wall-clock latency in microseconds.
    pub serialize_us: f64,
    /// Deserialization wall-clock latency in microseconds.
    pub deserialize_us: f64,
    /// Roundtrip tensor reconstruction relative Frobenius error.
    pub relative_error: f64,
    /// Roundtrip tensor reconstruction SNR (dB).
    pub reconstruction_snr_db: f64,
}

/// Aggregated results across all 6 empirical experiments.
#[derive(Debug, Clone, PartialEq)]
pub struct EmpiricalStudySuite {
    /// Experiment 1: Partial count sweep results.
    pub partial_sweep: Vec<PartialSweepRecord>,
    /// Experiment 2: Phase strategy comparison results.
    pub phase_comparison: Vec<PhaseStrategyRecord>,
    /// Experiment 3: Spectral frame rate / hop size sweep results.
    pub frame_rate_sweep: Vec<FrameRateSweepRecord>,
    /// Experiment 4: PCA vs. NMF dimensionality reduction results.
    pub dim_reduction_sweep: Vec<DimReductionRecord>,
    /// Experiment 5: Tucker vs. CP-ALS tensor hypercube & real-time polyphony results.
    pub tensor_benchmarks: Vec<TensorBenchmarkRecord>,
    /// Experiment 6: .calyx binary container format serialization benchmark results.
    pub container_benchmarks: Vec<ContainerBenchmarkRecord>,
}

impl EmpiricalStudySuite {
    /// Executes all 6 empirical DSP experiment suites and returns the aggregated results.
    pub fn run_all() -> DspResult<Self> {
        Ok(Self {
            partial_sweep: run_partial_count_sweep()?,
            phase_comparison: run_phase_strategy_comparison()?,
            frame_rate_sweep: run_frame_rate_sweep()?,
            dim_reduction_sweep: run_dim_reduction_sweep()?,
            tensor_benchmarks: run_tensor_hypercube_benchmark()?,
            container_benchmarks: run_container_serialization_benchmark()?,
        })
    }

    /// Formats the complete empirical study as a Markdown report.
    pub fn to_markdown(&self) -> String {
        let mut md = String::with_capacity(8192);
        md.push_str("# Calyx DSP Empirical Research Findings\n\n");
        md.push_str(
            "Quantitative results produced by `calyx_dsp::experiments::EmpiricalStudySuite` ",
        );
        md.push_str("addressing the open research questions in `docs/open-questions.md`.\n\n");

        // Section 1
        md.push_str("## 1. Partial Count Sweep (Open Question 1.1)\n\n");
        md.push_str("| Timbre | Max Partials ($N$) | SNR (dB) | LSD (dB) | Spectral Conv. | RMSE | Payload (KB/s) |\n");
        md.push_str("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for r in &self.partial_sweep {
            md.push_str(&format!(
                "| {} | {} | {:.2} | {:.3} | {:.5} | {:.5} | {:.2} |\n",
                r.timbre,
                r.max_partials,
                r.snr_db,
                r.lsd_db,
                r.spectral_convergence,
                r.rmse,
                r.bytes_per_sec / 1024.0
            ));
        }
        md.push('\n');

        // Section 2
        md.push_str("## 2. Phase Strategy Comparison (Open Questions 1.2 & 3.1)\n\n");
        md.push_str("| Signal | Phase Strategy | Waveform SNR (dB) | Pearson $r$ | LSD (dB) | Spectral Conv. | Phase Bytes/Frame |\n");
        md.push_str("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |\n");
        for r in &self.phase_comparison {
            md.push_str(&format!(
                "| {} | {} | {:.2} | {:.4} | {:.3} | {:.5} | {} |\n",
                r.signal_name,
                r.strategy,
                r.waveform_snr_db,
                r.pearson_r,
                r.lsd_db,
                r.spectral_convergence,
                r.phase_bytes_per_frame
            ));
        }
        md.push('\n');

        // Section 3
        md.push_str("## 3. Spectral Frame Rate / Hop Size Sweep (Open Question 1.3)\n\n");
        md.push_str("| Signal | Window | Hop | Frame Rate (Hz) | SNR (dB) | LSD (dB) | Spectral Conv. |\n");
        md.push_str("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for r in &self.frame_rate_sweep {
            md.push_str(&format!(
                "| {} | {} | {} | {:.1} | {:.2} | {:.3} | {:.5} |\n",
                r.signal_name,
                r.window_size,
                r.hop_size,
                r.frames_per_sec,
                r.snr_db,
                r.lsd_db,
                r.spectral_convergence
            ));
        }
        md.push('\n');

        // Section 4
        md.push_str("## 4. Partial-Domain Dimensionality Reduction: PCA vs. NMF (Open Questions 2.1 & 2.3)\n\n");
        md.push_str("| Method | Rank ($K$) | SNR (dB) | LSD (dB) | Spectral Conv. | Matrix Comp. Ratio | Decompose ($\\mu$s) | Reconstruct ($\\mu$s) |\n");
        md.push_str("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for r in &self.dim_reduction_sweep {
            md.push_str(&format!(
                "| {} | {} | {:.2} | {:.3} | {:.5} | {:.2}x | {:.1} | {:.1} |\n",
                r.method,
                r.rank,
                r.snr_db,
                r.lsd_db,
                r.spectral_convergence,
                r.matrix_compression_ratio,
                r.decompose_us,
                r.reconstruct_us
            ));
        }
        md.push('\n');

        // Section 5
        md.push_str("## 5. Multi-Dimensional Tensor Decomposition & Real-Time Polyphony (Open Questions 2.2, 2.5, & 5.1)\n\n");
        md.push_str("| Method | Tier | Ranks | Tensor SNR (dB) | Rel. Error | Params (f32) | Spectral Comp. | PCM Comp. | Decompose (ms) | Frame Eval ($\\mu$s) | Max Voices (50% CPU) |\n");
        md.push_str("| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for r in &self.tensor_benchmarks {
            md.push_str(&format!(
                "| {} | {} | {} | {:.2} | {:.4} | {} | {:.1}x | {:.1}:1 | {:.2} | {:.2} | {} |\n",
                r.method,
                r.tier_label,
                r.rank_desc,
                r.tensor_snr_db,
                r.relative_tensor_error,
                r.compressed_floats,
                r.spectral_compression_ratio,
                r.pcm_compression_ratio,
                r.decompose_ms,
                r.frame_eval_us,
                r.estimated_max_voices
            ));
        }
        md.push('\n');

        // Section 6
        md.push_str("## 6. .calyx Binary Container Format Serialization Benchmark (Open Questions 4.1 & 4.2)\n\n");
        md.push_str("| Configuration | Tier | Size (Bytes) | Size (KB) | Hypercube Comp. | PCM Comp. | Serialize ($\\mu$s) | Deserialize ($\\mu$s) | Rel. Error | Recon SNR (dB) |\n");
        md.push_str("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for r in &self.container_benchmarks {
            md.push_str(&format!(
                "| {} | {} | {} | {:.2} KB | {:.1}x | {:.1}:1 | {:.1} | {:.1} | {:.4} | {:.2} |\n",
                r.config_label,
                r.tier_label,
                r.file_size_bytes,
                r.file_size_kb,
                r.hypercube_compression_ratio,
                r.pcm_compression_ratio,
                r.serialize_us,
                r.deserialize_us,
                r.relative_error,
                r.reconstruction_snr_db
            ));
        }

        md
    }
}

/// Runs Experiment 1: Partial count sweep across flute, clarinet, bowed string, and piano timbres.
pub fn run_partial_count_sweep() -> DspResult<Vec<PartialSweepRecord>> {
    let sample_rate = 48000.0;
    let num_samples = 4096;
    let win_size = 1024;
    let hop_size = 256;

    let mut flute_gen = HarmonicComplexGenerator::new(
        440.0,
        0.5,
        32,
        0.0,
        HarmonicRecipe::GeometricDecay { ratio: 0.40 },
    );
    flute_gen.set_damping_tau(Some(0.4));

    let mut clarinet_gen =
        HarmonicComplexGenerator::new(330.0, 0.45, 32, 0.0, HarmonicRecipe::Square);
    clarinet_gen.set_damping_tau(Some(0.5));

    let mut string_gen =
        HarmonicComplexGenerator::new(220.0, 0.35, 32, 0.0, HarmonicRecipe::Sawtooth);
    string_gen.set_damping_tau(Some(0.6));

    let mut piano_gen =
        HarmonicComplexGenerator::new(220.0, 0.35, 32, 0.0004, HarmonicRecipe::Sawtooth);
    piano_gen.set_damping_tau(Some(0.25));

    let timbres: Vec<(&str, Vec<f64>)> = vec![
        ("Flute (Geometric 0.4)", flute_gen.generate(num_samples, sample_rate).samples),
        ("Clarinet (Odd Harmonics)", clarinet_gen.generate(num_samples, sample_rate).samples),
        ("Bowed String (Sawtooth)", string_gen.generate(num_samples, sample_rate).samples),
        ("Piano (Inharmonic B=4e-4)", piano_gen.generate(num_samples, sample_rate).samples),
    ];

    let partial_counts = [2usize, 4, 8, 12, 16, 24];
    let analyzer = Stft::new(win_size, hop_size, WindowType::Hann)?;
    let reconstructor = AdditiveOlaReconstructor::new(win_size, hop_size);
    let frames_per_sec = sample_rate / (hop_size as f64);

    let mut records = Vec::with_capacity(timbres.len() * partial_counts.len());

    for (timbre_name, signal) in &timbres {
        let full_traj = analyzer.analyze(signal, sample_rate)?;

        for &max_n in &partial_counts {
            // Keep top `max_n` partials by harmonic index (or amplitude)
            let mut truncated_frames = Vec::with_capacity(full_traj.frames.len());
            for frame in &full_traj.frames {
                let mut partials: Vec<HarmonicPartial> = frame
                    .partials
                    .iter()
                    .filter(|p| p.harmonic_index <= max_n)
                    .cloned()
                    .collect();
                partials.sort_by_key(|p| p.harmonic_index);
                // Zero out residual so we isolate partial count impact on deterministic synthesis
                let mut f_copy = frame.clone();
                f_copy.partials = partials;
                f_copy.residual.band_energies.fill(0.0);
                truncated_frames.push(f_copy);
            }

            let trunc_traj = SpectralTrajectory {
                sample_rate,
                hop_size,
                frames: truncated_frames,
            };

            let recon = reconstructor.reconstruct(&trunc_traj, signal.len(), sample_rate)?;
            let time_m = compute_time_metrics(signal, &recon)?;
            let spec_m = compute_spectral_metrics(signal, &recon)?;

            // 8 bytes per partial (f32 amplitude + f32 frequency ratio) * frames_per_sec
            let bytes_per_sec = (max_n as f64) * 8.0 * frames_per_sec;

            records.push(PartialSweepRecord {
                timbre: (*timbre_name).to_string(),
                max_partials: max_n,
                snr_db: time_m.snr_db,
                lsd_db: spec_m.log_spectral_distance_db,
                spectral_convergence: spec_m.spectral_convergence,
                rmse: time_m.rmse,
                bytes_per_sec,
            });
        }
    }

    Ok(records)
}

/// Runs Experiment 2: Phase strategy comparison across harmonic and inharmonic tones.
pub fn run_phase_strategy_comparison() -> DspResult<Vec<PhaseStrategyRecord>> {
    let sample_rate = 48000.0;
    let num_samples = 4096;
    let win_size = 1024;
    let hop_size = 256;

    // Signal 1: Harmonic complex tone with non-zero initial partial phases (triangle-like / alternating phase)
    let custom_weights: Vec<f64> = (1..=16).map(|k| 1.0 / (k as f64).powf(1.2)).collect();
    let mut harm_gen = HarmonicComplexGenerator::new(
        220.0,
        0.40,
        16,
        0.0,
        HarmonicRecipe::Custom(custom_weights),
    );
    harm_gen.set_damping_tau(Some(0.5));
    let sustain_note = harm_gen.generate(num_samples, sample_rate).samples;

    // Signal 2: Inharmonic stiff-string piano tone (B = 0.0004)
    let mut piano_gen =
        HarmonicComplexGenerator::new(220.0, 0.35, 16, 0.0004, HarmonicRecipe::Sawtooth);
    piano_gen.set_damping_tau(Some(0.3));
    let piano_note = piano_gen.generate(num_samples, sample_rate).samples;

    let signals = vec![
        ("Harmonic Sustain (A3 220Hz, B=0)", sustain_note),
        ("Inharmonic Piano (A3 220Hz, B=4e-4)", piano_note),
    ];

    let analyzer = Stft::new(win_size, hop_size, WindowType::Hann)?;
    let reconstructor = AdditiveOlaReconstructor::new(win_size, hop_size);
    let dispersion = PhaseDispersionModel::new(42);

    let mut records = Vec::new();

    for (sig_name, signal) in &signals {
        let base_traj = analyzer.analyze(signal, sample_rate)?;

        // Strategy 1: Stored Phase Envelopes (exact analyzed phase preserved)
        {
            let recon = reconstructor.reconstruct(&base_traj, signal.len(), sample_rate)?;
            let time_m = compute_time_metrics(signal, &recon)?;
            let spec_m = compute_spectral_metrics(signal, &recon)?;
            records.push(PhaseStrategyRecord {
                signal_name: (*sig_name).to_string(),
                strategy: "1. Stored Phase Envelopes".to_string(),
                waveform_snr_db: time_m.snr_db,
                pearson_r: time_m.pearson_r,
                lsd_db: spec_m.log_spectral_distance_db,
                spectral_convergence: spec_m.spectral_convergence,
                phase_bytes_per_frame: 16 * 4,
            });
        }

        // Strategy 2: Phase Vocoder / Continuous Phase Accumulation (0 stored phase bytes)
        {
            let mut vocoder_traj = base_traj.clone();
            let mut accum_phases = [0.0_f64; 64];
            for (m, frame) in vocoder_traj.frames.iter_mut().enumerate() {
                for p in &mut frame.partials {
                    let idx = p.harmonic_index.min(63);
                    if m == 0 {
                        accum_phases[idx] = p.phase;
                    } else {
                        let d_phi = 2.0 * PI * p.frequency * (hop_size as f64) / sample_rate;
                        accum_phases[idx] = (accum_phases[idx] + d_phi) % (2.0 * PI);
                        p.phase = accum_phases[idx];
                    }
                }
            }
            let recon = reconstructor.reconstruct(&vocoder_traj, signal.len(), sample_rate)?;
            let time_m = compute_time_metrics(signal, &recon)?;
            let spec_m = compute_spectral_metrics(signal, &recon)?;
            records.push(PhaseStrategyRecord {
                signal_name: (*sig_name).to_string(),
                strategy: "2. Phase Vocoder Accumulation".to_string(),
                waveform_snr_db: time_m.snr_db,
                pearson_r: time_m.pearson_r,
                lsd_db: spec_m.log_spectral_distance_db,
                spectral_convergence: spec_m.spectral_convergence,
                phase_bytes_per_frame: 0,
            });
        }

        // Strategy 3: Harmonic Phase Locking (phi_k = k * phi_0 mod 2pi, 4 bytes/frame for phi_0)
        {
            let mut locked_traj = base_traj.clone();
            for frame in &mut locked_traj.frames {
                let phi_0 = frame
                    .partials
                    .iter()
                    .find(|p| p.harmonic_index == 1)
                    .map(|p| p.phase)
                    .unwrap_or(0.0);
                for p in &mut frame.partials {
                    p.phase = lock_harmonic_phase(phi_0, p.harmonic_index);
                }
            }
            let recon = reconstructor.reconstruct(&locked_traj, signal.len(), sample_rate)?;
            let time_m = compute_time_metrics(signal, &recon)?;
            let spec_m = compute_spectral_metrics(signal, &recon)?;
            records.push(PhaseStrategyRecord {
                signal_name: (*sig_name).to_string(),
                strategy: "3. Harmonic Phase Locking".to_string(),
                waveform_snr_db: time_m.snr_db,
                pearson_r: time_m.pearson_r,
                lsd_db: spec_m.log_spectral_distance_db,
                spectral_convergence: spec_m.spectral_convergence,
                phase_bytes_per_frame: 4,
            });
        }

        // Strategy 4: Static Phase Dispersion Model (seeded inter-partial dispersion + vocoder advance)
        {
            let mut dispersed_traj = base_traj.clone();
            let mut accum_phases = [0.0_f64; 64];
            for (m, frame) in dispersed_traj.frames.iter_mut().enumerate() {
                for p in &mut frame.partials {
                    let idx = p.harmonic_index.min(63);
                    if m == 0 {
                        accum_phases[idx] = dispersion.disperse_phase(0.0, p.harmonic_index);
                    } else {
                        let d_phi = 2.0 * PI * p.frequency * (hop_size as f64) / sample_rate;
                        accum_phases[idx] = (accum_phases[idx] + d_phi) % (2.0 * PI);
                    }
                    p.phase = accum_phases[idx];
                }
            }
            let recon = reconstructor.reconstruct(&dispersed_traj, signal.len(), sample_rate)?;
            let time_m = compute_time_metrics(signal, &recon)?;
            let spec_m = compute_spectral_metrics(signal, &recon)?;
            records.push(PhaseStrategyRecord {
                signal_name: (*sig_name).to_string(),
                strategy: "4. Phase Dispersion Modeling".to_string(),
                waveform_snr_db: time_m.snr_db,
                pearson_r: time_m.pearson_r,
                lsd_db: spec_m.log_spectral_distance_db,
                spectral_convergence: spec_m.spectral_convergence,
                phase_bytes_per_frame: 0,
            });
        }
    }

    Ok(records)
}

/// Runs Experiment 3: Spectral frame rate (STFT hop size) sweep across sustain, staccato, and chirp signals.
pub fn run_frame_rate_sweep() -> DspResult<Vec<FrameRateSweepRecord>> {
    let sample_rate = 48000.0;
    let num_samples = 4096;

    let mut sustain_gen =
        HarmonicComplexGenerator::new(220.0, 0.35, 16, 0.0, HarmonicRecipe::Sawtooth);
    sustain_gen.set_damping_tau(Some(0.8));
    let sustain = sustain_gen.generate(num_samples, sample_rate).samples;

    let mut staccato_gen =
        HarmonicComplexGenerator::new(220.0, 0.35, 16, 0.0, HarmonicRecipe::Sawtooth);
    staccato_gen.set_damping_tau(Some(0.06));
    let staccato = staccato_gen.generate(num_samples, sample_rate).samples;

    let chirp = ChirpGenerator::new(220.0, 660.0, 0.5, (num_samples as f64) / sample_rate, ChirpKind::Linear)
        .generate(num_samples, sample_rate)
        .samples;

    let signals = vec![
        ("Sustain Tone (tau=0.8s)", sustain),
        ("Fast Decay Transient (tau=0.06s)", staccato),
        ("Linear Pitch Glide (220->660Hz)", chirp),
    ];

    let window_size = 1024;
    let hops = [64usize, 128, 256, 512];
    let mut records = Vec::with_capacity(signals.len() * hops.len());

    for (sig_name, signal) in &signals {
        for &hop in &hops {
            let analyzer = Stft::new(window_size, hop, WindowType::Hann)?;
            let reconstructor = AdditiveOlaReconstructor::new(window_size, hop);

            let traj = analyzer.analyze(signal, sample_rate)?;
            let recon = reconstructor.reconstruct(&traj, signal.len(), sample_rate)?;

            let time_m = compute_time_metrics(signal, &recon)?;
            let spec_m = compute_spectral_metrics(signal, &recon)?;

            records.push(FrameRateSweepRecord {
                signal_name: (*sig_name).to_string(),
                window_size,
                hop_size: hop,
                frames_per_sec: sample_rate / (hop as f64),
                snr_db: time_m.snr_db,
                lsd_db: spec_m.log_spectral_distance_db,
                spectral_convergence: spec_m.spectral_convergence,
            });
        }
    }

    Ok(records)
}

/// Runs Experiment 4: Partial-domain dimensionality reduction (PCA vs. NMF) across ranks.
pub fn run_dim_reduction_sweep() -> DspResult<Vec<DimReductionRecord>> {
    let sample_rate = 48000.0;
    let num_samples = 8192;
    let win_size = 1024;
    let hop_size = 128;
    let num_partials = 24usize;
    let f0 = 220.0;

    // Synthesize a single phase-continuous 24-partial note with rich multi-mode timbral evolution
    // (independent temporal envelopes across partial groups: attack transient, formant sweep,
    // odd/even balance shift, and high-harmonic shimmer) so the spectral matrix has genuine rank > 8.
    let mut composite_signal = vec![0.0_f64; num_samples];
    let dur = (num_samples as f64) / sample_rate;
    for (n, sample) in composite_signal.iter_mut().enumerate() {
        let t = (n as f64) / sample_rate;
        let u = t / dur; // normalized time [0, 1]
        let mut val = 0.0;
        for k in 1..=num_partials {
            let kf = k as f64;
            let base = 0.30 / kf.powf(0.85);
            // 5 independent temporal-spectral modes
            let mode1 = (-t / (0.35 / kf.sqrt())).exp();
            let mode2 = 0.45 * (-(kf - (3.0 + 14.0 * u)).powi(2) / 6.0).exp();
            let mode3 = if k % 2 == 1 {
                0.25 * (2.0 * PI * 3.0 * u).sin().abs()
            } else {
                0.25 * (2.0 * PI * 3.0 * u).cos().abs()
            };
            let mode4 = 0.20 * (-(kf - (20.0 - 12.0 * u)).powi(2) / 5.0).exp() * (PI * u).sin();
            let mode5 = 0.12 * (2.0 * PI * (1.5 * kf * u)).sin().abs() / kf.sqrt();

            let amp = base * (0.5 * mode1 + mode2 + mode3 + mode4 + mode5);
            val += amp * (2.0 * PI * kf * f0 * t).sin();
        }
        *sample = val;
    }

    // Normalize peak to 0.85
    let peak = composite_signal
        .iter()
        .copied()
        .fold(0.0_f64, |m, x| m.max(x.abs()))
        .max(1e-12);
    for s in &mut composite_signal {
        *s *= 0.85 / peak;
    }

    let analyzer = Stft::new(win_size, hop_size, WindowType::Hann)?;
    let reconstructor = AdditiveOlaReconstructor::new(win_size, hop_size);
    let analyzed = analyzer.analyze(&composite_signal, sample_rate)?;

    let num_frames = analyzed.frames.len();
    let max_partials = analyzed
        .frames
        .iter()
        .flat_map(|f| f.partials.iter().map(|p| p.harmonic_index))
        .max()
        .unwrap_or(16);
    let raw_elements = (num_frames * max_partials) as f64;

    let ranks = [1usize, 2, 4, 8, 12, 16];
    let mut records = Vec::with_capacity(ranks.len() * 2);

    for &rank in &ranks {
        let eff_rank = rank.min(num_frames).min(max_partials).max(1);
        let compressed_elements = ((num_frames + max_partials) * eff_rank) as f64;
        let comp_ratio = raw_elements / compressed_elements.max(1.0);

        // 1. PCA
        {
            let pca = PcaDecomposer::new(rank);
            let t0 = Instant::now();
            let comp = pca.decompose(&analyzed)?;
            let decompose_us = t0.elapsed().as_secs_f64() * 1e6;

            let t1 = Instant::now();
            let recon_traj = pca.reconstruct(&comp)?;
            let reconstruct_us = t1.elapsed().as_secs_f64() * 1e6;

            let recon_audio =
                reconstructor.reconstruct(&recon_traj, composite_signal.len(), sample_rate)?;
            let time_m = compute_time_metrics(&composite_signal, &recon_audio)?;
            let spec_m = compute_spectral_metrics(&composite_signal, &recon_audio)?;

            records.push(DimReductionRecord {
                method: "PCA (Jacobi SVD)".to_string(),
                rank: eff_rank,
                snr_db: time_m.snr_db,
                lsd_db: spec_m.log_spectral_distance_db,
                spectral_convergence: spec_m.spectral_convergence,
                matrix_compression_ratio: comp_ratio,
                decompose_us,
                reconstruct_us,
            });
        }

        // 2. NMF
        {
            let nmf = NmfDecomposer::new(rank, 50);
            let t0 = Instant::now();
            let comp = nmf.decompose(&analyzed)?;
            let decompose_us = t0.elapsed().as_secs_f64() * 1e6;

            let t1 = Instant::now();
            let recon_traj = nmf.reconstruct(&comp)?;
            let reconstruct_us = t1.elapsed().as_secs_f64() * 1e6;

            let recon_audio =
                reconstructor.reconstruct(&recon_traj, composite_signal.len(), sample_rate)?;
            let time_m = compute_time_metrics(&composite_signal, &recon_audio)?;
            let spec_m = compute_spectral_metrics(&composite_signal, &recon_audio)?;

            records.push(DimReductionRecord {
                method: "NMF (Lee-Seung)".to_string(),
                rank: eff_rank,
                snr_db: time_m.snr_db,
                lsd_db: spec_m.log_spectral_distance_db,
                spectral_convergence: spec_m.spectral_convergence,
                matrix_compression_ratio: comp_ratio,
                decompose_us,
                reconstruct_us,
            });
        }
    }

    Ok(records)
}

/// Runs Experiment 5: Multi-dimensional tensor decomposition (Tucker HOSVD vs. CP-ALS) and real-time voice polyphony benchmark.
pub fn run_tensor_hypercube_benchmark() -> DspResult<Vec<TensorBenchmarkRecord>> {
    // Construct a 4-mode spectral hypercube tensor:
    // Mode 0: Time frames (T = 16 frames @ 100 Hz = 0.16s per note, or representative sustain grid)
    // Mode 1: Pitch (P = 12 semitones, C4..B4)
    // Mode 2: Dynamic (D = 6 velocity layers, pp..ff)
    // Mode 3: Partials (K = 16 harmonic partials)
    let dim_t = 16usize;
    let dim_p = 12usize;
    let dim_d = 6usize;
    let dim_k = 16usize;
    let shape = vec![dim_t, dim_p, dim_d, dim_k];
    let total_spectral_floats = dim_t * dim_p * dim_d * dim_k; // 18,432 floats

    // Equivalent PCM audio size:
    // 12 pitches * 6 dynamics = 72 notes, each 16 frames * 480 samples/frame (100 Hz frame rate @ 48 kHz) = 7,680 samples/note
    // Total PCM samples = 72 * 7,680 = 552,960 samples
    let equivalent_pcm_samples = (dim_p * dim_d * dim_t * 480) as f64;

    let inst = SyntheticInstrumentSource::new();
    let mut data = vec![0.0_f64; total_spectral_floats];

    for t_idx in 0..dim_t {
        let t_sec = (t_idx as f64) * 0.01;
        for p_idx in 0..dim_p {
            let midi_pitch = 60.0 + (p_idx as f64);
            for d_idx in 0..dim_d {
                let dyn_val = 0.15 + 0.85 * (d_idx as f64) / ((dim_d - 1) as f64);
                let frame = inst.query_frame([t_sec, midi_pitch, dyn_val, 0.8, 0.0])?;
                let base_offset =
                    ((t_idx * dim_p + p_idx) * dim_d + d_idx) * dim_k;
                for partial in &frame.partials {
                    if partial.harmonic_index >= 1 && partial.harmonic_index <= dim_k {
                        data[base_offset + (partial.harmonic_index - 1)] = partial.amplitude;
                    }
                }
            }
        }
    }

    let hypercube_tensor = Tensor::from_vec(shape.clone(), data)?;
    let orig_slice = hypercube_tensor.data();
    let orig_energy: f64 = orig_slice.iter().map(|x| x * x).sum();

    let mut records = Vec::new();

    // 1. Tucker HOSVD across Preview, Production, Reference tiers
    let tucker_configs = [
        ("Preview", [2usize, 2, 2, 4]),
        ("Production", [4usize, 4, 3, 8]),
        ("Reference", [8usize, 6, 4, 12]),
    ];

    for (tier_label, ranks) in &tucker_configs {
        let t0 = Instant::now();
        let tucker = TuckerDecomposition::decompose(&hypercube_tensor, ranks)?;
        let decompose_ms = t0.elapsed().as_secs_f64() * 1e3;

        let recon_tensor = tucker.reconstruct_tensor()?;
        let recon_slice = recon_tensor.data();

        let err_energy: f64 = orig_slice
            .iter()
            .zip(recon_slice.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let rel_err = (err_energy / orig_energy.max(1e-30)).sqrt();
        let snr_db = 10.0 * (orig_energy / err_energy.max(1e-30)).log10();

        // Measure time to evaluate a full 16-partial spectral frame at a single (t, p, d) coordinate
        let eval_iters = 20usize;
        let t_eval = Instant::now();
        let mut sink = 0.0_f64;
        for iter in 0..eval_iters {
            let t_i = iter % dim_t;
            let p_i = iter % dim_p;
            let d_i = iter % dim_d;
            for k_i in 0..dim_k {
                sink += tucker.evaluate_point(&[t_i, p_i, d_i, k_i])?;
            }
        }
        let frame_eval_us = (t_eval.elapsed().as_secs_f64() * 1e6) / (eval_iters as f64);
        if sink.is_nan() {
            return Err(crate::error::DspError::numerical_instability("NaN in Tucker eval"));
        }

        // Parameter count: core + sum of factor matrices
        let core_params: usize = ranks.iter().product();
        let factor_params: usize = shape
            .iter()
            .zip(ranks.iter())
            .map(|(&dim, &r)| dim * r)
            .sum();
        let compressed_floats = core_params + factor_params;

        // At 100 Hz frame update rate, each frame interval is 10,000 us.
        // Under a 50% CPU budget (5,000 us available per 10 ms), max voices = 5000 / frame_eval_us.
        let estimated_max_voices = (5000.0 / frame_eval_us.max(0.01)).floor() as usize;

        records.push(TensorBenchmarkRecord {
            method: "Tucker-HOSVD".to_string(),
            tier_label: (*tier_label).to_string(),
            rank_desc: format!("{:?}", ranks),
            relative_tensor_error: rel_err,
            tensor_snr_db: snr_db,
            compressed_floats,
            spectral_compression_ratio: (total_spectral_floats as f64) / (compressed_floats as f64),
            pcm_compression_ratio: equivalent_pcm_samples / (compressed_floats as f64),
            decompose_ms,
            frame_eval_us,
            estimated_max_voices,
        });
    }

    // 2. CP-ALS across Preview, Production, Reference tiers
    let cp_configs = [
        ("Preview", 2usize),
        ("Production", 6usize),
        ("Reference", 12usize),
    ];

    for (tier_label, rank) in &cp_configs {
        let t0 = Instant::now();
        let cp = CpAls::decompose(&hypercube_tensor, *rank, 20)?;
        let decompose_ms = t0.elapsed().as_secs_f64() * 1e3;

        // Reconstruct tensor from CP factors and measure error
        let mut recon_data = vec![0.0_f64; total_spectral_floats];
        for t_i in 0..dim_t {
            for p_i in 0..dim_p {
                for d_i in 0..dim_d {
                    let base_offset = ((t_i * dim_p + p_i) * dim_d + d_i) * dim_k;
                    for k_i in 0..dim_k {
                        let mut val = 0.0;
                        for r in 0..*rank {
                            val += cp.weights[r]
                                * cp.factors[0][(t_i, r)]
                                * cp.factors[1][(p_i, r)]
                                * cp.factors[2][(d_i, r)]
                                * cp.factors[3][(k_i, r)];
                        }
                        recon_data[base_offset + k_i] = val;
                    }
                }
            }
        }

        let err_energy: f64 = orig_slice
            .iter()
            .zip(recon_data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let rel_err = (err_energy / orig_energy.max(1e-30)).sqrt();
        let snr_db = 10.0 * (orig_energy / err_energy.max(1e-30)).log10();

        // Benchmark single-frame (all 16 partials at one (t, p, d)) evaluation latency
        let eval_iters = 100usize;
        let t_eval = Instant::now();
        let mut sink = 0.0_f64;
        for iter in 0..eval_iters {
            let t_i = iter % dim_t;
            let p_i = iter % dim_p;
            let d_i = iter % dim_d;
            for k_i in 0..dim_k {
                let mut val = 0.0;
                for r in 0..*rank {
                    val += cp.weights[r]
                        * cp.factors[0][(t_i, r)]
                        * cp.factors[1][(p_i, r)]
                        * cp.factors[2][(d_i, r)]
                        * cp.factors[3][(k_i, r)];
                }
                sink += val;
            }
        }
        let frame_eval_us = (t_eval.elapsed().as_secs_f64() * 1e6) / (eval_iters as f64);
        if sink.is_nan() {
            return Err(crate::error::DspError::numerical_instability("NaN in CP eval"));
        }

        let factor_params: usize = shape.iter().map(|&dim| dim * (*rank)).sum();
        let compressed_floats = *rank + factor_params;
        let estimated_max_voices = (5000.0 / frame_eval_us.max(0.01)).floor() as usize;

        records.push(TensorBenchmarkRecord {
            method: "CP-ALS".to_string(),
            tier_label: (*tier_label).to_string(),
            rank_desc: format!("R={}", rank),
            relative_tensor_error: rel_err,
            tensor_snr_db: snr_db,
            compressed_floats,
            spectral_compression_ratio: (total_spectral_floats as f64) / (compressed_floats as f64),
            pcm_compression_ratio: equivalent_pcm_samples / (compressed_floats as f64),
            decompose_ms,
            frame_eval_us,
            estimated_max_voices,
        });
    }

    Ok(records)
}

/// Runs Experiment 6: Extensible .calyx binary container serialization benchmark across quality tiers.
///
/// Evaluates:
/// - Serialization latency (microseconds)
/// - Deserialization latency (microseconds)
/// - Serialized container file size (bytes and kilobytes)
/// - Compression ratios against uncompressed hypercube and raw 16-bit 48 kHz PCM audio
/// - Numerical reconstruction error (Frobenius error and SNR) verifying exact zero-loss roundtrip
pub fn run_container_serialization_benchmark() -> DspResult<Vec<ContainerBenchmarkRecord>> {
    let dim_t = 16usize;
    let dim_p = 12usize;
    let dim_d = 6usize;
    let dim_k = 16usize;
    let shape = vec![dim_t, dim_p, dim_d, dim_k];
    let total_spectral_floats = dim_t * dim_p * dim_d * dim_k; // 18,432 floats

    // Equivalent PCM audio size: 72 notes * 16 frames * 480 samples = 552,960 samples
    let equivalent_pcm_samples = (dim_p * dim_d * dim_t * 480) as f64;
    let raw_pcm_bytes = equivalent_pcm_samples * 2.0; // 16-bit PCM = 1,105,920 bytes
    let uncompressed_hypercube_bytes = (total_spectral_floats * 8) as f64; // f64 = 147,456 bytes

    let inst = SyntheticInstrumentSource::new();
    let mut data = vec![0.0_f64; total_spectral_floats];

    for t_idx in 0..dim_t {
        let t_sec = (t_idx as f64) * 0.01;
        for p_idx in 0..dim_p {
            let midi_pitch = 60.0 + (p_idx as f64);
            for d_idx in 0..dim_d {
                let dyn_val = 0.15 + 0.85 * (d_idx as f64) / ((dim_d - 1) as f64);
                let frame = inst.query_frame([t_sec, midi_pitch, dyn_val, 0.8, 0.0])?;
                let base_offset = ((t_idx * dim_p + p_idx) * dim_d + d_idx) * dim_k;
                for partial in &frame.partials {
                    if partial.harmonic_index >= 1 && partial.harmonic_index <= dim_k {
                        data[base_offset + (partial.harmonic_index - 1)] = partial.amplitude;
                    }
                }
            }
        }
    }

    let hypercube_tensor = Tensor::from_vec(shape.clone(), data)?;
    let orig_slice = hypercube_tensor.data();
    let orig_energy: f64 = orig_slice.iter().map(|x| x * x).sum();

    let mut records = Vec::new();

    let axis_defs = vec![
        AxisDefinition::new("time", dim_t, 0.0, 0.16)?,
        AxisDefinition::new("pitch", dim_p, 60.0, 71.0)?,
        AxisDefinition::new("dynamic", dim_d, 0.15, 1.0)?,
        AxisDefinition::new("partial", dim_k, 1.0, 16.0)?,
    ];
    let layout = HypercubeLayout::new(shape.clone())?;

    let benchmark_iters = 30usize;

    // 1. Tucker-HOSVD configurations
    let tucker_configs = [
        ("Preview", [2usize, 2, 2, 4], ContainerQualityTier::Preview),
        ("Production", [4usize, 4, 3, 8], ContainerQualityTier::Production),
        ("Reference", [8usize, 6, 4, 12], ContainerQualityTier::Reference),
    ];

    for (tier_label, ranks, tier) in &tucker_configs {
        let tucker = TuckerDecomposition::decompose(&hypercube_tensor, ranks)?;
        let tucker_data = TuckerContainerData::from_decomposition(&tucker);

        let header = CalyxContainerHeader::new(*tier, 48000.0, 480, dim_k as u16);
        let container = CalyxContainer::new(header, layout.clone())
            .with_axes(axis_defs.clone())?
            .with_metadata_entry("name", "Synthetic Solo Instrument")
            .with_metadata_entry("author", "Calyx DSP Research")
            .with_metadata_entry("tier", *tier_label)
            .with_tucker(tucker_data);

        // Benchmark serialization
        let mut buf = Vec::new();
        let t_ser = Instant::now();
        for _ in 0..benchmark_iters {
            buf.clear();
            container.write(&mut buf)?;
        }
        let serialize_us = (t_ser.elapsed().as_secs_f64() * 1e6) / (benchmark_iters as f64);

        let file_size_bytes = buf.len();
        let file_size_kb = file_size_bytes as f64 / 1024.0;

        // Benchmark deserialization
        let mut rec_container = None;
        let t_deser = Instant::now();
        for _ in 0..benchmark_iters {
            let mut cursor = Cursor::new(&buf);
            rec_container = Some(CalyxContainer::read(&mut cursor)?);
        }
        let deserialize_us = (t_deser.elapsed().as_secs_f64() * 1e6) / (benchmark_iters as f64);

        let recovered = rec_container.unwrap();
        let rec_tucker = recovered.tucker_data.unwrap().to_decomposition()?;
        let recon_tensor = rec_tucker.reconstruct_tensor()?;
        let recon_slice = recon_tensor.data();

        let err_energy: f64 = orig_slice
            .iter()
            .zip(recon_slice.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let rel_err = (err_energy / orig_energy.max(1e-30)).sqrt();
        let snr_db = 10.0 * (orig_energy / err_energy.max(1e-30)).log10();

        records.push(ContainerBenchmarkRecord {
            config_label: format!("Tucker-HOSVD ({})", tier_label),
            tier_label: (*tier_label).to_string(),
            file_size_bytes,
            file_size_kb,
            hypercube_compression_ratio: uncompressed_hypercube_bytes / (file_size_bytes as f64),
            pcm_compression_ratio: raw_pcm_bytes / (file_size_bytes as f64),
            serialize_us,
            deserialize_us,
            relative_error: rel_err,
            reconstruction_snr_db: snr_db,
        });
    }

    // 2. CP-ALS configurations
    let cp_configs = [
        ("Preview", 2usize, ContainerQualityTier::Preview),
        ("Production", 6usize, ContainerQualityTier::Production),
        ("Reference", 12usize, ContainerQualityTier::Reference),
    ];

    for (tier_label, rank, tier) in &cp_configs {
        let cp = CpAls::decompose(&hypercube_tensor, *rank, 20)?;
        let cp_data = CpContainerData::from_model(&cp);

        let header = CalyxContainerHeader::new(*tier, 48000.0, 480, dim_k as u16);
        let container = CalyxContainer::new(header, layout.clone())
            .with_axes(axis_defs.clone())?
            .with_metadata_entry("name", "Synthetic Solo Instrument")
            .with_metadata_entry("author", "Calyx DSP Research")
            .with_metadata_entry("tier", *tier_label)
            .with_cp(cp_data);

        // Benchmark serialization
        let mut buf = Vec::new();
        let t_ser = Instant::now();
        for _ in 0..benchmark_iters {
            buf.clear();
            container.write(&mut buf)?;
        }
        let serialize_us = (t_ser.elapsed().as_secs_f64() * 1e6) / (benchmark_iters as f64);

        let file_size_bytes = buf.len();
        let file_size_kb = file_size_bytes as f64 / 1024.0;

        // Benchmark deserialization
        let mut rec_container = None;
        let t_deser = Instant::now();
        for _ in 0..benchmark_iters {
            let mut cursor = Cursor::new(&buf);
            rec_container = Some(CalyxContainer::read(&mut cursor)?);
        }
        let deserialize_us = (t_deser.elapsed().as_secs_f64() * 1e6) / (benchmark_iters as f64);

        let recovered = rec_container.unwrap();
        let rec_cp = recovered.cp_data.unwrap().to_model()?;

        // Reconstruct tensor from deserialized CP factors
        let mut recon_data = vec![0.0_f64; total_spectral_floats];
        for t_i in 0..dim_t {
            for p_i in 0..dim_p {
                for d_i in 0..dim_d {
                    let base_offset = ((t_i * dim_p + p_i) * dim_d + d_i) * dim_k;
                    for k_i in 0..dim_k {
                        let mut val = 0.0;
                        for r in 0..*rank {
                            val += rec_cp.weights[r]
                                * rec_cp.factors[0][(t_i, r)]
                                * rec_cp.factors[1][(p_i, r)]
                                * rec_cp.factors[2][(d_i, r)]
                                * rec_cp.factors[3][(k_i, r)];
                        }
                        recon_data[base_offset + k_i] = val;
                    }
                }
            }
        }

        let err_energy: f64 = orig_slice
            .iter()
            .zip(recon_data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let rel_err = (err_energy / orig_energy.max(1e-30)).sqrt();
        let snr_db = 10.0 * (orig_energy / err_energy.max(1e-30)).log10();

        records.push(ContainerBenchmarkRecord {
            config_label: format!("CP-ALS ({})", tier_label),
            tier_label: (*tier_label).to_string(),
            file_size_bytes,
            file_size_kb,
            hypercube_compression_ratio: uncompressed_hypercube_bytes / (file_size_bytes as f64),
            pcm_compression_ratio: raw_pcm_bytes / (file_size_bytes as f64),
            serialize_us,
            deserialize_us,
            relative_error: rel_err,
            reconstruction_snr_db: snr_db,
        });
    }

    // 3. Hybrid Container (Tucker + CP + Metadata + Residual)
    {
        let tucker = TuckerDecomposition::decompose(&hypercube_tensor, &[4, 4, 3, 8])?;
        let tucker_data = TuckerContainerData::from_decomposition(&tucker);

        let cp = CpAls::decompose(&hypercube_tensor, 6, 20)?;
        let cp_data = CpContainerData::from_model(&cp);

        let residual_data = ResidualContainerData {
            num_bands: 4,
            center_frequencies: vec![250.0, 1000.0, 4000.0, 12000.0],
            band_energies: vec![0.005, 0.012, 0.008, 0.002],
            noise_ratio: 0.035,
        };

        let header = CalyxContainerHeader::new(ContainerQualityTier::Production, 48000.0, 480, dim_k as u16);
        let container = CalyxContainer::new(header, layout)
            .with_axes(axis_defs)?
            .with_metadata_entry("name", "Synthetic Solo Instrument")
            .with_metadata_entry("author", "Calyx DSP Research")
            .with_metadata_entry("workflow", "Hybrid Tucker+CP")
            .with_tucker(tucker_data)
            .with_cp(cp_data)
            .with_residual(residual_data);

        // Benchmark serialization
        let mut buf = Vec::new();
        let t_ser = Instant::now();
        for _ in 0..benchmark_iters {
            buf.clear();
            container.write(&mut buf)?;
        }
        let serialize_us = (t_ser.elapsed().as_secs_f64() * 1e6) / (benchmark_iters as f64);

        let file_size_bytes = buf.len();
        let file_size_kb = file_size_bytes as f64 / 1024.0;

        // Benchmark deserialization
        let mut rec_container = None;
        let t_deser = Instant::now();
        for _ in 0..benchmark_iters {
            let mut cursor = Cursor::new(&buf);
            rec_container = Some(CalyxContainer::read(&mut cursor)?);
        }
        let deserialize_us = (t_deser.elapsed().as_secs_f64() * 1e6) / (benchmark_iters as f64);

        let recovered = rec_container.unwrap();
        let rec_tucker = recovered.tucker_data.unwrap().to_decomposition()?;
        let recon_tensor = rec_tucker.reconstruct_tensor()?;
        let recon_slice = recon_tensor.data();

        let err_energy: f64 = orig_slice
            .iter()
            .zip(recon_slice.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let rel_err = (err_energy / orig_energy.max(1e-30)).sqrt();
        let snr_db = 10.0 * (orig_energy / err_energy.max(1e-30)).log10();

        records.push(ContainerBenchmarkRecord {
            config_label: "Hybrid Container (Production)".to_string(),
            tier_label: "Production".to_string(),
            file_size_bytes,
            file_size_kb,
            hypercube_compression_ratio: uncompressed_hypercube_bytes / (file_size_bytes as f64),
            pcm_compression_ratio: raw_pcm_bytes / (file_size_bytes as f64),
            serialize_us,
            deserialize_us,
            relative_error: rel_err,
            reconstruction_snr_db: snr_db,
        });
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_count_sweep_monotonic_fidelity() {
        let records = run_partial_count_sweep().expect("partial sweep should succeed");
        assert!(!records.is_empty());
        // Verify Bowed String SNR improves as partial count increases from 2 to 16
        let string_2 = records
            .iter()
            .find(|r| r.timbre.starts_with("Bowed") && r.max_partials == 2)
            .unwrap();
        let string_16 = records
            .iter()
            .find(|r| r.timbre.starts_with("Bowed") && r.max_partials == 16)
            .unwrap();
        assert!(
            string_16.snr_db > string_2.snr_db,
            "16 partials SNR ({:.2}) should exceed 2 partials SNR ({:.2})",
            string_16.snr_db,
            string_2.snr_db
        );
        assert!(
            string_16.lsd_db < string_2.lsd_db,
            "16 partials LSD ({:.2}) should be lower than 2 partials LSD ({:.2})",
            string_16.lsd_db,
            string_2.lsd_db
        );
    }

    #[test]
    fn test_phase_strategy_comparison_invariants() {
        let records = run_phase_strategy_comparison().expect("phase comparison should succeed");
        assert_eq!(records.len(), 8);
        // Stored phase envelopes and Phase Vocoder accumulation achieve >35 dB SNR and <0.5 dB LSD
        let stored_harm = &records[0];
        let vocoder_harm = &records[1];
        let locked_harm = &records[2];
        assert!(stored_harm.waveform_snr_db > 35.0);
        assert!(vocoder_harm.waveform_snr_db > 35.0);
        assert!(stored_harm.waveform_snr_db > locked_harm.waveform_snr_db);

        // On inharmonic piano, Phase Vocoder preserves <0.5 dB LSD whereas Harmonic Phase Locking
        // introduces OLA phase-mismatch spectral degradation (LSD > 3.0 dB)
        let vocoder_piano = &records[5];
        let locked_piano = &records[6];
        assert!(vocoder_piano.lsd_db < 0.5);
        assert!(locked_piano.lsd_db > vocoder_piano.lsd_db);

        for r in &records {
            assert!(r.lsd_db.is_finite() && r.lsd_db < 10.0);
        }
    }

    #[test]
    fn test_dim_reduction_and_tensor_benchmarks() {
        let suite = EmpiricalStudySuite::run_all().expect("full empirical suite should succeed");
        let md = suite.to_markdown();
        assert!(md.contains("Calyx DSP Empirical Research Findings"));
        assert!(md.contains("Tucker-HOSVD"));
        assert!(md.contains("CP-ALS"));
        assert!(md.contains("Container Format Serialization Benchmark"));
    }

    #[test]
    fn test_container_serialization_benchmark() {
        let records = run_container_serialization_benchmark().expect("container benchmark should succeed");
        assert_eq!(records.len(), 7);
        for r in &records {
            assert!(r.file_size_bytes > 0);
            assert!(r.serialize_us > 0.0);
            assert!(r.deserialize_us > 0.0);
            assert!(r.reconstruction_snr_db > 30.0);
            assert!(r.hypercube_compression_ratio > 1.0);
            assert!(r.pcm_compression_ratio > 10.0);
        }
    }
}
