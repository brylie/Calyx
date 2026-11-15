//! Composable DSP pipeline chaining Analysis, Hypercube Decomposition, and Reconstruction.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library safe Rust.

use crate::error::{DspError, DspResult};
use crate::hypercube::{HypercubeDecomposer, PcaDecomposer};
use crate::math::WindowType;
use crate::metrics::report::QualityTierConfig;
use crate::spectral::stft::Stft;
use crate::spectral::traits::{SpectralAnalyzer, SpectralTrajectory};
use crate::synthesis::ola::AdditiveOlaReconstructor;
use crate::synthesis::traits::SignalReconstructor;

/// Encapsulated result of an end-to-end DSP pipeline execution.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineResult {
    /// Reconstructed time-domain audio samples.
    pub reconstructed_audio: Vec<f64>,
    /// Intermediate analyzed spectral trajectory.
    pub analyzed_trajectory: Option<SpectralTrajectory>,
    /// Intermediate reconstructed spectral trajectory.
    pub reconstructed_trajectory: Option<SpectralTrajectory>,
}

/// Composable end-to-end DSP pipeline chaining Analysis -> Hypercube Decomposition -> Resynthesis.
#[derive(Debug, Clone)]
pub struct DspPipeline<
    A: SpectralAnalyzer = Stft,
    D: HypercubeDecomposer = PcaDecomposer,
    R: SignalReconstructor = AdditiveOlaReconstructor,
> {
    /// Spectral analysis stage.
    pub analyzer: A,
    /// Hypercube decomposition stage.
    pub decomposer: D,
    /// Signal resynthesis stage.
    pub reconstructor: R,
    /// Quality tier configuration.
    pub config: QualityTierConfig,
}

impl DspPipeline<Stft, PcaDecomposer, AdditiveOlaReconstructor> {
    /// Constructs default Reference Quality Tier pipeline (mastering fidelity, rank 8).
    pub fn default_reference() -> Self {
        Self::from_config(QualityTierConfig::reference())
    }

    /// Constructs default Production Quality Tier pipeline (standard production fidelity, rank 4).
    pub fn default_production() -> Self {
        Self::from_config(QualityTierConfig::production())
    }

    /// Constructs default Preview Quality Tier pipeline (high compression, rank 2).
    pub fn default_preview() -> Self {
        Self::from_config(QualityTierConfig::preview())
    }

    /// Constructs a pipeline instance from a QualityTierConfig.
    pub fn from_config(config: QualityTierConfig) -> Self {
        let (win, hop) = match config.rank {
            r if r >= 8 => (1024, 256),
            r if r >= 4 => (512, 128),
            _ => (512, 256),
        };

        let analyzer = Stft::new(win, hop, WindowType::Hann)
            .expect("STFT analyzer creation")
            .with_noise_bands(config.num_noise_bands);
        let decomposer = PcaDecomposer::new(config.rank);
        let reconstructor = AdditiveOlaReconstructor::new(win, hop);

        Self {
            analyzer,
            decomposer,
            reconstructor,
            config,
        }
    }
}

impl<A: SpectralAnalyzer, D: HypercubeDecomposer, R: SignalReconstructor> DspPipeline<A, D, R> {
    /// Constructs a customized DSP pipeline with swappable backend stages.
    pub fn new(analyzer: A, decomposer: D, reconstructor: R, config: QualityTierConfig) -> Self {
        Self {
            analyzer,
            decomposer,
            reconstructor,
            config,
        }
    }

    /// Runs the end-to-end analysis, decomposition, and resynthesis pipeline on audio input.
    ///
    /// # Errors
    /// Returns `DspError::EmptyInput` if `input.is_empty()`.
    /// Returns `DspError::InvalidParameter` if `sample_rate <= 0.0` or non-finite.
    /// Returns `DspError::NumericalInstability` if any sample is NaN or Infinite.
    pub fn run(&self, input: &[f64], sample_rate: f64) -> DspResult<PipelineResult> {
        if input.is_empty() {
            return Err(DspError::EmptyInput);
        }

        if sample_rate <= 0.0 || !sample_rate.is_finite() {
            return Err(DspError::invalid_parameter(
                "sample_rate",
                format!("Sample rate must be positive and finite, got {}", sample_rate),
            ));
        }

        for (i, &s) in input.iter().enumerate() {
            if !s.is_finite() {
                return Err(DspError::numerical_instability(format!(
                    "Pipeline input sample at index {} is not finite: {}",
                    i, s
                )));
            }
        }

        // Pure silence fast-path
        if input.iter().all(|&s| s.abs() < 1e-12) {
            return Ok(PipelineResult {
                reconstructed_audio: vec![0.0; input.len()],
                analyzed_trajectory: None,
                reconstructed_trajectory: None,
            });
        }

        // Stage 1: Spectral Analysis
        let analyzed_trajectory = self.analyzer.analyze(input, sample_rate)?;

        // Stage 2: Hypercube Decomposition & Trajectory Reconstruction
        let compressed = self.decomposer.decompose(&analyzed_trajectory)?;
        let reconstructed_trajectory = self.decomposer.reconstruct(&compressed)?;

        // Stage 3: Resynthesis
        let reconstructed_audio = self.reconstructor.reconstruct(
            &reconstructed_trajectory,
            input.len(),
            sample_rate,
        )?;

        Ok(PipelineResult {
            reconstructed_audio,
            analyzed_trajectory: Some(analyzed_trajectory),
            reconstructed_trajectory: Some(reconstructed_trajectory),
        })
    }

    /// Resets internal pipeline state and buffers.
    pub fn reset(&mut self) {
        // Clear state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_empty_input() {
        let pipeline = DspPipeline::default_reference();
        assert_eq!(pipeline.run(&[], 44100.0), Err(DspError::EmptyInput));
    }

    #[test]
    fn test_pipeline_zero_sample_rate() {
        let pipeline = DspPipeline::default_reference();
        assert!(pipeline.run(&[0.1; 100], 0.0).is_err());
    }

    #[test]
    fn test_pipeline_pure_silence() {
        let pipeline = DspPipeline::default_reference();
        let silence = vec![0.0; 1024];
        let res = pipeline.run(&silence, 44100.0).unwrap();
        assert!(res.reconstructed_audio.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_pipeline_nan_rejection() {
        let pipeline = DspPipeline::default_reference();
        let bad = vec![f64::NAN; 512];
        assert!(pipeline.run(&bad, 44100.0).is_err());
    }
}
