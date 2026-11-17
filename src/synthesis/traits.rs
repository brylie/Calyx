//! Trait definitions for spectral reconstruction and acoustic resynthesis.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library safe Rust.

use crate::error::DspResult;
use crate::spectral::{HarmonicPartial, NoiseBandEnvelope, SpectralFrame, SpectralTrajectory};

/// Trait for spectral synthesis engines generating audio from spectral frames or trajectories.
pub trait SpectralSynthesizer: Send + Sync {
    /// Synthesizes time-domain samples from a single spectral frame.
    fn synthesize_frame(&self, frame: &SpectralFrame, num_samples: usize) -> Vec<f64>;

    /// Synthesizes smooth time-domain transition between two spectral frames with linear ramping.
    fn synthesize_transition(&self, f1: &SpectralFrame, f2: &SpectralFrame, num_samples: usize) -> Vec<f64>;

    /// Synthesizes continuous audio from stationary harmonic partials.
    fn synthesize_trajectory(&self, partials: &[HarmonicPartial], num_samples: usize) -> Vec<f64>;
}

/// Trait for full signal reconstruction engines converting spectral trajectories into time-domain audio.
pub trait SignalReconstructor: Send + Sync {
    /// Reconstructs time-domain audio of the requested length from a spectral trajectory.
    fn reconstruct(
        &self,
        trajectory: &SpectralTrajectory,
        num_samples: usize,
        sample_rate: f64,
    ) -> DspResult<Vec<f64>>;
}

/// Trait for phase modeling strategies (harmonic locking, continuous vocoder, dispersion).
pub trait PhaseModel: Send + Sync {
    /// Computes or adjusts the phase for a given harmonic partial.
    fn compute_phase(&self, phase: f64, harmonic_index: usize, fundamental_phase: f64) -> f64;
}

/// Trait for stochastic noise resynthesis matching residual filterbank envelopes.
pub trait NoiseSynthesizer: Send + Sync {
    /// Synthesizes stochastic residual noise for a single frame.
    fn synthesize_frame(&self, env: &NoiseBandEnvelope, num_samples: usize) -> DspResult<Vec<f64>>;
}
