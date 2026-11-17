//! Spectral analysis, Fourier transforms, sub-bin peak detection,
//! harmonic partial tracking, and stochastic residual modeling.
//!
//! Complies strictly with Demo Integrity Mode: zero external dependencies,
//! 100% pure standard library, safe Rust.

pub mod dft;
pub mod fft;
pub mod peaks;
pub mod residual;
pub mod stft;
pub mod tracking;
pub mod traits;

// Re-exports
pub use dft::Dft;
pub use fft::Fft;
pub use peaks::PeakDetector;
pub use residual::StochasticResidualExtractor;
pub use stft::Stft;
pub use tracking::HarmonicPartialTracker;
pub use traits::{
    FourierTransform, HarmonicPartial, NoiseBandEnvelope, PartialTracker, ResidualAnalyzer,
    SpectralAnalyzer, SpectralFrame, SpectralPeak, SpectralTrajectory,
};
