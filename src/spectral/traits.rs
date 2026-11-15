//! Core spectral traits and data structures for the Calyx DSP framework.
//!
//! Complies strictly with Demo Integrity Mode: zero external dependencies,
//! 100% pure standard library, safe Rust (#![deny(unsafe_code)]).
//!
//! Defines the foundational data models and trait interfaces for:
//! - Fourier transforms ([`FourierTransform`])
//! - Spectral frame and trajectory analysis ([`SpectralAnalyzer`])
//! - Sub-bin spectral peak detection ([`SpectralPeak`])
//! - Harmonic partial tracking ([`HarmonicPartial`], [`PartialTracker`])
//! - Stochastic residual modeling ([`NoiseBandEnvelope`], [`ResidualAnalyzer`])

use crate::error::{DspError, DspResult};
use crate::math::Complex;

// ============================================================================
// Core Spectral Data Structures
// ============================================================================

/// A detected local peak in a discrete Fourier magnitude spectrum.
///
/// Contains discrete bin coordinates alongside continuous sub-bin refined
/// frequency, amplitude, and phase estimates obtained via parabolic interpolation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralPeak {
    /// Discrete FFT / DFT bin index where the local maximum occurs.
    pub bin_index: usize,
    /// Continuous sub-bin refined frequency in Hertz (Hz).
    pub frequency_hz: f64,
    /// Sub-bin refined linear peak amplitude.
    pub amplitude: f64,
    /// Phase angle at the peak in radians within $(-\pi, \pi]$.
    pub phase_rad: f64,
}

impl SpectralPeak {
    /// Construct a new spectral peak.
    #[inline]
    pub fn new(bin_index: usize, frequency_hz: f64, amplitude: f64, phase_rad: f64) -> Self {
        Self {
            bin_index,
            frequency_hz,
            amplitude,
            phase_rad,
        }
    }
}

/// A detected or tracked sinusoidal partial component within a spectral frame.
///
/// Partials represent deterministic harmonic or inharmonic sinusoidal oscillations
/// defined by $s_k(t) = A_k \cos(2\pi f_k t + \phi_k)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicPartial {
    /// 1-based harmonic index (1 for fundamental $f_0$, 2 for 2nd harmonic, etc.,
    /// or 0 for DC / sub-harmonic components).
    pub harmonic_index: usize,
    /// Partial frequency in Hertz (Hz).
    pub frequency: f64,
    /// Linear amplitude of the partial component ($\ge 0.0$).
    pub amplitude: f64,
    /// Instantaneous phase angle in radians within $(-\pi, \pi]$.
    pub phase: f64,
}

impl HarmonicPartial {
    /// Construct a new harmonic partial.
    #[inline]
    pub fn new(harmonic_index: usize, frequency: f64, amplitude: f64, phase: f64) -> Self {
        Self {
            harmonic_index,
            frequency,
            amplitude,
            phase,
        }
    }
}

/// Sub-band noise energies and stochastic-to-deterministic ratio for residual modeling.
///
/// Represents the non-deterministic (stochastic/turbulent) components of an audio signal
/// integrated across perceptual critical filterbanks (e.g., Bark or ERB bands).
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseBandEnvelope {
    /// Perceptual filterbank energies across critical subbands.
    pub band_energies: Vec<f64>,
    /// Stochastic noise energy fraction relative to total frame energy, bounded within $[0.0, 1.0]$.
    pub noise_ratio: f64,
}

impl NoiseBandEnvelope {
    /// Construct a new noise band envelope with given critical band energies and noise ratio.
    #[inline]
    pub fn new(band_energies: Vec<f64>, noise_ratio: f64) -> Self {
        Self {
            band_energies,
            noise_ratio,
        }
    }

    /// Construct an empty noise band envelope with zero bands and 0.0 noise ratio.
    #[inline]
    pub fn empty() -> Self {
        Self {
            band_energies: Vec::new(),
            noise_ratio: 0.0,
        }
    }
}

impl Default for NoiseBandEnvelope {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

/// A time-localized spectral representation containing deterministic partials and stochastic residual.
///
/// Represents the complete parametric state of an audio signal at a discrete analysis window.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralFrame {
    /// Center timestamp of the analysis window in seconds.
    pub timestamp: f64,
    /// Estimated fundamental frequency $f_0$ in Hertz (0.0 if unvoiced or silence).
    pub fundamental_freq: f64,
    /// Extracted deterministic sinusoidal partials for this frame.
    pub partials: Vec<HarmonicPartial>,
    /// Stochastic residual noise envelope across perceptual subbands.
    pub residual: NoiseBandEnvelope,
}

impl SpectralFrame {
    /// Construct a new spectral frame with all fields specified.
    #[inline]
    pub fn new(
        timestamp: f64,
        fundamental_freq: f64,
        partials: Vec<HarmonicPartial>,
        residual: NoiseBandEnvelope,
    ) -> Self {
        Self {
            timestamp,
            fundamental_freq,
            partials,
            residual,
        }
    }

    /// Construct an empty spectral frame at a given timestamp.
    ///
    /// The frame has zero partials, fundamental frequency 0.0, and an empty noise envelope.
    #[inline]
    pub fn empty(timestamp: f64) -> Self {
        Self {
            timestamp,
            fundamental_freq: 0.0,
            partials: Vec::new(),
            residual: NoiseBandEnvelope::empty(),
        }
    }
}

/// Time-series sequence of spectral frames representing a continuous audio segment or gesture.
///
/// Serves as the primary data exchange currency between Spectral Analysis (M3)
/// and Hypercube Decomposition & Synthesis (M4).
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralTrajectory {
    /// Audio sample rate in Hertz (e.g. 44100.0, 48000.0).
    pub sample_rate: f64,
    /// STFT hop size in samples between consecutive frame timestamps.
    pub hop_size: usize,
    /// Sequential collection of analyzed spectral frames.
    pub frames: Vec<SpectralFrame>,
}

impl SpectralTrajectory {
    /// Construct a new spectral trajectory, validating parameters.
    ///
    /// # Errors
    /// Returns `DspError::InvalidParameter` if `sample_rate <= 0.0` or `hop_size == 0`.
    pub fn new(sample_rate: f64, hop_size: usize, frames: Vec<SpectralFrame>) -> DspResult<Self> {
        if sample_rate <= 0.0 || !sample_rate.is_finite() {
            return Err(DspError::invalid_parameter(
                "sample_rate",
                format!("Sample rate must be positive and finite, got {}", sample_rate),
            ));
        }
        if hop_size == 0 {
            return Err(DspError::invalid_parameter(
                "hop_size",
                "Hop size must be greater than zero",
            ));
        }
        Ok(Self {
            sample_rate,
            hop_size,
            frames,
        })
    }

    /// Returns the number of spectral frames in the trajectory.
    #[inline]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` if the trajectory contains no spectral frames.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns the temporal duration of the trajectory in seconds.
    #[inline]
    pub fn duration(&self) -> f64 {
        if self.frames.is_empty() {
            0.0
        } else {
            (self.frames.len() as f64) * (self.hop_size as f64) / self.sample_rate
        }
    }
}

// ============================================================================
// Core Trait Definitions
// ============================================================================

/// Trait for discrete Fourier transform algorithms (direct DFT and fast FFT).
///
/// Provides forward and inverse transforms for real-to-complex, complex-to-complex,
/// and in-place buffer operations.
pub trait FourierTransform {
    /// Perform a forward Fourier transform on real-valued time-domain samples,
    /// returning a full two-sided complex frequency spectrum of length $N$.
    ///
    /// # Errors
    /// - `DspError::EmptyInput` if `input` is empty.
    /// - `DspError::NonPowerOfTwo` if the implementation requires a power-of-two length.
    /// - `DspError::NumericalInstability` if any input sample is NaN or infinite.
    fn forward_real(&self, input: &[f64]) -> DspResult<Vec<Complex<f64>>>;

    /// Perform a forward Fourier transform on complex-valued time-domain samples,
    /// returning a complex spectrum of length $N$.
    ///
    /// # Errors
    /// - `DspError::EmptyInput` if `input` is empty.
    /// - `DspError::NonPowerOfTwo` if the implementation requires a power-of-two length.
    /// - `DspError::NumericalInstability` if any sample is NaN or infinite.
    fn forward_complex(&self, input: &[Complex<f64>]) -> DspResult<Vec<Complex<f64>>>;

    /// Perform an in-place forward Fourier transform on a mutable slice of complex numbers.
    ///
    /// # Errors
    /// - `DspError::EmptyInput` if `buffer` is empty.
    /// - `DspError::NonPowerOfTwo` if the implementation requires a power-of-two length.
    /// - `DspError::NumericalInstability` if any sample is NaN or infinite.
    fn forward_complex_inplace(&self, buffer: &mut [Complex<f64>]) -> DspResult<()>;

    /// Perform an inverse Fourier transform from a complex spectrum,
    /// reconstructing real-valued time-domain samples of length $N$.
    ///
    /// The output is scaled by $1/N$.
    ///
    /// # Errors
    /// - `DspError::EmptyInput` if `spectrum` is empty.
    /// - `DspError::NonPowerOfTwo` if the implementation requires a power-of-two length.
    /// - `DspError::NumericalInstability` if any spectrum bin is NaN or infinite.
    fn inverse_real(&self, spectrum: &[Complex<f64>]) -> DspResult<Vec<f64>>;

    /// Perform an inverse Fourier transform on complex-valued frequency bins,
    /// reconstructing complex time-domain samples of length $N$.
    ///
    /// The output is scaled by $1/N$.
    ///
    /// # Errors
    /// - `DspError::EmptyInput` if `spectrum` is empty.
    /// - `DspError::NonPowerOfTwo` if the implementation requires a power-of-two length.
    /// - `DspError::NumericalInstability` if any spectrum bin is NaN or infinite.
    fn inverse_complex(&self, spectrum: &[Complex<f64>]) -> DspResult<Vec<Complex<f64>>>;

    /// Perform an in-place inverse Fourier transform on a mutable slice of complex numbers.
    ///
    /// The buffer is scaled in-place by $1/N$.
    ///
    /// # Errors
    /// - `DspError::EmptyInput` if `buffer` is empty.
    /// - `DspError::NonPowerOfTwo` if the implementation requires a power-of-two length.
    /// - `DspError::NumericalInstability` if any sample is NaN or infinite.
    fn inverse_complex_inplace(&self, buffer: &mut [Complex<f64>]) -> DspResult<()>;
}

/// Trait for full spectral analysis engines converting raw time-domain audio
/// into a time-varying sequence of harmonic partials and stochastic residuals.
///
/// Satisfies the Milestone 3 to Milestone 4 interface contract.
pub trait SpectralAnalyzer {
    /// Analyze an audio signal buffer at the given sample rate,
    /// producing a complete analyzed [`SpectralTrajectory`].
    ///
    /// # Errors
    /// - `DspError::EmptyInput` if `signal` is empty.
    /// - `DspError::InvalidParameter` if `sample_rate <= 0.0`.
    /// - `DspError::NumericalInstability` if non-finite samples (NaN/Inf) are detected.
    fn analyze(&self, signal: &[f64], sample_rate: f64) -> Result<SpectralTrajectory, DspError>;
}

/// Trait for tracking harmonic partials across spectral frames and trajectories.
pub trait PartialTracker {
    /// Track harmonic partials from a set of detected spectral peaks in a single frame.
    ///
    /// # Arguments
    /// - `peaks`: Slice of detected spectral peaks in the current frame.
    /// - `sample_rate`: Sampling rate in Hz.
    ///
    /// # Returns
    /// A vector of tracked harmonic partials, sorted by harmonic index.
    fn track_partials(&self, peaks: &[SpectralPeak], sample_rate: f64) -> Vec<HarmonicPartial>;

    /// Track harmonic partials across an entire analyzed spectral trajectory.
    ///
    /// # Arguments
    /// - `trajectory`: Analyzed trajectory containing time-stamped spectral frames.
    ///
    /// # Returns
    /// A vector of tracked harmonic partials across the trajectory.
    fn track_trajectory(&self, trajectory: &SpectralTrajectory) -> Vec<HarmonicPartial>;
}

/// Trait for stochastic residual analysis and perceptual filterbank energy extraction.
pub trait ResidualAnalyzer {
    /// Extract the residual noise band envelope by comparing the original magnitude spectrum
    /// against the reconstructed deterministic harmonic spectrum.
    ///
    /// # Arguments
    /// - `original_spectrum`: Magnitude spectrum of the analyzed frame.
    /// - `deterministic_spectrum`: Synthesized magnitude spectrum of the tracked partials.
    ///
    /// # Returns
    /// A [`NoiseBandEnvelope`] containing perceptual subband energies and overall noise ratio.
    fn extract_residual(
        &self,
        original_spectrum: &[f64],
        deterministic_spectrum: &[f64],
    ) -> NoiseBandEnvelope;

    /// Extract perceptual subband envelope from matching original and deterministic magnitude spectra.
    fn extract(&self, original_spectrum: &[f64], deterministic_spectrum: &[f64]) -> NoiseBandEnvelope;

    /// Extract perceptual subband energies across an entire spectral trajectory.
    ///
    /// # Arguments
    /// - `trajectory`: Analyzed trajectory containing time-stamped frames.
    /// - `partials`: Extracted harmonic partials.
    ///
    /// # Returns
    /// A vector of subband energies (one `f64` per filterbank band).
    fn extract_from_trajectory(
        &self,
        trajectory: &SpectralTrajectory,
        partials: &[HarmonicPartial],
    ) -> Vec<f64>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_peak_creation() {
        let peak = SpectralPeak::new(5, 440.0, 0.8, 0.1);
        assert_eq!(peak.bin_index, 5);
        assert!((peak.frequency_hz - 440.0).abs() < 1e-12);
        assert!((peak.amplitude - 0.8).abs() < 1e-12);
        assert!((peak.phase_rad - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_harmonic_partial_creation() {
        let partial = HarmonicPartial::new(1, 440.0, 1.0, 0.0);
        assert_eq!(partial.harmonic_index, 1);
        assert_eq!(partial.frequency, 440.0);
        assert_eq!(partial.amplitude, 1.0);
        assert_eq!(partial.phase, 0.0);
    }

    #[test]
    fn test_noise_band_envelope_defaults() {
        let env = NoiseBandEnvelope::empty();
        assert!(env.band_energies.is_empty());
        assert_eq!(env.noise_ratio, 0.0);

        let default_env = NoiseBandEnvelope::default();
        assert_eq!(env, default_env);
    }

    #[test]
    fn test_spectral_frame_empty() {
        let frame = SpectralFrame::empty(0.5);
        assert_eq!(frame.timestamp, 0.5);
        assert_eq!(frame.fundamental_freq, 0.0);
        assert!(frame.partials.is_empty());
        assert!(frame.residual.band_energies.is_empty());
        assert_eq!(frame.residual.noise_ratio, 0.0);
    }

    #[test]
    fn test_spectral_trajectory_creation_and_methods() {
        let f1 = SpectralFrame::empty(0.0);
        let f2 = SpectralFrame::empty(0.01);
        let traj = SpectralTrajectory::new(44100.0, 256, vec![f1, f2]).unwrap();

        assert_eq!(traj.len(), 2);
        assert!(!traj.is_empty());
        assert!((traj.duration() - (2.0 * 256.0 / 44100.0)).abs() < 1e-12);

        // Parameter validation
        assert!(SpectralTrajectory::new(0.0, 256, vec![]).is_err());
        assert!(SpectralTrajectory::new(-44100.0, 256, vec![]).is_err());
        assert!(SpectralTrajectory::new(44100.0, 0, vec![]).is_err());
    }
}
