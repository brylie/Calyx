//! Signal buffer and analytical ground-truth metadata containers.

pub use crate::spectral::{HarmonicPartial, NoiseBandEnvelope, SpectralFrame, SpectralPeak};
use crate::synthetic::noise::NoiseColor;

/// Classification of synthetic signal generators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SignalKind {
    /// Unspecified or composite signal.
    #[default]
    Unknown,
    /// Single sinusoidal tone.
    Sine,
    /// Additive harmonic series.
    HarmonicComplex,
    /// Continuous frequency sweep.
    Chirp,
    /// Discrete impulse, sinc pulse, or pluck transient.
    Impulse,
    /// Stochastic pseudorandom noise.
    Noise,
    /// Parameterized acoustic instrument note.
    InstrumentNote,
}

/// Description of a known analytical harmonic partial.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KnownPartial {
    /// Harmonic order index ($k \ge 1$, where 1 is fundamental).
    pub harmonic_index: usize,
    /// Analytical frequency in Hertz.
    pub frequency_hz: f64,
    /// Linear peak amplitude ($A_k \ge 0.0$).
    pub amplitude: f64,
    /// Initial phase offset in radians ($\phi_k$).
    pub phase: f64,
}

impl KnownPartial {
    /// Creates a new known harmonic partial descriptor.
    #[inline]
    pub fn new(harmonic_index: usize, frequency_hz: f64, amplitude: f64, phase: f64) -> Self {
        Self {
            harmonic_index,
            frequency_hz,
            amplitude,
            phase,
        }
    }
}

/// Type alias for analytical partials.
pub type AnalyticalPartial = KnownPartial;

impl From<KnownPartial> for HarmonicPartial {
    #[inline]
    fn from(k: KnownPartial) -> Self {
        HarmonicPartial::new(k.harmonic_index, k.frequency_hz, k.amplitude, k.phase)
    }
}

impl From<HarmonicPartial> for KnownPartial {
    #[inline]
    fn from(h: HarmonicPartial) -> Self {
        KnownPartial::new(h.harmonic_index, h.frequency, h.amplitude, h.phase)
    }
}

/// Analytical ground truth metadata attached to a synthetic signal.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SignalMetadata {
    /// Signal generator classification.
    pub signal_type: SignalKind,
    /// Known fundamental frequency $f_0$ in Hertz, if periodic.
    pub fundamental_freq: Option<f64>,
    /// Exact theoretical root-mean-square value.
    pub theoretical_rms: Option<f64>,
    /// Exact theoretical Parseval energy.
    pub theoretical_energy: Option<f64>,
    /// Alias for expected root-mean-square value.
    pub expected_rms: Option<f64>,
    /// Alias for expected Parseval energy.
    pub expected_energy: Option<f64>,
    /// Analytical harmonic partials present in the signal.
    pub known_partials: Vec<KnownPartial>,
    /// Inharmonicity coefficient $B$, if applicable.
    pub inharmonicity_b: Option<f64>,
    /// Frequency sweep bounds $(f_{\text{start}}, f_{\text{end}})$ in Hertz.
    pub sweep_range: Option<(f64, f64)>,
    /// Stochastic noise color, if applicable.
    pub noise_color: Option<NoiseColor>,
    /// 5D hypercube coordinate point $(t, p, d, a, v)$ if generated from an instrument model.
    pub hypercube_coords: Option<[f64; 5]>,
    /// Human-readable description of the synthetic signal.
    pub signal_description: String,
}

/// Time-domain audio buffer holding sample values and analytical ground truth.
#[derive(Debug, Clone, PartialEq)]
pub struct SignalBuffer {
    /// Sampling rate in Hertz.
    pub sample_rate: f64,
    /// Contiguous floating-point time-domain audio samples.
    pub samples: Vec<f64>,
    /// Attached analytical metadata.
    pub metadata: SignalMetadata,
}

impl SignalBuffer {
    /// Constructs a new signal buffer.
    #[inline]
    pub fn new(sample_rate: f64, samples: Vec<f64>, metadata: SignalMetadata) -> Self {
        Self {
            sample_rate,
            samples,
            metadata,
        }
    }

    /// Constructs an empty signal buffer with the specified sample rate.
    #[inline]
    pub fn empty(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            samples: Vec::new(),
            metadata: SignalMetadata::default(),
        }
    }

    /// Number of audio samples in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the buffer contains zero audio samples.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Duration of the audio buffer in seconds ($N / f_s$).
    #[inline]
    pub fn duration(&self) -> f64 {
        if self.sample_rate <= 0.0 {
            0.0
        } else {
            self.samples.len() as f64 / self.sample_rate
        }
    }

    /// Alias for [`duration`].
    #[inline]
    pub fn duration_seconds(&self) -> f64 {
        self.duration()
    }

    /// Computes total discrete energy: $\sum_{n=0}^{N-1} s[n]^2$.
    #[inline]
    pub fn energy(&self) -> f64 {
        self.samples.iter().map(|&s| s * s).sum()
    }

    /// Alias for [`energy`].
    #[inline]
    pub fn compute_energy(&self) -> f64 {
        self.energy()
    }

    /// Computes Root Mean Square (RMS) value: $\sqrt{\frac{1}{N} \sum s[n]^2}$.
    #[inline]
    pub fn rms(&self) -> f64 {
        if self.samples.is_empty() {
            0.0
        } else {
            (self.energy() / (self.samples.len() as f64)).sqrt()
        }
    }

    /// Alias for [`rms`].
    #[inline]
    pub fn compute_rms(&self) -> f64 {
        self.rms()
    }

    /// Computes absolute peak sample value: $\max_n |s[n]|$.
    #[inline]
    pub fn peak_amplitude(&self) -> f64 {
        self.samples
            .iter()
            .fold(0.0_f64, |acc, &s| acc.max(s.abs()))
    }

    /// Borrows the audio samples as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.samples
    }
}

impl std::ops::Deref for SignalBuffer {
    type Target = [f64];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_buffer_properties() {
        let samples = vec![0.5, -0.5, 0.5, -0.5];
        let buf = SignalBuffer::new(4.0, samples, SignalMetadata::default());
        assert_eq!(buf.len(), 4);
        assert!(!buf.is_empty());
        assert_eq!(buf.duration(), 1.0);
        assert_eq!(buf.energy(), 1.0);
        assert_eq!(buf.rms(), 0.5);
        assert_eq!(buf.peak_amplitude(), 0.5);
        assert_eq!(buf[0], 0.5);
    }
}
