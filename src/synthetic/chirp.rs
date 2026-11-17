//! Continuous linear and logarithmic frequency sweep generators.

use crate::error::{DspError, DspResult};
use crate::synthetic::buffer::{SignalBuffer, SignalKind, SignalMetadata};
use crate::synthetic::traits::SignalGenerator;
use std::f64::consts::PI;

/// Type of frequency trajectory during a chirp sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChirpKind {
    /// Linear frequency progression: $f(t) = f_0 + k t$.
    #[default]
    Linear,
    /// Logarithmic (geometric / exponential) progression: $f(t) = f_0 (f_1 / f_0)^{t/T}$.
    Logarithmic,
}

/// Continuous frequency sweep generator.
#[derive(Debug, Clone, PartialEq)]
pub struct ChirpGenerator {
    /// Starting instantaneous frequency in Hertz ($f_{\text{start}}$).
    pub f_start: f64,
    /// Ending instantaneous frequency in Hertz ($f_{\text{end}}$).
    pub f_end: f64,
    /// Peak amplitude ($A \ge 0.0$).
    pub amplitude: f64,
    /// Target sweep duration in seconds ($T > 0.0$).
    pub duration: f64,
    /// Trajectory profile (linear or logarithmic).
    pub kind: ChirpKind,
}

impl ChirpGenerator {
    /// Constructs a new chirp generator.
    pub fn new(
        f_start: f64,
        f_end: f64,
        amplitude: f64,
        duration: f64,
        kind: ChirpKind,
    ) -> Self {
        Self {
            f_start,
            f_end,
            amplitude,
            duration,
            kind,
        }
    }

    /// Validates sweep parameters against mathematical and domain constraints.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] if:
    /// - `duration <= 0.0` or non-finite.
    /// - `f_start < 0.0` or `f_end < 0.0` or non-finite.
    /// - `kind == ChirpKind::Logarithmic` and `(f_start <= 0.0 || f_end <= 0.0)`.
    /// - `amplitude < 0.0` or non-finite.
    pub fn validate_params(
        f_start: f64,
        f_end: f64,
        amplitude: f64,
        duration: f64,
        kind: ChirpKind,
    ) -> DspResult<()> {
        if duration <= 0.0 || !duration.is_finite() {
            return Err(DspError::invalid_parameter(
                "duration",
                "duration must be positive and finite",
            ));
        }
        if f_start < 0.0 || !f_start.is_finite() {
            return Err(DspError::invalid_parameter(
                "f_start",
                "f_start cannot be negative or non-finite",
            ));
        }
        if f_end < 0.0 || !f_end.is_finite() {
            return Err(DspError::invalid_parameter(
                "f_end",
                "f_end cannot be negative or non-finite",
            ));
        }
        if amplitude < 0.0 || !amplitude.is_finite() {
            return Err(DspError::invalid_parameter(
                "amplitude",
                "amplitude cannot be negative or non-finite",
            ));
        }
        if kind == ChirpKind::Logarithmic && (f_start <= 0.0 || f_end <= 0.0) {
            return Err(DspError::invalid_parameter(
                "f_start",
                "logarithmic chirp requires strictly positive start and end frequencies",
            ));
        }
        Ok(())
    }
}

impl SignalGenerator for ChirpGenerator {
    fn generate(&self, num_samples: usize, sample_rate: f64) -> SignalBuffer {
        let mut samples = vec![0.0; num_samples];
        self.generate_into(&mut samples, sample_rate);

        let metadata = SignalMetadata {
            signal_type: SignalKind::Chirp,
            sweep_range: Some((self.f_start, self.f_end)),
            signal_description: format!(
                "Chirp {:?} from {} Hz to {} Hz, duration {} s",
                self.kind, self.f_start, self.f_end, self.duration
            ),
            ..Default::default()
        };

        SignalBuffer::new(sample_rate, samples, metadata)
    }

    fn generate_into(&self, buffer: &mut [f64], sample_rate: f64) {
        if buffer.is_empty() || sample_rate <= 0.0 || self.duration <= 0.0 {
            return;
        }

        match self.kind {
            ChirpKind::Linear => {
                if (self.f_end - self.f_start).abs() < 1e-12 {
                    // Fall back to pure sine
                    let two_pi_f = 2.0 * PI * self.f_start;
                    for (n, s) in buffer.iter_mut().enumerate() {
                        let t = n as f64 / sample_rate;
                        *s = self.amplitude * (two_pi_f * t).sin();
                    }
                } else {
                    let k = (self.f_end - self.f_start) / self.duration;
                    for (n, s) in buffer.iter_mut().enumerate() {
                        let t = n as f64 / sample_rate;
                        let phase = 2.0 * PI * (self.f_start * t + 0.5 * k * t * t);
                        *s = self.amplitude * phase.sin();
                    }
                }
            }
            ChirpKind::Logarithmic => {
                if (self.f_end - self.f_start).abs() < 1e-12 || self.f_start <= 0.0 || self.f_end <= 0.0 {
                    // Fall back to pure sine
                    let two_pi_f = 2.0 * PI * self.f_start;
                    for (n, s) in buffer.iter_mut().enumerate() {
                        let t = n as f64 / sample_rate;
                        *s = self.amplitude * (two_pi_f * t).sin();
                    }
                } else {
                    let ratio = self.f_end / self.f_start;
                    let factor = 2.0 * PI * self.f_start * self.duration / ratio.ln();
                    for (n, s) in buffer.iter_mut().enumerate() {
                        let t = n as f64 / sample_rate;
                        let norm_t = t / self.duration;
                        let phase = factor * (ratio.powf(norm_t) - 1.0);
                        *s = self.amplitude * phase.sin();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chirp_identical_frequencies() {
        let chirp = ChirpGenerator::new(440.0, 440.0, 1.0, 1.0, ChirpKind::Linear);
        let buf = chirp.generate(100, 48000.0);
        let sine_expected = 1.0 * (2.0 * PI * 440.0 * (10.0 / 48000.0)).sin();
        assert!((buf.samples[10] - sine_expected).abs() < 1e-12);
    }
}
