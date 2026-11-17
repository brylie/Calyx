//! Pure sinusoidal tone generator with exact analytical RMS and Parseval energy.

use crate::error::{DspError, DspResult};
use crate::synthetic::buffer::{KnownPartial, SignalBuffer, SignalKind, SignalMetadata};
use crate::synthetic::traits::SignalGenerator;
use std::f64::consts::{PI, SQRT_2};

/// Analytical sinusoidal signal generator.
#[derive(Debug, Clone, PartialEq)]
pub struct SineGenerator {
    /// Oscillation frequency in Hertz ($f_0$).
    pub frequency: f64,
    /// Peak amplitude ($A \ge 0.0$).
    pub amplitude: f64,
    /// Initial phase in radians ($\phi \in [-\pi, \pi]$).
    pub phase: f64,
}

impl SineGenerator {
    /// Constructs a new sine wave generator.
    #[inline]
    pub fn new(frequency: f64, amplitude: f64, phase: f64) -> Self {
        Self {
            frequency,
            amplitude,
            phase,
        }
    }

    /// Validates parameters against sampling rate and physical limits.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] if:
    /// - `sample_rate <= 0.0` or non-finite.
    /// - `frequency < 0.0` or non-finite.
    /// - `frequency > sample_rate / 2.0` (super-Nyquist).
    /// - `amplitude < 0.0` or non-finite.
    /// - `phase` is non-finite.
    pub fn validate_params(
        frequency: f64,
        amplitude: f64,
        phase: f64,
        sample_rate: f64,
    ) -> DspResult<()> {
        if sample_rate <= 0.0 || !sample_rate.is_finite() {
            return Err(DspError::invalid_parameter(
                "sample_rate",
                "sample_rate must be positive and finite",
            ));
        }
        if frequency < 0.0 || !frequency.is_finite() {
            return Err(DspError::invalid_parameter(
                "frequency",
                "frequency cannot be negative or non-finite",
            ));
        }
        let nyquist = sample_rate / 2.0;
        if frequency > nyquist {
            return Err(DspError::invalid_parameter(
                "frequency",
                format!(
                    "frequency {} Hz exceeds Nyquist limit {} Hz",
                    frequency, nyquist
                ),
            ));
        }
        if amplitude < 0.0 || !amplitude.is_finite() {
            return Err(DspError::invalid_parameter(
                "amplitude",
                "amplitude cannot be negative or non-finite",
            ));
        }
        if !phase.is_finite() {
            return Err(DspError::invalid_parameter(
                "phase",
                "phase must be finite",
            ));
        }
        Ok(())
    }

    /// Theoretical closed-form Root Mean Square (RMS) value: $A / \sqrt{2}$.
    #[inline]
    pub fn theoretical_rms(&self) -> f64 {
        self.amplitude / SQRT_2
    }

    /// Theoretical Parseval energy for $N$ samples over integer cycles: $N \cdot A^2 / 2$.
    #[inline]
    pub fn theoretical_energy(&self, num_samples: usize) -> f64 {
        (num_samples as f64) * self.amplitude * self.amplitude / 2.0
    }
}

impl SignalGenerator for SineGenerator {
    fn generate(&self, num_samples: usize, sample_rate: f64) -> SignalBuffer {
        let mut samples = vec![0.0; num_samples];
        self.generate_into(&mut samples, sample_rate);

        let metadata = SignalMetadata {
            signal_type: SignalKind::Sine,
            fundamental_freq: Some(self.frequency),
            theoretical_rms: Some(self.theoretical_rms()),
            theoretical_energy: Some(self.theoretical_energy(num_samples)),
            expected_rms: Some(self.theoretical_rms()),
            expected_energy: Some(self.theoretical_energy(num_samples)),
            known_partials: vec![KnownPartial::new(
                1,
                self.frequency,
                self.amplitude,
                self.phase,
            )],
            signal_description: format!(
                "Sine wave: f0={} Hz, A={}, phi={}",
                self.frequency, self.amplitude, self.phase
            ),
            ..Default::default()
        };

        SignalBuffer::new(sample_rate, samples, metadata)
    }

    fn generate_into(&self, buffer: &mut [f64], sample_rate: f64) {
        if buffer.is_empty() || sample_rate <= 0.0 {
            return;
        }

        let two_pi_f = 2.0 * PI * self.frequency;
        for (n, s) in buffer.iter_mut().enumerate() {
            let t = n as f64 / sample_rate;
            *s = self.amplitude * (two_pi_f * t + self.phase).sin();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_dc_and_nyquist() {
        let gen_dc = SineGenerator::new(0.0, 1.0, PI / 2.0);
        let buf_dc = gen_dc.generate(10, 1000.0);
        for &s in buf_dc.samples.iter() {
            assert!((s - 1.0).abs() < 1e-12);
        }

        let gen_nyq = SineGenerator::new(500.0, 1.0, PI / 2.0);
        let buf_nyq = gen_nyq.generate(4, 1000.0);
        assert!((buf_nyq.samples[0] - 1.0).abs() < 1e-12);
        assert!((buf_nyq.samples[1] - (-1.0)).abs() < 1e-12);
    }
}
