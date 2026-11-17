//! Discrete impulses, bandlimited sinc pulses, and damped attack transients.

use crate::error::{DspError, DspResult};
use crate::synthetic::buffer::{SignalBuffer, SignalKind, SignalMetadata};
use crate::synthetic::traits::SignalGenerator;
use std::f64::consts::PI;

/// Internal variants of transient signals.
#[derive(Debug, Clone, PartialEq)]
enum ImpulseType {
    Kronecker {
        delay: usize,
        amplitude: f64,
    },
    BandlimitedSinc {
        cutoff_hz: f64,
        delay: usize,
        amplitude: f64,
    },
    ExponentialPluck {
        frequency_hz: f64,
        decay_rate: f64,
        amplitude: f64,
    },
}

/// Impulse and transient signal generator.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpulseGenerator {
    impulse_type: ImpulseType,
}

impl ImpulseGenerator {
    /// Constructs a Kronecker delta generator at sample index `delay`.
    pub fn kronecker(delay: usize, amplitude: f64) -> Self {
        Self {
            impulse_type: ImpulseType::Kronecker { delay, amplitude },
        }
    }

    /// Constructs a bandlimited sinc pulse centered at sample index `delay`.
    pub fn bandlimited_sinc(cutoff_hz: f64, delay: usize, amplitude: f64) -> Self {
        Self {
            impulse_type: ImpulseType::BandlimitedSinc {
                cutoff_hz,
                delay,
                amplitude,
            },
        }
    }

    /// Constructs an exponentially damped sinusoidal pluck transient.
    pub fn exponential_pluck(frequency_hz: f64, decay_rate: f64, amplitude: f64) -> Self {
        Self {
            impulse_type: ImpulseType::ExponentialPluck {
                frequency_hz,
                decay_rate,
                amplitude,
            },
        }
    }

    /// Generates audio with strict bounds verification on delay and parameters.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] if delay is out of bounds or parameters are invalid.
    pub fn generate_checked(&self, num_samples: usize, sample_rate: f64) -> DspResult<SignalBuffer> {
        if sample_rate <= 0.0 || !sample_rate.is_finite() {
            return Err(DspError::invalid_parameter(
                "sample_rate",
                "sample_rate must be positive and finite",
            ));
        }

        match &self.impulse_type {
            ImpulseType::Kronecker { delay, .. } => {
                if *delay >= num_samples {
                    return Err(DspError::invalid_parameter(
                        "delay",
                        format!("delay index {} exceeds buffer length {}", delay, num_samples),
                    ));
                }
            }
            ImpulseType::BandlimitedSinc { delay, cutoff_hz, .. } => {
                if *delay >= num_samples {
                    return Err(DspError::invalid_parameter(
                        "delay",
                        format!("delay index {} exceeds buffer length {}", delay, num_samples),
                    ));
                }
                if *cutoff_hz <= 0.0 || !cutoff_hz.is_finite() {
                    return Err(DspError::invalid_parameter(
                        "bandwidth",
                        "bandwidth must be positive",
                    ));
                }
            }
            ImpulseType::ExponentialPluck { decay_rate, .. } => {
                if *decay_rate <= 0.0 || !decay_rate.is_finite() {
                    return Err(DspError::invalid_parameter(
                        "decay_tau",
                        "decay rate must be positive",
                    ));
                }
            }
        }

        Ok(self.generate(num_samples, sample_rate))
    }

    /// Direct helper constructing a Kronecker delta vector with strict bounds check.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] if `delay >= length` or [`DspError::EmptyInput`] if `length == 0`.
    pub fn kronecker_delta(delay: usize, length: usize, amplitude: f64) -> DspResult<Vec<f64>> {
        if length == 0 {
            return Err(DspError::EmptyInput);
        }
        if delay >= length {
            return Err(DspError::invalid_parameter(
                "delay",
                format!("delay {} exceeds buffer length {}", delay, length),
            ));
        }
        let mut buffer = vec![0.0; length];
        buffer[delay] = amplitude;
        Ok(buffer)
    }

    /// Direct helper constructing a bandlimited sinc vector with bandwidth validation.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] if `bandwidth <= 0.0` or `sample_rate <= 0.0`.
    pub fn bandlimited_sinc_checked(
        bandwidth: f64,
        delay: usize,
        amplitude: f64,
        sample_rate: f64,
    ) -> DspResult<Vec<f64>> {
        if bandwidth <= 0.0 || !bandwidth.is_finite() {
            return Err(DspError::invalid_parameter(
                "bandwidth",
                "bandwidth must be positive",
            ));
        }
        if sample_rate <= 0.0 || !sample_rate.is_finite() {
            return Err(DspError::invalid_parameter(
                "sample_rate",
                "sample_rate must be positive and finite",
            ));
        }
        let length = (delay * 2).max(512);
        let gen = Self::bandlimited_sinc(bandwidth, delay, amplitude);
        Ok(gen.generate(length, sample_rate).samples)
    }

    /// Direct helper constructing a damped transient vector with decay rate validation.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] if `decay_tau <= 0.0` or `sample_rate <= 0.0`.
    pub fn damped_transient_checked(
        f0: f64,
        decay_tau: f64,
        length: usize,
        sample_rate: f64,
    ) -> DspResult<Vec<f64>> {
        if decay_tau <= 0.0 || !decay_tau.is_finite() {
            return Err(DspError::invalid_parameter(
                "decay_tau",
                "decay_tau must be positive and finite",
            ));
        }
        if f0 < 0.0 || !f0.is_finite() {
            return Err(DspError::invalid_parameter(
                "f0",
                "f0 cannot be negative or non-finite",
            ));
        }
        if sample_rate <= 0.0 || !sample_rate.is_finite() {
            return Err(DspError::invalid_parameter(
                "sample_rate",
                "sample_rate must be positive and finite",
            ));
        }
        let decay_rate = 1.0 / decay_tau;
        let gen = Self::exponential_pluck(f0, decay_rate, 1.0);
        Ok(gen.generate(length, sample_rate).samples)
    }
}

impl SignalGenerator for ImpulseGenerator {
    fn generate(&self, num_samples: usize, sample_rate: f64) -> SignalBuffer {
        let mut samples = vec![0.0; num_samples];
        self.generate_into(&mut samples, sample_rate);

        let metadata = SignalMetadata {
            signal_type: SignalKind::Impulse,
            signal_description: format!("Impulse transient: {:?}", self.impulse_type),
            ..Default::default()
        };

        SignalBuffer::new(sample_rate, samples, metadata)
    }

    fn generate_into(&self, buffer: &mut [f64], sample_rate: f64) {
        if buffer.is_empty() {
            return;
        }

        buffer.fill(0.0);

        match self.impulse_type {
            ImpulseType::Kronecker { delay, amplitude } => {
                if delay < buffer.len() {
                    buffer[delay] = amplitude;
                }
            }
            ImpulseType::BandlimitedSinc {
                cutoff_hz,
                delay,
                amplitude,
            } => {
                if sample_rate <= 0.0 {
                    return;
                }
                let fc_ratio = cutoff_hz / sample_rate;
                for (n, s) in buffer.iter_mut().enumerate() {
                    let diff = n as f64 - delay as f64;
                    let x = 2.0 * PI * fc_ratio * diff;
                    *s = if x.abs() < 1e-12 {
                        amplitude
                    } else {
                        amplitude * (x.sin() / x)
                    };
                }
            }
            ImpulseType::ExponentialPluck {
                frequency_hz,
                decay_rate,
                amplitude,
            } => {
                if sample_rate <= 0.0 {
                    return;
                }
                let two_pi_f = 2.0 * PI * frequency_hz;
                for (n, s) in buffer.iter_mut().enumerate() {
                    let t = n as f64 / sample_rate;
                    *s = amplitude * (-decay_rate * t).exp() * (two_pi_f * t).sin();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kronecker_delta() {
        let delta = ImpulseGenerator::kronecker_delta(5, 10, 1.0).unwrap();
        assert_eq!(delta[5], 1.0);
        assert_eq!(delta[4], 0.0);
        assert_eq!(delta[6], 0.0);
    }
}
