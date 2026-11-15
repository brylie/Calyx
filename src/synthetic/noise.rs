//! Deterministic pseudo-random noise generator using first-principles PCG32.

use crate::error::{DspError, DspResult};
use crate::synthetic::buffer::{SignalBuffer, SignalKind, SignalMetadata};
use crate::synthetic::traits::SignalGenerator;
use std::f64::consts::PI;

/// Spectral distribution / coloration of generated noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NoiseColor {
    /// White uniform distribution in $[-A, A]$.
    #[default]
    WhiteUniform,
    /// White Gaussian normal distribution $\mathcal{N}(0, A^2)$ via Box-Muller.
    WhiteGaussian,
    /// Alias for [`NoiseColor::WhiteGaussian`].
    Gaussian,
    /// Pink noise ($1/f$, $-3$ dB/octave) via Paul Kellet 3-pole filter.
    Pink,
    /// Brownian / Red noise ($1/f^2$, $-6$ dB/octave) via leaky integrator.
    Brownian,
}

/// First-principles PCG32 pseudo-random number generator (64-bit state, XSH-RR output).
#[derive(Debug, Clone)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Constructs a new PCG32 generator with state and sequence stream selector.
    pub fn new(init_state: u64, init_seq: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (init_seq << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(init_state);
        rng.next_u32();
        rng
    }

    /// Advances the internal LCG state and produces a 32-bit pseudo-random integer.
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(self.inc);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Produces a uniformly distributed floating-point value in $[0.0, 1.0)$.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64) / 4294967296.0
    }

    /// Produces a symmetric uniformly distributed floating-point value in $[-1.0, 1.0)$.
    #[inline]
    pub fn next_symmetric_f64(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }
}

/// Deterministic, seedable synthetic noise generator.
#[derive(Debug, Clone)]
pub struct NoiseGenerator {
    /// Noise spectral color.
    pub color: NoiseColor,
    /// Peak or standard-deviation amplitude scale.
    pub amplitude: f64,
    /// Initial 64-bit seed.
    pub seed: u64,
}

impl NoiseGenerator {
    /// Constructs a new noise generator.
    pub fn new(color: NoiseColor, amplitude: f64, seed: u64) -> Self {
        Self {
            color,
            amplitude,
            seed,
        }
    }

    /// Validates generator parameters against physical bounds.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] if `amplitude < 0.0` or non-finite.
    pub fn validate_params(color: NoiseColor, amplitude: f64, _seed: u64) -> DspResult<()> {
        let _ = color;
        if amplitude < 0.0 || !amplitude.is_finite() {
            return Err(DspError::invalid_parameter(
                "amplitude",
                "amplitude cannot be negative or non-finite",
            ));
        }
        Ok(())
    }
}

impl SignalGenerator for NoiseGenerator {
    fn generate(&self, num_samples: usize, sample_rate: f64) -> SignalBuffer {
        let mut samples = vec![0.0; num_samples];
        self.generate_into(&mut samples, sample_rate);

        let metadata = SignalMetadata {
            signal_type: SignalKind::Noise,
            noise_color: Some(self.color),
            signal_description: format!("Noise color={:?} amplitude={} seed={}", self.color, self.amplitude, self.seed),
            ..Default::default()
        };

        SignalBuffer::new(sample_rate, samples, metadata)
    }

    fn generate_into(&self, buffer: &mut [f64], _sample_rate: f64) {
        if buffer.is_empty() {
            return;
        }

        if self.amplitude == 0.0 {
            buffer.fill(0.0);
            return;
        }

        let mut rng = Pcg32::new(self.seed, 54);

        match self.color {
            NoiseColor::WhiteUniform => {
                for sample in buffer.iter_mut() {
                    *sample = self.amplitude * rng.next_symmetric_f64();
                }
            }
            NoiseColor::WhiteGaussian | NoiseColor::Gaussian => {
                let n = buffer.len();
                let mut i = 0;
                while i < n {
                    let mut u1 = rng.next_f64();
                    if u1 < 1e-15 {
                        u1 = 1e-15;
                    }
                    let u2 = rng.next_f64();
                    let radius = (-2.0 * u1.ln()).sqrt();
                    let theta = 2.0 * PI * u2;

                    buffer[i] = self.amplitude * radius * theta.cos();
                    if i + 1 < n {
                        buffer[i + 1] = self.amplitude * radius * theta.sin();
                    }
                    i += 2;
                }
            }
            NoiseColor::Pink => {
                // Paul Kellet's refined 3-pole filter with 4 state variables
                let mut b0 = 0.0_f64;
                let mut b1 = 0.0_f64;
                let mut b2 = 0.0_f64;
                for sample in buffer.iter_mut() {
                    let white = rng.next_symmetric_f64();
                    b0 = 0.99886 * b0 + 0.0555179 * white;
                    b1 = 0.99332 * b1 + 0.0750759 * white;
                    b2 = 0.96900 * b2 + 0.1538520 * white;
                    let pink = (b0 + b1 + b2 + 0.5362 * white) * 0.25;
                    *sample = self.amplitude * pink.clamp(-1.0, 1.0);
                }
            }
            NoiseColor::Brownian => {
                // Single-pole leaky integrator
                let mut state = 0.0_f64;
                for sample in buffer.iter_mut() {
                    let white = rng.next_symmetric_f64();
                    state = 0.99 * state + 0.1 * white;
                    state = state.clamp(-1.0, 1.0);
                    *sample = self.amplitude * state;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_reproducibility() {
        let gen1 = NoiseGenerator::new(NoiseColor::WhiteUniform, 0.8, 12345);
        let gen2 = NoiseGenerator::new(NoiseColor::WhiteUniform, 0.8, 12345);
        let b1 = gen1.generate(100, 48000.0);
        let b2 = gen2.generate(100, 48000.0);
        assert_eq!(b1.samples, b2.samples);
    }
}
