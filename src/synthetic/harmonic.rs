//! Multi-tone additive harmonic series with inharmonicity, recipes, and damping envelopes.

use crate::error::{DspError, DspResult};
use crate::synthetic::buffer::{KnownPartial, SignalBuffer, SignalKind, SignalMetadata};
use crate::synthetic::traits::SignalGenerator;
use std::f64::consts::PI;

/// Spectral amplitude weighting recipe for harmonic partials.
#[derive(Debug, Clone, PartialEq)]
pub enum HarmonicRecipe {
    /// Sawtooth wave: all integer harmonics with $A_k = A_0 / k$.
    Sawtooth,
    /// Square wave: odd integer harmonics only ($k \in \{1, 3, 5, \dots\}$) with $A_k = A_0 / k$.
    Square,
    /// Triangle wave: odd integer harmonics with alternating signs and $A_k = A_0 / k^2$.
    Triangle,
    /// Geometric decay: $A_k = A_0 \cdot \text{ratio}^{k-1}$.
    GeometricDecay {
        /// Amplitude decay ratio per harmonic step ($0.0 \le \text{ratio} \le 1.0$).
        ratio: f64,
    },
    /// Custom relative amplitude weights per harmonic index.
    Custom(Vec<f64>),
}

/// Synthesizer for multi-partial harmonic complex tones with stiff-string inharmonicity.
#[derive(Debug, Clone, PartialEq)]
pub struct HarmonicComplexGenerator {
    /// Nominal fundamental frequency $f_0$ in Hertz.
    pub f0: f64,
    /// Base peak amplitude ($A_0$).
    pub base_amplitude: f64,
    /// Maximum number of partials to generate before Nyquist culling.
    pub num_partials: usize,
    /// Stiff-string inharmonicity coefficient ($B \ge 0$).
    pub inharmonicity_b: f64,
    /// Amplitude recipe across partials.
    pub recipe: HarmonicRecipe,
    /// Base temporal exponential decay time constant in seconds ($\tau_0$).
    pub damping_tau: Option<f64>,
}

impl HarmonicComplexGenerator {
    /// Constructs a new harmonic complex generator.
    pub fn new(
        f0: f64,
        base_amplitude: f64,
        num_partials: usize,
        inharmonicity_b: f64,
        recipe: HarmonicRecipe,
    ) -> Self {
        Self {
            f0,
            base_amplitude,
            num_partials,
            inharmonicity_b,
            recipe,
            damping_tau: None,
        }
    }

    /// Sets or clears the frequency-dependent damping envelope time constant.
    pub fn set_damping_tau(&mut self, tau: Option<f64>) {
        self.damping_tau = tau;
    }

    /// Validates the harmonic recipe configuration.
    ///
    /// # Errors
    /// Returns [`DspError::EmptyInput`] or [`DspError::InvalidParameter`] if weights are empty
    /// or geometric ratio is outside $[0.0, 1.0]$.
    pub fn validate_recipe(recipe: &HarmonicRecipe) -> DspResult<()> {
        match recipe {
            HarmonicRecipe::Custom(weights) => {
                if weights.is_empty() {
                    return Err(DspError::EmptyInput);
                }
                for &w in weights {
                    if !w.is_finite() {
                        return Err(DspError::invalid_parameter(
                            "recipe",
                            "weights must be finite",
                        ));
                    }
                }
            }
            HarmonicRecipe::GeometricDecay { ratio } => {
                if *ratio < 0.0 || *ratio > 1.0 || !ratio.is_finite() {
                    return Err(DspError::invalid_parameter(
                        "recipe",
                        "geometric decay ratio must be in [0.0, 1.0]",
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Validates synthesis parameters against physical and numerical limits.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] if parameters violate physical domains.
    pub fn validate_params(
        f0: f64,
        base_amplitude: f64,
        num_partials: usize,
        inharmonicity_b: f64,
        sample_rate: f64,
    ) -> DspResult<()> {
        if sample_rate <= 0.0 || !sample_rate.is_finite() {
            return Err(DspError::invalid_parameter(
                "sample_rate",
                "sample_rate must be positive and finite",
            ));
        }
        if f0 <= 0.0 || !f0.is_finite() {
            return Err(DspError::invalid_parameter(
                "f0",
                "fundamental frequency f0 must be positive and finite",
            ));
        }
        if num_partials == 0 {
            return Err(DspError::invalid_parameter(
                "num_partials",
                "num_partials must be greater than zero",
            ));
        }
        if inharmonicity_b < 0.0 || !inharmonicity_b.is_finite() {
            return Err(DspError::invalid_parameter(
                "inharmonicity_b",
                "inharmonicity_b cannot be negative or non-finite",
            ));
        }
        if base_amplitude < 0.0 || !base_amplitude.is_finite() {
            return Err(DspError::invalid_parameter(
                "base_amplitude",
                "base_amplitude cannot be negative or non-finite",
            ));
        }
        Ok(())
    }

    /// Computes the active harmonic partials below the Nyquist limit.
    fn compute_active_partials(&self, sample_rate: f64) -> Vec<KnownPartial> {
        let nyquist = sample_rate / 2.0;
        let mut partials = Vec::with_capacity(self.num_partials);

        for k in 1..=self.num_partials {
            // Apply recipe harmonic filtering and amplitude calculation
            let (amplitude, phase) = match &self.recipe {
                HarmonicRecipe::Sawtooth => {
                    let a = self.base_amplitude / (k as f64);
                    (a, 0.0)
                }
                HarmonicRecipe::Square => {
                    if k % 2 == 0 {
                        continue; // Skip even harmonics
                    }
                    let a = self.base_amplitude / (k as f64);
                    (a, 0.0)
                }
                HarmonicRecipe::Triangle => {
                    if k % 2 == 0 {
                        continue; // Skip even harmonics
                    }
                    let k_f = k as f64;
                    let a = self.base_amplitude / (k_f * k_f);
                    // Alternating sign: k=1 -> +, k=3 -> -, k=5 -> +, ...
                    let sign_idx = (k - 1) / 2;
                    let phase = if sign_idx % 2 == 1 { PI } else { 0.0 };
                    (a, phase)
                }
                HarmonicRecipe::GeometricDecay { ratio } => {
                    let a = self.base_amplitude * ratio.powi((k - 1) as i32);
                    (a, 0.0)
                }
                HarmonicRecipe::Custom(weights) => {
                    if k - 1 >= weights.len() {
                        continue;
                    }
                    let a = self.base_amplitude * weights[k - 1];
                    (a, 0.0)
                }
            };

            // Inharmonicity: f_k = k * f0 * sqrt(1 + B * k^2)
            let k_f = k as f64;
            let dispersion = (1.0 + self.inharmonicity_b * k_f * k_f).sqrt();
            let fk = k_f * self.f0 * dispersion;

            // Anti-aliasing: cull partials at or above Nyquist
            if fk >= nyquist {
                continue;
            }

            partials.push(KnownPartial::new(k, fk, amplitude, phase));
        }

        partials
    }
}

impl SignalGenerator for HarmonicComplexGenerator {
    fn generate(&self, num_samples: usize, sample_rate: f64) -> SignalBuffer {
        let mut samples = vec![0.0; num_samples];
        self.generate_into(&mut samples, sample_rate);

        let active_partials = self.compute_active_partials(sample_rate);

        let metadata = SignalMetadata {
            signal_type: SignalKind::HarmonicComplex,
            fundamental_freq: Some(self.f0),
            inharmonicity_b: Some(self.inharmonicity_b),
            known_partials: active_partials,
            signal_description: format!(
                "Harmonic complex f0={}Hz, B={}, recipe={:?}",
                self.f0, self.inharmonicity_b, self.recipe
            ),
            ..Default::default()
        };

        SignalBuffer::new(sample_rate, samples, metadata)
    }

    fn generate_into(&self, buffer: &mut [f64], sample_rate: f64) {
        if buffer.is_empty() || sample_rate <= 0.0 {
            return;
        }

        buffer.fill(0.0);
        let active_partials = self.compute_active_partials(sample_rate);

        for p in &active_partials {
            let two_pi_f = 2.0 * PI * p.frequency_hz;
            let tau_k = self.damping_tau.map(|tau0| tau0 / (p.harmonic_index as f64).sqrt());

            for (n, s) in buffer.iter_mut().enumerate() {
                let t = n as f64 / sample_rate;
                let env = match tau_k {
                    Some(tau) => (-t / tau).exp(),
                    None => 1.0,
                };
                *s += p.amplitude * env * (two_pi_f * t + p.phase).sin();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sawtooth_harmonics() {
        let gen = HarmonicComplexGenerator::new(100.0, 1.0, 4, 0.0, HarmonicRecipe::Sawtooth);
        let buf = gen.generate(1000, 48000.0);
        assert_eq!(buf.metadata.known_partials.len(), 4);
        assert_eq!(buf.metadata.known_partials[0].harmonic_index, 1);
        assert_eq!(buf.metadata.known_partials[3].harmonic_index, 4);
    }
}
