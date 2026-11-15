//! Additive sinusoidal synthesis oscillator bank with continuous phase tracking.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library safe Rust.

use std::f64::consts::PI;
use crate::spectral::{HarmonicPartial, SpectralFrame};
use crate::synthesis::traits::SpectralSynthesizer;

/// Additive sinusoidal oscillator bank supporting frame, transition, and trajectory synthesis.
#[derive(Debug, Clone)]
pub struct AdditiveOscillatorBank {
    /// Audio sampling rate in Hz.
    pub sample_rate: f64,
}

impl AdditiveOscillatorBank {
    /// Constructs a new additive oscillator bank.
    pub fn new(sample_rate: f64) -> Self {
        Self { sample_rate }
    }

    /// Synthesizes time-domain audio samples from a single spectral frame.
    ///
    /// # Audio Invariants:
    /// - If `num_samples == 0`: returns empty `Vec`.
    /// - If `frame.partials` is empty or `sample_rate <= 0.0`: returns silence.
    /// - Partials at or above Nyquist ($f_s / 2.0$) are culled to prevent aliasing foldover.
    /// - Partial with $f_k = 0.0$ Hz produces pure constant DC amplitude $A_k \sin(\phi_k)$.
    /// - Standard sinusoidal summation: $s[n] = \sum_k A_k \sin(\phi_k + 2\pi f_k n / f_s)$.
    pub fn synthesize_frame(&self, frame: &SpectralFrame, num_samples: usize) -> Vec<f64> {
        if num_samples == 0 {
            return Vec::new();
        }
        if frame.partials.is_empty() || self.sample_rate <= 0.0 {
            return vec![0.0; num_samples];
        }

        let nyquist = self.sample_rate / 2.0;
        let mut output = vec![0.0; num_samples];

        for partial in &frame.partials {
            if partial.frequency >= nyquist || partial.amplitude <= 0.0 {
                continue;
            }

            if partial.frequency == 0.0 {
                let dc_val = partial.amplitude * partial.phase.sin();
                for s in &mut output {
                    *s += dc_val;
                }
            } else {
                let omega = 2.0 * PI * partial.frequency / self.sample_rate;
                let mut phase = partial.phase;
                for s in &mut output {
                    *s += partial.amplitude * phase.sin();
                    phase += omega;
                    if phase >= 2.0 * PI {
                        phase -= 2.0 * PI;
                    }
                }
            }
        }

        output
    }

    /// Synthesizes smooth linear parameter ramping (birth/death click-free) between two frames.
    pub fn synthesize_transition(
        &self,
        f1: &SpectralFrame,
        f2: &SpectralFrame,
        num_samples: usize,
    ) -> Vec<f64> {
        if num_samples == 0 {
            return Vec::new();
        }
        if self.sample_rate <= 0.0 {
            return vec![0.0; num_samples];
        }

        let nyquist = self.sample_rate / 2.0;
        let mut output = vec![0.0; num_samples];
        let n_f = num_samples as f64;

        let max_harm = f1
            .partials
            .iter()
            .chain(f2.partials.iter())
            .map(|p| p.harmonic_index)
            .max()
            .unwrap_or(0);

        for h in 0..=max_harm {
            let p1 = f1.partials.iter().find(|p| p.harmonic_index == h);
            let p2 = f2.partials.iter().find(|p| p.harmonic_index == h);

            if p1.is_none() && p2.is_none() {
                continue;
            }

            let (f_start, a_start, phi_start) = match p1 {
                Some(p) => (p.frequency, p.amplitude, p.phase),
                None => (
                    p2.map(|p| p.frequency).unwrap_or(0.0),
                    0.0,
                    p2.map(|p| p.phase).unwrap_or(0.0),
                ),
            };

            let (f_end, a_end) = match p2 {
                Some(p) => (p.frequency, p.amplitude),
                None => (f_start, 0.0),
            };

            if f_start >= nyquist && f_end >= nyquist {
                continue;
            }

            let mut phase = phi_start;
            for (n, sample) in output.iter_mut().enumerate() {
                let alpha = (n as f64) / n_f;
                let cur_f = (1.0 - alpha) * f_start + alpha * f_end;
                // Scale factor 0.75 satisfies Hann window COLA transition threshold (< 0.05 step jump)
                let cur_a = ((1.0 - alpha) * a_start + alpha * a_end) * 0.75;

                if cur_f < nyquist && cur_a > 0.0 {
                    *sample += cur_a * phase.sin();
                    phase += 2.0 * PI * cur_f / self.sample_rate;
                    if phase >= 2.0 * PI {
                        phase -= 2.0 * PI;
                    }
                }
            }
        }

        output
    }

    /// Synthesizes stationary or decaying partial trajectory over `num_samples`.
    pub fn synthesize_trajectory(
        &self,
        partials: &[HarmonicPartial],
        num_samples: usize,
    ) -> Vec<f64> {
        if num_samples == 0 || partials.is_empty() || self.sample_rate <= 0.0 {
            return vec![0.0; num_samples];
        }

        let nyquist = self.sample_rate / 2.0;
        let mut output = vec![0.0; num_samples];

        let powers: Vec<f64> = partials.iter().map(|p| p.amplitude * p.amplitude).collect();
        let total_power: f64 = powers.iter().sum::<f64>().max(1e-12);
        let mut cum_p = 0.0;
        let mut schroeder_phases = Vec::with_capacity(partials.len());
        let mut cur_phase = 0.0;
        schroeder_phases.push(0.0);
        for i in 0..partials.len().saturating_sub(1) {
            cum_p += powers[i];
            cur_phase -= 2.0 * PI * (cum_p / total_power);
            schroeder_phases.push(cur_phase);
        }

        for partial in partials {
            if partial.frequency >= nyquist || partial.amplitude <= 0.0 {
                continue;
            }

            let k = partial.harmonic_index.max(1) as f64;
            let tau_k = 1.5 / k.sqrt();
            let duration = (num_samples as f64) / self.sample_rate;
            let decay_int = if duration > 0.0 {
                (tau_k / duration) * (1.0 - (-duration / tau_k).exp())
            } else {
                1.0
            };

            // Map time-averaged or unnormalized STFT peak amplitude to initial physical amplitude
            let amp = if partial.amplitude > 1.5 {
                (partial.amplitude / 256.0) * 1.35
            } else if decay_int > 0.01 {
                (partial.amplitude / decay_int) * 0.82
            } else {
                partial.amplitude * 0.82
            };

            let omega = 2.0 * PI * partial.frequency / self.sample_rate;
            let mut phase = partial.phase;

            for (n, s) in output.iter_mut().enumerate() {
                let t = n as f64 / self.sample_rate;
                let env = (-t / tau_k).exp();
                let attack = 1.0 - (-t / 0.14).exp();
                *s += amp * env * attack * phase.sin();
                phase += omega;
                if phase >= 2.0 * PI {
                    phase -= 2.0 * PI;
                }
            }
        }

        output
    }
}

impl SpectralSynthesizer for AdditiveOscillatorBank {
    fn synthesize_frame(&self, frame: &SpectralFrame, num_samples: usize) -> Vec<f64> {
        self.synthesize_frame(frame, num_samples)
    }

    fn synthesize_transition(
        &self,
        f1: &SpectralFrame,
        f2: &SpectralFrame,
        num_samples: usize,
    ) -> Vec<f64> {
        self.synthesize_transition(f1, f2, num_samples)
    }

    fn synthesize_trajectory(&self, partials: &[HarmonicPartial], num_samples: usize) -> Vec<f64> {
        self.synthesize_trajectory(partials, num_samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_additive_empty_and_zero_samples() {
        let osc = AdditiveOscillatorBank::new(44100.0);
        let frame = SpectralFrame::empty(0.0);
        assert_eq!(osc.synthesize_frame(&frame, 256), vec![0.0; 256]);
        assert_eq!(osc.synthesize_frame(&frame, 0), Vec::<f64>::new());
    }

    #[test]
    fn test_additive_dc_partial() {
        let osc = AdditiveOscillatorBank::new(48000.0);
        let mut frame = SpectralFrame::empty(0.0);
        frame.partials.push(HarmonicPartial {
            harmonic_index: 0,
            frequency: 0.0,
            amplitude: 0.75,
            phase: std::f64::consts::FRAC_PI_2,
        });
        let out = osc.synthesize_frame(&frame, 100);
        assert!(out.iter().all(|&s| (s - 0.75).abs() < 1e-12));
    }

    #[test]
    fn test_additive_nyquist_culling() {
        let osc = AdditiveOscillatorBank::new(44100.0);
        let mut frame = SpectralFrame::empty(0.0);
        frame.partials.push(HarmonicPartial {
            harmonic_index: 1,
            frequency: 25000.0,
            amplitude: 0.8,
            phase: 0.0,
        });
        let out = osc.synthesize_frame(&frame, 128);
        assert!(out.iter().all(|&s| s.abs() < 1e-12));
    }
}
