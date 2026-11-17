//! 5D parameterized synthetic acoustic instrument note synthesizer.

use crate::error::{DspError, DspResult};
use crate::synthetic::buffer::{
    HarmonicPartial, KnownPartial, NoiseBandEnvelope, SignalBuffer, SignalKind, SignalMetadata,
    SpectralFrame,
};
use crate::synthetic::noise::Pcg32;
use std::f64::consts::PI;

/// Performance articulation style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Articulation {
    /// Sustained note (long envelope, gradual decay).
    #[default]
    Sustain,
    /// Staccato note (short duration, rapid decay).
    Staccato,
}

/// 5D parameterized acoustic instrument source spanning Time, Pitch, Dynamic, Articulation, Variation.
#[derive(Debug, Clone, Default)]
pub struct SyntheticInstrumentSource;

impl SyntheticInstrumentSource {
    /// Constructs a new synthetic instrument source.
    pub fn new() -> Self {
        Self
    }

    /// Generates an acoustic instrument note from discrete musical parameters.
    pub fn generate_note(
        &self,
        pitch: u8,
        dynamic: f64,
        articulation: Articulation,
        variation: usize,
    ) -> SignalBuffer {
        let sample_rate = 48000.0;
        let pitch_f = pitch as f64;
        let f0 = 440.0 * 2.0_f64.powf((pitch_f - 69.0) / 12.0);

        let (duration, tau_base): (f64, f64) = match articulation {
            Articulation::Staccato => (1024.0 / sample_rate, 0.02),
            Articulation::Sustain => (2048.0 / sample_rate, 0.1),
        };

        let num_samples = (duration * sample_rate).round() as usize;
        let d = dynamic.clamp(0.0, 1.0);

        if d == 0.0 || num_samples == 0 {
            let mut buf = SignalBuffer::empty(sample_rate);
            buf.samples = vec![0.0; num_samples];
            buf.metadata.fundamental_freq = Some(f0);
            return buf;
        }

        let mut samples = vec![0.0; num_samples];
        let nyquist = sample_rate / 2.0;

        // Brightness exponent: beta = 2.5 (dark) down to 1.0 (bright)
        let beta = 2.5 - 1.5 * d;

        let mut rng = Pcg32::new((variation as u64).wrapping_mul(1000).wrapping_add(pitch as u64), 17);
        let mut partials = Vec::with_capacity(16);

        for k in 1..=16 {
            let k_f = k as f64;
            let fk = k_f * f0;
            if fk >= nyquist {
                break;
            }

            let ak = d * k_f.powf(-beta);
            let phase = rng.next_f64() * 2.0 * PI;
            partials.push(KnownPartial::new(k, fk, ak, phase));
        }

        let release_len = ((0.01 * sample_rate) as usize).min(num_samples / 4);

        for (n, s) in samples.iter_mut().enumerate() {
            let t = n as f64 / sample_rate;
            let mut val = 0.0_f64;
            for p in &partials {
                let tau_k = tau_base / (p.harmonic_index as f64).sqrt();
                let env = (-t / tau_k).exp();
                val += p.amplitude * env * (2.0 * PI * p.frequency_hz * t + p.phase).sin();
            }

            // Smooth attack and release to eliminate boundary spectral splatter
            let window_gain = if n < release_len {
                0.5 * (1.0 - (PI * n as f64 / release_len as f64).cos())
            } else if n >= num_samples - release_len {
                let rem = (num_samples - 1 - n) as f64;
                0.5 * (1.0 - (PI * rem / release_len as f64).cos())
            } else {
                1.0
            };

            *s = val * window_gain;
        }

        let max_peak = samples.iter().copied().fold(0.0_f64, |m, x| m.max(x.abs()));
        if max_peak > 1.0 {
            let scale = 1.0 / max_peak;
            for s in &mut samples {
                *s *= scale;
            }
        }

        let metadata = SignalMetadata {
            signal_type: SignalKind::InstrumentNote,
            fundamental_freq: Some(f0),
            known_partials: partials,
            signal_description: format!(
                "Instrument Note: pitch={}, dynamic={:.2}, art={:?}, var={}",
                pitch, dynamic, articulation, variation
            ),
            ..Default::default()
        };

        SignalBuffer::new(sample_rate, samples, metadata)
    }

    /// Synthesizes audio from normalized 5D hypercube coordinates $[t, p, d, a, v] \in [0.0, 1.0]^5$.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] if any coordinate is outside $[0.0, 1.0]$.
    pub fn synthesize(
        &self,
        coords: [f64; 5],
        num_samples: usize,
        sample_rate: f64,
    ) -> DspResult<SignalBuffer> {
        for &c in &coords {
            if c < 0.0 || c > 1.0 || !c.is_finite() {
                return Err(DspError::invalid_parameter(
                    "coordinates",
                    format!("Coordinate {} outside normalized range [0.0, 1.0]", c),
                ));
            }
        }

        if sample_rate <= 0.0 || !sample_rate.is_finite() {
            return Err(DspError::invalid_parameter(
                "sample_rate",
                "sample_rate must be positive and finite",
            ));
        }

        if num_samples == 0 {
            return Ok(SignalBuffer::empty(sample_rate));
        }

        let dynamic = coords[2];
        if dynamic == 0.0 {
            let mut buf = SignalBuffer::empty(sample_rate);
            buf.samples = vec![0.0; num_samples];
            return Ok(buf);
        }

        // Map pitch: [0, 1] -> MIDI [21, 108]
        let midi_pitch = 21.0 + coords[1] * (108.0 - 21.0);
        let f0 = 440.0 * 2.0_f64.powf((midi_pitch - 69.0) / 12.0);

        let articulation = if coords[3] > 0.5 {
            Articulation::Sustain
        } else {
            Articulation::Staccato
        };

        let tau_base = match articulation {
            Articulation::Staccato => 0.15,
            Articulation::Sustain => 1.0,
        };

        let beta = 2.5 - 1.5 * dynamic;
        let variation = (coords[4] * 1000.0).round() as u64;
        let mut rng = Pcg32::new(variation, 23);

        let mut samples = vec![0.0; num_samples];
        let nyquist = sample_rate / 2.0;
        let mut partials = Vec::with_capacity(16);

        for k in 1..=16 {
            let k_f = k as f64;
            let fk = k_f * f0;
            if fk >= nyquist {
                break;
            }

            let ak = dynamic * k_f.powf(-beta);
            let phase = rng.next_f64() * 2.0 * PI;
            partials.push(KnownPartial::new(k, fk, ak, phase));
        }

        let release_len = ((0.01 * sample_rate) as usize).min(num_samples / 4);

        for (n, s) in samples.iter_mut().enumerate() {
            let t = n as f64 / sample_rate;
            let mut val = 0.0_f64;
            for p in &partials {
                let tau_k = tau_base / (p.harmonic_index as f64).sqrt();
                let env = (-t / tau_k).exp();
                val += p.amplitude * env * (2.0 * PI * p.frequency_hz * t + p.phase).sin();
            }

            // Smooth attack and release to eliminate boundary spectral splatter
            let window_gain = if n < release_len {
                0.5 * (1.0 - (PI * n as f64 / release_len as f64).cos())
            } else if n >= num_samples - release_len {
                let rem = (num_samples - 1 - n) as f64;
                0.5 * (1.0 - (PI * rem / release_len as f64).cos())
            } else {
                1.0
            };

            *s = val * window_gain;
        }

        let max_peak = samples.iter().copied().fold(0.0_f64, |m, x| m.max(x.abs()));
        if max_peak > 1.0 {
            let scale = 1.0 / max_peak;
            for s in &mut samples {
                *s *= scale;
            }
        }

        let metadata = SignalMetadata {
            signal_type: SignalKind::InstrumentNote,
            fundamental_freq: Some(f0),
            known_partials: partials,
            hypercube_coords: Some(coords),
            signal_description: format!("Synthesized hypercube note at {:?}", coords),
            ..Default::default()
        };

        Ok(SignalBuffer::new(sample_rate, samples, metadata))
    }

    /// Queries an instantaneous analytical [`SpectralFrame`] at 5D coordinates $[t, p, d, a, v]$.
    ///
    /// Accepts unnormalized coordinates where $p$ is continuous MIDI pitch (e.g. 60.5).
    pub fn query_frame(&self, coords: [f64; 5]) -> DspResult<SpectralFrame> {
        let t = coords[0];
        let pitch_midi = if coords[1] > 1.0 {
            coords[1]
        } else {
            21.0 + coords[1] * (108.0 - 21.0)
        };
        let dynamic = coords[2].clamp(0.0, 1.0);
        let tau_base = if coords[3] > 0.5 { 1.0 } else { 0.15 };

        let f0 = 440.0 * 2.0_f64.powf((pitch_midi - 69.0) / 12.0);
        let beta = 2.5 - 1.5 * dynamic;

        let mut partials = Vec::with_capacity(16);
        for k in 1..=16 {
            let k_f = k as f64;
            let fk = k_f * f0;
            let tau_k = tau_base / k_f.sqrt();
            let ak = dynamic * k_f.powf(-beta) * (-t / tau_k).exp();

            partials.push(HarmonicPartial::new(k, fk, ak, 0.0));
        }

        Ok(SpectralFrame::new(
            t,
            f0,
            partials,
            NoiseBandEnvelope::empty(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instrument_generate_and_query() {
        let inst = SyntheticInstrumentSource::new();
        let note = inst.generate_note(69, 0.8, Articulation::Sustain, 1);
        assert_eq!(note.metadata.fundamental_freq, Some(440.0));

        let frame = inst.query_frame([0.1, 69.0, 0.8, 0.8, 1.0]).unwrap();
        assert!((frame.fundamental_freq - 440.0).abs() < 1e-6);
        assert_eq!(frame.partials.len(), 16);
    }
}
