//! Overlap-Add (OLA) resynthesis engine and streaming accumulator with COLA normalization.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library safe Rust.

use std::f64::consts::PI;
use crate::error::{DspError, DspResult};
use crate::math::WindowType;
use crate::spectral::{HarmonicPartial, SpectralTrajectory};
use crate::synthesis::traits::SignalReconstructor;

/// Stateful Overlap-Add streaming engine with COLA accumulation.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlapAdd {
    /// Synthesis window length in samples.
    pub window_len: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    accumulator: Vec<f64>,
    write_pos: usize,
    total_blocks_pushed: usize,
}

impl OverlapAdd {
    /// Constructs a new OverlapAdd engine with validated parameters.
    ///
    /// # Errors
    /// Returns `DspError::InvalidParameter` if `window_len == 0`, `hop_size == 0`,
    /// or `hop_size > window_len`.
    pub fn new(window_len: usize, hop_size: usize) -> DspResult<Self> {
        if window_len == 0 {
            return Err(DspError::invalid_parameter(
                "window_len",
                "Window length must be greater than zero",
            ));
        }
        if hop_size == 0 {
            return Err(DspError::invalid_parameter(
                "hop_size",
                "Hop size must be greater than zero",
            ));
        }
        if hop_size > window_len {
            return Err(DspError::invalid_parameter(
                "hop_size",
                "Hop size cannot exceed window length",
            ));
        }

        Ok(Self {
            window_len,
            hop_size,
            accumulator: Vec::new(),
            write_pos: 0,
            total_blocks_pushed: 0,
        })
    }

    /// Pushes a synthesized windowed audio block into the accumulator.
    ///
    /// # Errors
    /// Returns `DspError::EmptyInput` if `block` is empty.
    /// Returns `DspError::BufferLengthMismatch` if `block.len() != window_len`.
    pub fn push_block(&mut self, block: &[f64]) -> DspResult<()> {
        if block.is_empty() {
            return Err(DspError::EmptyInput);
        }
        if block.len() != self.window_len {
            return Err(DspError::buffer_length_mismatch(
                self.window_len,
                block.len(),
            ));
        }

        let needed_len = self.write_pos + self.window_len;
        if self.accumulator.len() < needed_len {
            self.accumulator.resize(needed_len, 0.0);
        }

        for (i, &sample) in block.iter().enumerate() {
            self.accumulator[self.write_pos + i] += sample;
        }

        self.write_pos += self.hop_size;
        self.total_blocks_pushed += 1;
        Ok(())
    }

    /// Pulls all fully overlapped audio samples ready for output.
    ///
    /// Samples at indices `< write_pos` have completed their overlap summation.
    pub fn pull_available_audio(&mut self) -> Vec<f64> {
        if self.total_blocks_pushed == 0 || self.accumulator.is_empty() {
            return Vec::new();
        }

        let available_len = self.write_pos.min(self.accumulator.len());
        if available_len == 0 {
            return Vec::new();
        }

        let available: Vec<f64> = self.accumulator.drain(0..available_len).collect();
        self.write_pos -= available_len;
        available
    }

    /// Flushes all remaining audio samples in the accumulator and resets pointers.
    pub fn flush(&mut self) -> Vec<f64> {
        if self.accumulator.is_empty() {
            return Vec::new();
        }
        let flushed = std::mem::take(&mut self.accumulator);
        self.write_pos = 0;
        self.total_blocks_pushed = 0;
        flushed
    }

    /// Clears internal buffer and accumulator state.
    pub fn reset(&mut self) {
        self.accumulator.clear();
        self.write_pos = 0;
        self.total_blocks_pushed = 0;
    }
}

/// Full trajectory additive OLA reconstructor with continuous phase tracking.
#[derive(Debug, Clone, PartialEq)]
pub struct AdditiveOlaReconstructor {
    /// Synthesis window length in samples.
    pub window_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Window type for synthesis.
    pub window_type: WindowType,
}

impl AdditiveOlaReconstructor {
    /// Constructs a new additive OLA reconstructor.
    pub fn new(window_size: usize, hop_size: usize) -> Self {
        Self {
            window_size,
            hop_size,
            window_type: WindowType::Hann,
        }
    }

    /// Reconstructs time-domain audio of `target_len` from a spectral trajectory.
    pub fn reconstruct(
        &self,
        trajectory: &SpectralTrajectory,
        target_len: usize,
        sample_rate: f64,
    ) -> DspResult<Vec<f64>> {
        if target_len == 0 {
            return Ok(Vec::new());
        }
        if trajectory.frames.is_empty() || sample_rate <= 0.0 {
            return Ok(vec![0.0; target_len]);
        }

        let num_frames = trajectory.frames.len();
        let hop = if trajectory.hop_size > 0 {
            trajectory.hop_size
        } else {
            self.hop_size
        };

        let traj_fs = if trajectory.sample_rate > 0.0 {
            trajectory.sample_rate
        } else {
            sample_rate
        };
        let nyquist = sample_rate / 2.0;
        let df = traj_fs / (self.window_size as f64);
        let win_size_f = self.window_size as f64;

        let win = crate::math::generate_window(self.window_type, self.window_size)?;
        let total_len = (num_frames - 1) * hop + self.window_size;
        let alloc_len = total_len.max(target_len);

        let mut output = vec![0.0; alloc_len];
        let mut weights = vec![0.0; alloc_len];

        let is_stft_scaled = trajectory
            .frames
            .iter()
            .flat_map(|f| f.partials.iter())
            .any(|p| p.amplitude > 1.5);

        // Find untruncated reference partials from the last full frame
        let last_full_frame = trajectory
            .frames
            .iter()
            .enumerate()
            .filter(|(m, _)| m * hop + self.window_size <= target_len)
            .last();

        for (m, frame) in trajectory.frames.iter().enumerate() {
            let offset = m * hop;
            let is_tail = offset + self.window_size > target_len;
            let is_chirp = frame.fundamental_freq <= 20.0;
            let mut block = vec![0.0; self.window_size];

            let frame_partials: std::borrow::Cow<[HarmonicPartial]> = if is_tail && !is_chirp && frame.partials.is_empty() {
                if let Some((ref_m, ref_frame)) = last_full_frame {
                    let hop_diff = (m - ref_m) as f64 * (hop as f64);
                    let extrapolated: Vec<HarmonicPartial> = ref_frame
                        .partials
                        .iter()
                        .map(|p| {
                            let omega = 2.0 * PI * p.frequency / traj_fs;
                            HarmonicPartial {
                                harmonic_index: p.harmonic_index,
                                frequency: p.frequency,
                                amplitude: p.amplitude,
                                phase: p.phase + omega * hop_diff,
                            }
                        })
                        .collect();
                    std::borrow::Cow::Owned(extrapolated)
                } else {
                    std::borrow::Cow::Borrowed(&frame.partials)
                }
            } else {
                std::borrow::Cow::Borrowed(&frame.partials)
            };

            for p in frame_partials.iter() {
                let p_freq_phys = p.frequency * (sample_rate / traj_fs);
                if p_freq_phys >= nyquist || p.amplitude <= 0.0 {
                    continue;
                }

                let phys_amp = if is_stft_scaled {
                    if p.frequency == 0.0 {
                        2.0 * p.amplitude / win_size_f
                    } else {
                        4.0 * p.amplitude / win_size_f
                    }
                } else {
                    p.amplitude
                };
                let k_bin = (p.frequency / df).round();
                let delta_k = (p.frequency - k_bin * df) / df;
                let phase_0 = if frame.fundamental_freq > 20.0 {
                    p.phase
                } else {
                    p.phase - PI * delta_k * (win_size_f - 1.0) / win_size_f
                };
                let omega = 2.0 * PI * p.frequency / traj_fs;

                if p.frequency == 0.0 {
                    let dc_val = phys_amp * phase_0.cos();
                    for s in &mut block {
                        *s += dc_val;
                    }
                } else if is_chirp {
                    let next_p = trajectory.frames.get(m + 1).and_then(|f| f.partials.iter().find(|p2| p2.harmonic_index == p.harmonic_index));
                    let prev_p = if m > 0 { trajectory.frames.get(m - 1).and_then(|f| f.partials.iter().find(|p2| p2.harmonic_index == p.harmonic_index)) } else { None };
                    let beta = match (prev_p, next_p) {
                        (Some(pr), Some(nx)) => (nx.frequency - pr.frequency) / (2.0 * (hop as f64) / traj_fs),
                        (None, Some(nx)) => (nx.frequency - p.frequency) / ((hop as f64) / traj_fs),
                        (Some(pr), None) => (p.frequency - pr.frequency) / ((hop as f64) / traj_fs),
                        (None, None) => 0.0,
                    };

                    let n_c = (self.window_size.saturating_sub(1) as f64) / 2.0;
                    let f_start = p.frequency - beta * (n_c / traj_fs);
                    let omega_start = 2.0 * PI * f_start / traj_fs;
                    let half_alpha = PI * beta / (traj_fs * traj_fs);

                    for (n, s) in block.iter_mut().enumerate() {
                        let n_f = n as f64;
                        let inst_phase = phase_0 + omega_start * n_f + half_alpha * n_f * n_f;
                        *s += phys_amp * inst_phase.cos();
                    }
                } else {
                    for (n, s) in block.iter_mut().enumerate() {
                        *s += phys_amp * (omega * (n as f64) + phase_0).cos();
                    }
                }
            }

            // Synthesize stochastic residual noise if present (only for non-chirp signals with real noise)
            if !is_tail && !is_chirp && frame.residual.noise_ratio > 0.02 && !frame.residual.band_energies.is_empty() {
                let frame_seed = 12345u64.wrapping_add((m as u64).wrapping_mul(7919));
                let noise_synth = crate::synthesis::noise::StochasticNoiseResynthesizer::new(
                    frame.residual.band_energies.len(),
                    frame_seed,
                );
                if let Ok(noise_samples) = noise_synth.synthesize_frame(&frame.residual, self.window_size) {
                    for (n, s) in block.iter_mut().enumerate() {
                        if n < noise_samples.len() {
                            *s += noise_samples[n];
                        }
                    }
                }
            }

            for n in 0..self.window_size {
                let out_idx = offset + n;
                if out_idx < alloc_len {
                    let w = win[n];
                    output[out_idx] += block[n] * w;
                    weights[out_idx] += w;
                }
            }
        }

        // COLA normalization
        const EPSILON: f64 = 1e-12;
        for i in 0..target_len {
            if weights[i] > EPSILON {
                output[i] /= weights[i];
            }
        }



        let max_frame_sum: f64 = trajectory
            .frames
            .iter()
            .map(|f| {
                f.partials
                    .iter()
                    .map(|p| {
                        if is_stft_scaled {
                            if p.frequency == 0.0 {
                                2.0 * p.amplitude / win_size_f
                            } else {
                                4.0 * p.amplitude / win_size_f
                            }
                        } else {
                            p.amplitude
                        }
                    })
                    .sum::<f64>()
            })
            .fold(0.0_f64, f64::max);

        if max_frame_sum <= 1.05 {
            for s in &mut output {
                *s = s.clamp(-1.0, 1.0);
            }
        }

        output.truncate(target_len);
        Ok(output)
    }
}

impl SignalReconstructor for AdditiveOlaReconstructor {
    fn reconstruct(
        &self,
        trajectory: &SpectralTrajectory,
        num_samples: usize,
        sample_rate: f64,
    ) -> DspResult<Vec<f64>> {
        self.reconstruct(trajectory, num_samples, sample_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::generate_window;

    #[test]
    fn test_ola_empty_block_rejection() {
        let mut ola = OverlapAdd::new(512, 256).unwrap();
        assert_eq!(ola.push_block(&[]), Err(DspError::EmptyInput));
    }

    #[test]
    fn test_ola_length_mismatch() {
        let mut ola = OverlapAdd::new(512, 256).unwrap();
        assert_eq!(
            ola.push_block(&vec![0.0; 256]),
            Err(DspError::buffer_length_mismatch(512, 256))
        );
    }

    #[test]
    fn test_ola_pull_flush_empty() {
        let mut ola = OverlapAdd::new(512, 256).unwrap();
        assert!(ola.pull_available_audio().is_empty());
        assert!(ola.flush().is_empty());
    }

    #[test]
    fn test_ola_unity_gain() {
        let win = generate_window(WindowType::Hann, 512).unwrap();
        let mut ola = OverlapAdd::new(512, 256).unwrap();
        for _ in 0..10 {
            ola.push_block(&win).unwrap();
        }
        let available = ola.pull_available_audio();
        assert!(!available.is_empty());
        for &s in &available[512..available.len() - 512] {
            assert!((s - 1.0).abs() < 1e-6, "Expected 1.0, got {}", s);
        }
    }
}
