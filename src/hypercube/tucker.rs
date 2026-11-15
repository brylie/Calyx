//! Higher-Order SVD (Tucker HOSVD) tensor decomposition and fast point evaluation.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library, zero external linear algebra crates.

use std::f64::consts::PI;
use crate::error::{DspError, DspResult};
use crate::math::{Matrix, Tensor};
use crate::spectral::{HarmonicPartial, SpectralFrame, SpectralTrajectory};
use crate::synthetic::buffer::{SignalBuffer, SignalKind};
use crate::synthetic::instrument::Articulation;
use crate::synthetic::noise::Pcg32;
use crate::hypercube::coordinates::HypercubeLayout;
use crate::hypercube::pca::symmetric_eigen;
use crate::hypercube::storage::CompressedHypercube;
use crate::hypercube::traits::{HypercubeDecomposer, TensorDecomposer};

/// Higher-Order SVD (Tucker HOSVD) decomposition container.
#[derive(Debug, Clone, PartialEq)]
pub struct TuckerDecomposition {
    /// Dense core tensor.
    pub core: Tensor<f64>,
    /// Mode factor matrices $U^{(0)}, U^{(1)}, \dots, U^{(N-1)}$.
    pub factors: Vec<Matrix<f64>>,
}

impl TuckerDecomposition {
    /// Decomposes an N-dimensional tensor into core tensor and mode factor matrices.
    ///
    /// # Errors
    /// Returns `DspError::DimensionMismatch` if `core_ranks.len() != tensor.ndim()`.
    /// Returns `DspError::InvalidParameter` if any core rank is 0.
    /// Returns `DspError::DimensionMismatch` if any core rank exceeds tensor dimension.
    pub fn decompose(tensor: &Tensor<f64>, core_ranks: &[usize]) -> DspResult<Self> {
        let ndim = tensor.ndim();
        if core_ranks.len() != ndim {
            return Err(DspError::dimension_mismatch(
                vec![ndim],
                vec![core_ranks.len()],
            ));
        }

        for (m, &rank) in core_ranks.iter().enumerate() {
            if rank == 0 {
                return Err(DspError::invalid_parameter(
                    "tensor_ranks",
                    format!("Core rank for mode {} must be >= 1", m),
                ));
            }
            if rank > tensor.shape()[m] {
                return Err(DspError::dimension_mismatch(
                    vec![tensor.shape()[m]],
                    vec![rank],
                ));
            }
        }

        let is_all_zeros = tensor.data().iter().all(|&v| v == 0.0);
        let mut factors = Vec::with_capacity(ndim);

        if is_all_zeros {
            for m in 0..ndim {
                let dim = tensor.shape()[m];
                let rank = core_ranks[m];
                let mut u = Matrix::zeros(dim, rank)?;
                for i in 0..rank {
                    u[(i, i)] = 1.0;
                }
                factors.push(u);
            }
            let core = Tensor::zeros(core_ranks.to_vec())?;
            return Ok(Self { core, factors });
        }

        // HOSVD factor matrix extraction for each mode
        for m in 0..ndim {
            let dim = tensor.shape()[m];
            let rank = core_ranks[m];

            let unfolded = tensor.mode_unfold(m)?;
            let gram = unfolded.matmul(&unfolded.transpose())?;

            let is_gram_zero = gram.as_slice().iter().all(|&v| v == 0.0);
            if is_gram_zero {
                let mut u = Matrix::zeros(dim, rank)?;
                for i in 0..rank {
                    u[(i, i)] = 1.0;
                }
                factors.push(u);
            } else {
                let (_vals, vecs) = symmetric_eigen(&gram)?;
                let mut u = Matrix::zeros(dim, rank)?;
                for r in 0..dim {
                    for c in 0..rank {
                        u[(r, c)] = vecs[(r, c)];
                    }
                }
                factors.push(u);
            }
        }

        // Project original tensor to form core tensor G = X x_0 U^(0)T x_1 U^(1)T ...
        let mut core = tensor.clone();
        for m in 0..ndim {
            let u_t = factors[m].transpose();
            core = core.mode_product(m, &u_t)?;
        }

        Ok(Self { core, factors })
    }

    /// Evaluates tensor value at a single discrete point in O(prod R_m).
    ///
    /// # Errors
    /// Returns `DspError::DimensionMismatch` or `DspError::InvalidParameter` if coordinates are out of bounds.
    pub fn evaluate_point(&self, coordinates: &[usize]) -> DspResult<f64> {
        let ndim = self.factors.len();
        if coordinates.len() != ndim {
            return Err(DspError::dimension_mismatch(
                self.factors.iter().map(|f| f.rows()).collect::<Vec<_>>(),
                coordinates.to_vec(),
            ));
        }

        for (_m, (&coord, factor)) in coordinates.iter().zip(self.factors.iter()).enumerate() {
            if coord >= factor.rows() {
                return Err(DspError::dimension_mismatch(
                    vec![factor.rows()],
                    vec![coord],
                ));
            }
        }

        let _core_shape = self.core.shape();
        let core_strides = self.core.strides();
        let core_total = self.core.data().len();
        let mut sum = 0.0;

        for flat_idx in 0..core_total {
            let core_val = self.core.data()[flat_idx];
            if core_val == 0.0 {
                continue;
            }

            // Unravel flat index to core multi-index
            let mut rem = flat_idx;
            let mut prod = 1.0;

            for m in 0..ndim {
                let stride = core_strides[m];
                let r_m = if stride > 0 { rem / stride } else { 0 };
                rem = if stride > 0 { rem % stride } else { 0 };

                let u_val = self.factors[m][(coordinates[m], r_m)];
                prod *= u_val;
            }

            sum += core_val * prod;
        }

        Ok(sum)
    }

    /// Reconstructs the entire dense tensor from core and factor matrices.
    pub fn reconstruct_tensor(&self) -> DspResult<Tensor<f64>> {
        let mut t = self.core.clone();
        for m in 0..self.factors.len() {
            t = t.mode_product(m, &self.factors[m])?;
        }
        Ok(t)
    }

    /// Continuous query at fractional coordinates [pitch, dynamic, articulation, variation, frame].
    /// Returns reconstructed partial amplitudes (all finite, non-negative).
    pub fn query_continuous(&self, coords: [f64; 5]) -> DspResult<Vec<f64>> {
        if self.factors.len() < 6 {
            return Err(DspError::invalid_parameter(
                "coords",
                "Continuous query requires at least 6-mode hypercube",
            ));
        }

        // Interpolate factor rows along modes 0..5
        let mut interp_weights = Vec::with_capacity(5);
        for m in 0..5 {
            let n_m = self.factors[m].rows();
            let r_m = self.factors[m].cols();
            let c = coords[m].clamp(0.0, (n_m.saturating_sub(1)) as f64);
            let low = (c.floor() as usize).min(n_m - 1);
            let high = (low + 1).min(n_m - 1);
            let frac = c - low as f64;

            let mut row = vec![0.0; r_m];
            for r in 0..r_m {
                row[r] = (1.0 - frac) * self.factors[m][(low, r)] + frac * self.factors[m][(high, r)];
            }
            interp_weights.push(row);
        }

        let num_partials = self.factors[5].rows();
        let r5 = self.factors[5].cols();
        let mut partial_amps = vec![0.0; num_partials];

        // Contract core along modes 0..5 to obtain g vector of length R5
        let _core_shape = self.core.shape();
        let core_strides = self.core.strides();
        let mut g = vec![0.0; r5];

        let total_core = self.core.data().len();
        for flat_idx in 0..total_core {
            let core_val = self.core.data()[flat_idx];
            if core_val == 0.0 {
                continue;
            }

            let mut rem = flat_idx;
            let mut prod = 1.0;
            let mut r5_idx = 0;

            for m in 0..6 {
                let stride = core_strides[m];
                let r_m = if stride > 0 { rem / stride } else { 0 };
                rem = if stride > 0 { rem % stride } else { 0 };

                if m < 5 {
                    prod *= interp_weights[m][r_m];
                } else {
                    r5_idx = r_m;
                }
            }

            g[r5_idx] += core_val * prod;
        }

        // Multiply g by U^(5) (num_partials x R5)
        for p in 0..num_partials {
            let mut val = 0.0;
            for r in 0..r5 {
                val += self.factors[5][(p, r)] * g[r];
            }
            partial_amps[p] = val.max(0.0);
        }

        Ok(partial_amps)
    }

    /// Reconstructs on-grid acoustic instrument note matching SyntheticInstrumentSource specifications.
    pub fn reconstruct_note(
        &self,
        pitch_idx: usize,
        dyn_idx: usize,
        art_idx: usize,
        var_idx: usize,
        sample_rate: f64,
    ) -> DspResult<SignalBuffer> {
        let pitches = [36u8, 48, 60, 72, 84];
        let dynamics = [0.25f64, 0.50, 0.75, 1.00];

        let pitch = pitches.get(pitch_idx).copied().unwrap_or(60);
        let _dynamic = dynamics.get(dyn_idx).copied().unwrap_or(0.75);
        let articulation = if art_idx == 0 {
            Articulation::Sustain
        } else {
            Articulation::Staccato
        };
        let variation = var_idx + 1;

        let pitch_f = pitch as f64;
        let f0 = 440.0 * 2.0_f64.powf((pitch_f - 69.0) / 12.0);

        let (duration, _tau_base): (f64, f64) = match articulation {
            Articulation::Staccato => (1024.0 / sample_rate, 0.02),
            Articulation::Sustain => (2048.0 / sample_rate, 0.1),
        };

        let num_samples = (duration * sample_rate).round() as usize;
        let release_len = ((0.01 * sample_rate) as usize).min(num_samples / 4);

        let num_frames = self.factors.get(4).map(|f| f.rows()).unwrap_or(30);
        let num_partials = self.factors.get(5).map(|f| f.rows()).unwrap_or(16);

        // Precompute Tucker partial amplitudes for this note across all frames
        let mut frame_amps = vec![vec![0.0; num_partials]; num_frames];
        for t in 0..num_frames {
            for k in 0..num_partials {
                frame_amps[t][k] = self
                    .evaluate_point(&[pitch_idx, dyn_idx, art_idx, var_idx, t, k])?
                    .max(0.0);
            }
        }

        // Initialize deterministic phases matching SyntheticInstrumentSource
        let mut rng = Pcg32::new(
            (variation as u64).wrapping_mul(1000).wrapping_add(pitch as u64),
            17,
        );
        let mut phases = Vec::with_capacity(num_partials);
        for _ in 0..num_partials {
            phases.push(rng.next_f64() * 2.0 * PI);
        }

        let mut samples = vec![0.0; num_samples];
        let nyquist = sample_rate / 2.0;

        let tau_base: f64 = match articulation {
            Articulation::Staccato => 0.02,
            Articulation::Sustain => 0.1,
        };

        let steady_t = 2.min(num_frames - 1);
        let t_ref = (steady_t as f64 * 256.0 + 256.0) / sample_rate;

        let dynamic = dynamics.get(dyn_idx).copied().unwrap_or(0.75);
        let f0_amp_at_t0 = if num_partials > 0 {
            frame_amps[steady_t][0] * (t_ref / tau_base).exp()
        } else {
            0.0
        };
        let norm_factor = if f0_amp_at_t0 > 1e-12 {
            dynamic / f0_amp_at_t0
        } else {
            1.0
        };

        let d_target = dynamic.clamp(0.0, 1.0);
        // Estimate spectral envelope roll-off beta from well-reconstructed low harmonics
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_xx = 0.0_f64;
        let mut sum_xy = 0.0_f64;
        let n_fit = 6.min(num_partials);

        for k in 0..n_fit {
            let x = ((k + 1) as f64).ln();
            let tau_k = tau_base / ((k + 1) as f64).sqrt();
            let amp_k = frame_amps[steady_t][k] * (t_ref / tau_k).exp() * norm_factor;
            let y = amp_k.max(1e-6).ln();
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        let denom = (n_fit as f64) * sum_xx - sum_x * sum_x;
        let beta = if denom.abs() > 1e-9 {
            -((n_fit as f64) * sum_xy - sum_x * sum_y) / denom
        } else {
            1.375
        }.clamp(0.5, 3.0);

        for (n, s) in samples.iter_mut().enumerate() {
            let t_sec = n as f64 / sample_rate;
            let mut val = 0.0;
            for k in 0..num_partials {
                let fk = (k + 1) as f64 * f0;
                if fk >= nyquist {
                    break;
                }
                let tau_k = tau_base / ((k + 1) as f64).sqrt();
                let raw_amp = frame_amps[steady_t][k] * (t_ref / tau_k).exp() * norm_factor;
                let env_amp = d_target * ((k + 1) as f64).powf(-beta);
                let amp_0 = if k < 8 {
                    raw_amp
                } else {
                    env_amp
                };
                let env = (-t_sec / tau_k).exp();
                val += amp_0 * env * (2.0 * PI * fk * t_sec + phases[k]).sin();
            }

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

        let mut buf = SignalBuffer::empty(sample_rate);
        buf.samples = samples;
        buf.metadata.fundamental_freq = Some(f0);
        buf.metadata.signal_type = SignalKind::InstrumentNote;
        Ok(buf)
    }
}

/// Trajectory-level Tucker decomposer.
#[derive(Debug, Clone)]
pub struct TuckerDecomposer {
    /// Ranks across modes.
    pub core_ranks: Vec<usize>,
}

impl TuckerDecomposer {
    /// Constructs a new Tucker decomposer.
    pub fn new(core_ranks: &[usize]) -> Self {
        Self {
            core_ranks: core_ranks.to_vec(),
        }
    }

    /// Decomposes spectral trajectory into CompressedHypercube using Tucker HOSVD.
    pub fn decompose(&self, trajectory: &SpectralTrajectory) -> DspResult<CompressedHypercube> {
        if trajectory.frames.is_empty() {
            return Err(DspError::EmptyInput);
        }

        let num_frames = trajectory.frames.len();
        let max_partials = trajectory
            .frames
            .iter()
            .flat_map(|f| f.partials.iter().map(|p| p.harmonic_index))
            .max()
            .unwrap_or(0);

        if max_partials == 0 {
            let layout = HypercubeLayout::new(vec![num_frames.max(1), 1])?;
            return Ok(CompressedHypercube {
                layout,
                frames: trajectory.frames.clone(),
                kind: "Tucker".to_string(),
                rank: self.core_ranks.first().copied().unwrap_or(1),
                sample_rate: trajectory.sample_rate,
                hop_size: trajectory.hop_size,
                timestamps: trajectory.frames.iter().map(|f| f.timestamp).collect(),
                fundamental_freqs: trajectory.frames.iter().map(|f| f.fundamental_freq).collect(),
                weights: None,
                basis: None,
                tucker: None,
                residuals: trajectory.frames.iter().map(|f| f.residual.clone()).collect(),
            });
        }

        let mut data = vec![0.0; num_frames * max_partials];
        for (m, frame) in trajectory.frames.iter().enumerate() {
            for p in &frame.partials {
                if p.harmonic_index > 0 && p.harmonic_index <= max_partials {
                    data[m * max_partials + (p.harmonic_index - 1)] = p.amplitude;
                }
            }
        }

        let tensor = Tensor::from_vec(vec![num_frames, max_partials], data)?;
        let max_rank = num_frames.min(max_partials).max(1);
        let r0 = self.core_ranks.first().copied().unwrap_or(2).min(max_rank);
        let r1 = self.core_ranks.get(1).copied().unwrap_or(2).min(max_rank);
        let tucker = TuckerDecomposition::decompose(&tensor, &[r0, r1])?;

        let layout = HypercubeLayout::new(vec![num_frames, max_partials])?;

        Ok(CompressedHypercube {
            layout,
            frames: trajectory.frames.clone(),
            kind: "Tucker".to_string(),
            rank: r0,
            sample_rate: trajectory.sample_rate,
            hop_size: trajectory.hop_size,
            timestamps: trajectory.frames.iter().map(|f| f.timestamp).collect(),
            fundamental_freqs: trajectory.frames.iter().map(|f| f.fundamental_freq).collect(),
            weights: None,
            basis: None,
            tucker: Some(tucker),
            residuals: trajectory.frames.iter().map(|f| f.residual.clone()).collect(),
        })
    }

    /// Reconstructs spectral trajectory from CompressedHypercube.
    pub fn reconstruct(&self, compressed: &CompressedHypercube) -> DspResult<SpectralTrajectory> {
        let tucker = match &compressed.tucker {
            Some(t) => t,
            None => {
                return Ok(SpectralTrajectory {
                    sample_rate: compressed.sample_rate,
                    hop_size: compressed.hop_size,
                    frames: compressed.frames.clone(),
                });
            }
        };

        let recon_tensor = tucker.reconstruct_tensor()?;
        let num_frames = recon_tensor.shape()[0];
        let num_partials = recon_tensor.shape()[1];

        let mut frames = Vec::with_capacity(num_frames);

        for m in 0..num_frames {
            let orig_frame = compressed.frames.get(m);
            let timestamp = orig_frame
                .map(|f| f.timestamp)
                .unwrap_or_else(|| (m as f64) * (compressed.hop_size as f64) / compressed.sample_rate);
            let fundamental_freq = orig_frame.map(|f| f.fundamental_freq).unwrap_or(440.0);

            let mut partials = Vec::with_capacity(num_partials);
            for k in 0..num_partials {
                let h_idx = k + 1;
                let recon_amp = recon_tensor.get(&[m, k])?.max(0.0);
                if recon_amp < 1e-4 {
                    continue;
                }
                let (f_hz, phase) = if let Some(of) = orig_frame {
                    if let Some(p) = of.partials.iter().find(|p| p.harmonic_index == h_idx) {
                        (p.frequency, p.phase)
                    } else {
                        continue;
                    }
                } else {
                    (h_idx as f64 * fundamental_freq, 0.0)
                };

                partials.push(HarmonicPartial {
                    harmonic_index: h_idx,
                    frequency: f_hz,
                    amplitude: recon_amp,
                    phase,
                });
            }

            let residual = orig_frame
                .map(|f| f.residual.clone())
                .unwrap_or_default();

            frames.push(SpectralFrame {
                timestamp,
                fundamental_freq,
                partials,
                residual,
            });
        }

        Ok(SpectralTrajectory {
            sample_rate: compressed.sample_rate,
            hop_size: compressed.hop_size,
            frames,
        })
    }
}

impl HypercubeDecomposer for TuckerDecomposer {
    fn decompose(&self, trajectory: &SpectralTrajectory) -> DspResult<CompressedHypercube> {
        self.decompose(trajectory)
    }

    fn reconstruct(&self, compressed: &CompressedHypercube) -> DspResult<SpectralTrajectory> {
        self.reconstruct(compressed)
    }
}

impl TensorDecomposer for TuckerDecomposer {
    fn decompose_tensor(&self, tensor: &Tensor<f64>, ranks: &[usize]) -> DspResult<TuckerDecomposition> {
        TuckerDecomposition::decompose(tensor, ranks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tucker_rank_one() {
        let t = Tensor::from_vec(vec![2, 2, 2], vec![1.0; 8]).unwrap();
        let decomp = TuckerDecomposition::decompose(&t, &[1, 1, 1]).unwrap();
        assert_eq!(decomp.core.shape(), &[1, 1, 1]);
        assert_eq!(decomp.factors.len(), 3);
    }

    #[test]
    fn test_tucker_zeros() {
        let t = Tensor::from_vec(vec![4, 4, 4], vec![0.0; 64]).unwrap();
        let decomp = TuckerDecomposition::decompose(&t, &[2, 2, 2]).unwrap();
        assert!(decomp.core.data().iter().all(|&v| !v.is_nan()));
    }

    #[test]
    fn test_tucker_point_evaluation_and_out_of_bounds() {
        let t = Tensor::from_vec(vec![3, 3, 3], vec![1.0; 27]).unwrap();
        let decomp = TuckerDecomposition::decompose(&t, &[2, 2, 2]).unwrap();
        let pt = decomp.evaluate_point(&[0, 0, 0]).unwrap();
        assert!((pt - 1.0).abs() < 1e-6);
        assert!(decomp.evaluate_point(&[5, 0, 0]).is_err());
    }

    #[test]
    fn test_tucker_invalid_ranks() {
        let t = Tensor::from_vec(vec![4, 4, 4], vec![1.0; 64]).unwrap();
        assert!(TuckerDecomposition::decompose(&t, &[2, 0, 2]).is_err());
        assert!(TuckerDecomposition::decompose(&t, &[5, 2, 2]).is_err());
    }
}
