//! First-principles Non-Negative Matrix Factorization (NMF) via Lee-Seung multiplicative updates.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library, zero external dependencies.

use crate::error::{DspError, DspResult};
use crate::math::Matrix;
use crate::spectral::{HarmonicPartial, SpectralFrame, SpectralTrajectory};
use crate::synthetic::noise::Pcg32;
use crate::hypercube::coordinates::HypercubeLayout;
use crate::hypercube::storage::CompressedHypercube;
use crate::hypercube::traits::{DimensionalityReducer, HypercubeDecomposer};

/// Direct Non-Negative Matrix Factorization engine.
pub struct Nmf;

impl Nmf {
    /// Factorizes non-negative matrix `V` (M x N) into `(W: M x rank, H: rank x N)`.
    ///
    /// # Errors
    /// Returns `DspError::InvalidParameter` if `rank == 0`.
    /// Returns `DspError::InvalidParameter` if any matrix entry is negative.
    /// Returns `DspError::InvalidParameter` if `max_iterations == 0`.
    pub fn decompose(
        matrix: &Matrix<f64>,
        rank: usize,
        max_iterations: usize,
    ) -> DspResult<(Matrix<f64>, Matrix<f64>)> {
        let rows = matrix.rows();
        let cols = matrix.cols();

        if rank == 0 {
            return Err(DspError::invalid_parameter(
                "rank",
                "Target rank must be >= 1",
            ));
        }

        if max_iterations == 0 {
            return Err(DspError::invalid_parameter(
                "max_iterations",
                "Max iterations must be > 0",
            ));
        }

        for (i, &val) in matrix.as_slice().iter().enumerate() {
            if val < 0.0 {
                return Err(DspError::invalid_parameter(
                    "data",
                    format!("NMF requires non-negative matrix entries; found {} at index {}", val, i),
                ));
            }
        }

        // Fast zero-matrix path
        let is_all_zeros = matrix.as_slice().iter().all(|&v| v == 0.0);
        if is_all_zeros {
            let w = Matrix::zeros(rows, rank)?;
            let h = Matrix::zeros(rank, cols)?;
            return Ok((w, h));
        }

        // Deterministic initialization using internal PCG32 PRNG
        let mut rng = Pcg32::new(42, 54);
        let mut w_data = Vec::with_capacity(rows * rank);
        for _ in 0..(rows * rank) {
            w_data.push(0.1 + 0.8 * rng.next_f64());
        }
        let mut h_data = Vec::with_capacity(rank * cols);
        for _ in 0..(rank * cols) {
            h_data.push(0.1 + 0.8 * rng.next_f64());
        }

        let mut w = Matrix::from_vec(rows, rank, w_data)?;
        let mut h = Matrix::from_vec(rank, cols, h_data)?;

        const EPSILON: f64 = 1e-12;

        for _ in 0..max_iterations {
            // 1. Update H: H <- H * (W^T * V) / (W^T * W * H + EPS)
            let wt = w.transpose();
            let wt_v = wt.matmul(matrix)?;
            let wt_w = wt.matmul(&w)?;
            let wt_w_h = wt_w.matmul(&h)?;

            for k in 0..rank {
                for j in 0..cols {
                    let num = wt_v[(k, j)];
                    let den = wt_w_h[(k, j)] + EPSILON;
                    let factor = (num / den).max(0.0);
                    h[(k, j)] *= factor;
                }
            }

            // 2. Update W: W <- W * (V * H^T) / (W * H * H^T + EPS)
            let ht = h.transpose();
            let v_ht = matrix.matmul(&ht)?;
            let h_ht = h.matmul(&ht)?;
            let w_h_ht = w.matmul(&h_ht)?;

            for i in 0..rows {
                for k in 0..rank {
                    let num = v_ht[(i, k)];
                    let den = w_h_ht[(i, k)] + EPSILON;
                    let factor = (num / den).max(0.0);
                    w[(i, k)] *= factor;
                }
            }
        }

        Ok((w, h))
    }
}

impl DimensionalityReducer for Nmf {
    fn reduce(&self, matrix: &Matrix<f64>, rank: usize) -> DspResult<(Matrix<f64>, Matrix<f64>)> {
        Nmf::decompose(matrix, rank, 40)
    }
}

/// Trajectory-level NMF decomposer for STFT spectral trajectories.
#[derive(Debug, Clone)]
pub struct NmfDecomposer {
    /// Compression rank.
    pub rank: usize,
    /// Maximum multiplicative update iterations.
    pub max_iterations: usize,
}

impl NmfDecomposer {
    /// Construct a new NMF decomposer.
    pub fn new(rank: usize, max_iterations: usize) -> Self {
        Self {
            rank,
            max_iterations,
        }
    }

    /// Decomposes a spectral trajectory into CompressedHypercube using NMF.
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
                kind: "NMF".to_string(),
                rank: self.rank,
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

        // Construct matrix M x K of non-negative partial amplitudes aligned by harmonic index
        let mut data = vec![0.0; num_frames * max_partials];
        for (m, frame) in trajectory.frames.iter().enumerate() {
            for p in &frame.partials {
                if p.harmonic_index > 0 && p.harmonic_index <= max_partials {
                    data[m * max_partials + (p.harmonic_index - 1)] = p.amplitude.max(0.0);
                }
            }
        }

        let matrix = Matrix::from_vec(num_frames, max_partials, data)?;
        let effective_rank = self.rank.min(num_frames).min(max_partials).max(1);
        let (w, h) = Nmf::decompose(&matrix, effective_rank, self.max_iterations)?;

        let layout = HypercubeLayout::new(vec![num_frames, max_partials])?;

        Ok(CompressedHypercube {
            layout,
            frames: trajectory.frames.clone(),
            kind: "NMF".to_string(),
            rank: effective_rank,
            sample_rate: trajectory.sample_rate,
            hop_size: trajectory.hop_size,
            timestamps: trajectory.frames.iter().map(|f| f.timestamp).collect(),
            fundamental_freqs: trajectory.frames.iter().map(|f| f.fundamental_freq).collect(),
            weights: Some(w),
            basis: Some(h),
            tucker: None,
            residuals: trajectory.frames.iter().map(|f| f.residual.clone()).collect(),
        })
    }

    /// Reconstructs spectral trajectory from CompressedHypercube.
    pub fn reconstruct(&self, compressed: &CompressedHypercube) -> DspResult<SpectralTrajectory> {
        let (w, h) = match (&compressed.weights, &compressed.basis) {
            (Some(w), Some(h)) => (w, h),
            _ => {
                return Ok(SpectralTrajectory {
                    sample_rate: compressed.sample_rate,
                    hop_size: compressed.hop_size,
                    frames: compressed.frames.clone(),
                });
            }
        };

        let recon_matrix = w.matmul(h)?;
        let num_frames = recon_matrix.rows();
        let num_partials = recon_matrix.cols();

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
                let recon_amp = recon_matrix[(m, k)].max(0.0);
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

impl HypercubeDecomposer for NmfDecomposer {
    fn decompose(&self, trajectory: &SpectralTrajectory) -> DspResult<CompressedHypercube> {
        self.decompose(trajectory)
    }

    fn reconstruct(&self, compressed: &CompressedHypercube) -> DspResult<SpectralTrajectory> {
        self.reconstruct(compressed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nmf_zeros() {
        let x = Matrix::from_vec(8, 8, vec![0.0; 64]).unwrap();
        let (w, h) = Nmf::decompose(&x, 2, 20).unwrap();
        assert!(w.as_slice().iter().all(|&v| !v.is_nan() && v >= 0.0));
        assert!(h.as_slice().iter().all(|&v| !v.is_nan() && v >= 0.0));
    }

    #[test]
    fn test_nmf_negative_rejection() {
        let x = Matrix::from_vec(2, 2, vec![1.0, -0.5, 0.5, 1.0]).unwrap();
        assert!(Nmf::decompose(&x, 1, 10).is_err());
    }

    #[test]
    fn test_nmf_target_rank_zero() {
        let x = Matrix::from_vec(2, 2, vec![1.0; 4]).unwrap();
        assert!(Nmf::decompose(&x, 0, 10).is_err());
    }

    #[test]
    fn test_nmf_rank_one() {
        let x = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0]).unwrap();
        let (w, h) = Nmf::decompose(&x, 1, 25).unwrap();
        assert!(w.as_slice().iter().all(|&v| v >= 0.0));
        assert!(h.as_slice().iter().all(|&v| v >= 0.0));
    }
}
