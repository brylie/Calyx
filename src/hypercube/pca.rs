//! First-principles Principal Component Analysis (PCA) and SVD decomposition.
//!
//! Evaluates symmetric eigendecomposition via cyclic Jacobi rotation.
//! Complies strictly with Demo Integrity Mode: 100% pure standard library, zero external linear algebra crates.

use crate::error::{DspError, DspResult};
use crate::math::Matrix;
use crate::spectral::{HarmonicPartial, SpectralFrame, SpectralTrajectory};
use crate::hypercube::coordinates::HypercubeLayout;
use crate::hypercube::storage::CompressedHypercube;
use crate::hypercube::traits::{DimensionalityReducer, HypercubeDecomposer};

/// Computes all eigenvalues and orthonormal eigenvectors of an N x N symmetric matrix.
///
/// Returns (eigenvalues: Vec<f64>, eigenvectors: Matrix<f64>) sorted descending by eigenvalue.
/// Columns of `eigenvectors` correspond to the sorted eigenvalues.
pub(crate) fn symmetric_eigen(a: &Matrix<f64>) -> DspResult<(Vec<f64>, Matrix<f64>)> {
    let n = a.rows();
    if n != a.cols() {
        return Err(DspError::dimension_mismatch(vec![n, n], vec![a.rows(), a.cols()]));
    }
    if n == 0 {
        return Err(DspError::EmptyInput);
    }
    if n == 1 {
        let val = a[(0, 0)].max(0.0);
        let v = Matrix::identity(1)?;
        return Ok((vec![val], v));
    }

    let mut d = a.clone();
    let mut v = Matrix::identity(n)?;

    let max_sweeps = 100;
    for _ in 0..max_sweeps {
        let mut off_diag = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diag += d[(i, j)].powi(2);
            }
        }
        if off_diag.sqrt() < 1e-13 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = d[(p, q)];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let app = d[(p, p)];
                let aqq = d[(q, q)];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                d[(p, p)] -= t * apq;
                d[(q, q)] += t * apq;
                d[(p, q)] = 0.0;
                d[(q, p)] = 0.0;

                for r in 0..n {
                    if r != p && r != q {
                        let arp = d[(r, p)];
                        let arq = d[(r, q)];
                        d[(r, p)] = c * arp - s * arq;
                        d[(p, r)] = d[(r, p)];
                        d[(r, q)] = s * arp + c * arq;
                        d[(q, r)] = d[(r, q)];
                    }
                }
                for r in 0..n {
                    let vrp = v[(r, p)];
                    let vrq = v[(r, q)];
                    v[(r, p)] = c * vrp - s * vrq;
                    v[(r, q)] = s * vrp + c * vrq;
                }
            }
        }
    }

    // Extract diagonal eigenvalues and sort descending
    let mut eig_pairs: Vec<(f64, usize)> = (0..n).map(|i| (d[(i, i)].max(0.0), i)).collect();
    eig_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut sorted_vals = Vec::with_capacity(n);
    let mut sorted_vecs = Matrix::zeros(n, n)?;
    for (new_col, &(val, old_col)) in eig_pairs.iter().enumerate() {
        sorted_vals.push(val);
        for r in 0..n {
            sorted_vecs[(r, new_col)] = v[(r, old_col)];
        }
    }

    Ok((sorted_vals, sorted_vecs))
}

/// Inverts an N x N matrix via Gauss-Jordan elimination with partial pivoting.
pub(crate) fn invert_matrix(a: &Matrix<f64>) -> DspResult<Matrix<f64>> {
    let n = a.rows();
    if n != a.cols() {
        return Err(DspError::dimension_mismatch(vec![n, n], vec![a.rows(), a.cols()]));
    }
    if n == 0 {
        return Err(DspError::EmptyInput);
    }
    if n == 1 {
        let v = a[(0, 0)];
        if v.abs() < 1e-15 {
            return Err(DspError::numerical_instability("Matrix is singular (1x1 near zero)"));
        }
        return Matrix::from_vec(1, 1, vec![1.0 / v]);
    }

    // Augmented matrix [A | I]
    let mut aug = vec![0.0; n * 2 * n];
    for r in 0..n {
        for c in 0..n {
            aug[r * (2 * n) + c] = a[(r, c)];
        }
        aug[r * (2 * n) + (n + r)] = 1.0;
    }

    for k in 0..n {
        // Pivot selection
        let mut pivot_row = k;
        let mut max_val = aug[k * (2 * n) + k].abs();
        for r in (k + 1)..n {
            let val = aug[r * (2 * n) + k].abs();
            if val > max_val {
                max_val = val;
                pivot_row = r;
            }
        }
        if max_val < 1e-14 {
            return Err(DspError::numerical_instability("Matrix is singular or ill-conditioned"));
        }
        if pivot_row != k {
            for c in 0..(2 * n) {
                aug.swap(k * (2 * n) + c, pivot_row * (2 * n) + c);
            }
        }

        let pivot = aug[k * (2 * n) + k];
        for c in 0..(2 * n) {
            aug[k * (2 * n) + c] /= pivot;
        }

        for r in 0..n {
            if r != k {
                let factor = aug[r * (2 * n) + k];
                for c in 0..(2 * n) {
                    aug[r * (2 * n) + c] -= factor * aug[k * (2 * n) + c];
                }
            }
        }
    }

    let mut inv = Matrix::zeros(n, n)?;
    for r in 0..n {
        for c in 0..n {
            inv[(r, c)] = aug[r * (2 * n) + (n + c)];
        }
    }
    Ok(inv)
}

/// Direct matrix Principal Component Analysis engine.
pub struct Pca;

impl Pca {
    /// Decomposes data matrix `X` (M x N) into `(weights: M x target_dim, basis: target_dim x N)`.
    ///
    /// # Errors
    /// Returns `DspError::InvalidParameter` if `target_dim == 0`.
    /// Returns `DspError::DimensionMismatch` if `target_dim > min(M, N)`.
    pub fn decompose(
        matrix: &Matrix<f64>,
        target_dim: usize,
    ) -> DspResult<(Matrix<f64>, Matrix<f64>)> {
        let rows = matrix.rows();
        let cols = matrix.cols();

        if target_dim == 0 {
            return Err(DspError::invalid_parameter(
                "target_dim",
                "Target dimension must be at least 1",
            ));
        }

        let max_rank = rows.min(cols);
        if target_dim > max_rank {
            return Err(DspError::dimension_mismatch(
                vec![max_rank],
                vec![target_dim],
            ));
        }

        // Special case: 1x1 matrix
        if rows == 1 && cols == 1 {
            let weights = matrix.clone();
            let basis = Matrix::from_vec(1, 1, vec![1.0])?;
            return Ok((weights, basis));
        }

        // Check if matrix is all zeros
        let is_all_zeros = matrix.as_slice().iter().all(|&v| v == 0.0);
        if is_all_zeros {
            let weights = Matrix::zeros(rows, target_dim)?;
            let basis = Matrix::zeros(target_dim, cols)?;
            return Ok((weights, basis));
        }

        if cols <= rows {
            // Covariance / Gram along columns: C = X^T * X (cols x cols)
            let gram = matrix.transpose().matmul(matrix)?;
            let (_eig_vals, eig_vecs) = symmetric_eigen(&gram)?;

            // Basis rows are the top target_dim column eigenvectors transposed: (target_dim x cols)
            let mut basis = Matrix::zeros(target_dim, cols)?;
            for k in 0..target_dim {
                for c in 0..cols {
                    basis[(k, c)] = eig_vecs[(c, k)];
                }
            }

            // Weights: W = X * B^T (rows x target_dim)
            let weights = matrix.matmul(&basis.transpose())?;
            Ok((weights, basis))
        } else {
            // Covariance / Gram along rows: C = X * X^T (rows x rows)
            let gram = matrix.matmul(&matrix.transpose())?;
            let (eig_vals, eig_vecs) = symmetric_eigen(&gram)?;

            let mut basis = Matrix::zeros(target_dim, cols)?;
            let x_t = matrix.transpose();

            for k in 0..target_dim {
                let lambda = eig_vals[k];
                if lambda > 1e-12 {
                    let mut u_k = Matrix::zeros(rows, 1)?;
                    for r in 0..rows {
                        u_k[(r, 0)] = eig_vecs[(r, k)];
                    }
                    let v_k = x_t.matmul(&u_k)?;
                    let norm = (0..cols).map(|c| v_k[(c, 0)].powi(2)).sum::<f64>().sqrt();
                    let inv_norm = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
                    for c in 0..cols {
                        basis[(k, c)] = v_k[(c, 0)] * inv_norm;
                    }
                } else {
                    // Standard unit basis fallback orthogonalized
                    if k < cols {
                        basis[(k, k)] = 1.0;
                    }
                }
            }

            let weights = matrix.matmul(&basis.transpose())?;
            Ok((weights, basis))
        }
    }
}

impl DimensionalityReducer for Pca {
    fn reduce(&self, matrix: &Matrix<f64>, rank: usize) -> DspResult<(Matrix<f64>, Matrix<f64>)> {
        Pca::decompose(matrix, rank)
    }
}

/// Trajectory-level PCA decomposer for STFT spectral trajectories.
#[derive(Debug, Clone)]
pub struct PcaDecomposer {
    /// Target compression rank.
    pub rank: usize,
}

impl PcaDecomposer {
    /// Construct a new PCA decomposer with the specified rank.
    pub fn new(rank: usize) -> Self {
        Self { rank }
    }

    /// Decomposes spectral trajectory into CompressedHypercube.
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
                kind: "PCA".to_string(),
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

        // Construct matrix M x K of partial amplitudes aligned by harmonic index
        let mut data = vec![0.0; num_frames * max_partials];
        for (m, frame) in trajectory.frames.iter().enumerate() {
            for p in &frame.partials {
                if p.harmonic_index > 0 && p.harmonic_index <= max_partials {
                    data[m * max_partials + (p.harmonic_index - 1)] = p.amplitude;
                }
            }
        }

        let matrix = Matrix::from_vec(num_frames, max_partials, data)?;
        let effective_rank = self.rank.min(num_frames).min(max_partials).max(1);
        let (weights, basis) = Pca::decompose(&matrix, effective_rank)?;

        let layout = HypercubeLayout::new(vec![num_frames, max_partials])?;

        Ok(CompressedHypercube {
            layout,
            frames: trajectory.frames.clone(),
            kind: "PCA".to_string(),
            rank: effective_rank,
            sample_rate: trajectory.sample_rate,
            hop_size: trajectory.hop_size,
            timestamps: trajectory.frames.iter().map(|f| f.timestamp).collect(),
            fundamental_freqs: trajectory.frames.iter().map(|f| f.fundamental_freq).collect(),
            weights: Some(weights),
            basis: Some(basis),
            tucker: None,
            residuals: trajectory.frames.iter().map(|f| f.residual.clone()).collect(),
        })
    }

    /// Reconstructs spectral trajectory from CompressedHypercube.
    pub fn reconstruct(&self, compressed: &CompressedHypercube) -> DspResult<SpectralTrajectory> {
        let (weights, basis) = match (&compressed.weights, &compressed.basis) {
            (Some(w), Some(b)) => (w, b),
            _ => {
                return Ok(SpectralTrajectory {
                    sample_rate: compressed.sample_rate,
                    hop_size: compressed.hop_size,
                    frames: compressed.frames.clone(),
                });
            }
        };

        let recon_matrix = weights.matmul(basis)?;
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

impl HypercubeDecomposer for PcaDecomposer {
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
    fn test_pca_1x1() {
        let x = Matrix::from_vec(1, 1, vec![5.0]).unwrap();
        let (weights, basis) = Pca::decompose(&x, 1).unwrap();
        assert_eq!(weights.rows(), 1);
        assert_eq!(basis.cols(), 1);
        let recon = weights.matmul(&basis).unwrap();
        assert!((recon[(0, 0)] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_pca_zeros() {
        let x = Matrix::from_vec(4, 4, vec![0.0; 16]).unwrap();
        let (weights, basis) = Pca::decompose(&x, 2).unwrap();
        assert!(weights.as_slice().iter().all(|&v| !v.is_nan()));
        assert!(basis.as_slice().iter().all(|&v| !v.is_nan()));
    }

    #[test]
    fn test_pca_collinear() {
        let row = vec![1.0, 2.0, 3.0, 4.0];
        let mut data = Vec::with_capacity(16);
        for _ in 0..4 {
            data.extend_from_slice(&row);
        }
        let x = Matrix::from_vec(4, 4, data).unwrap();
        let (weights, basis) = Pca::decompose(&x, 2).unwrap();
        assert_eq!(weights.cols(), 2);
        assert_eq!(basis.rows(), 2);
    }

    #[test]
    fn test_pca_target_dim_bounds() {
        let x = Matrix::from_vec(3, 4, vec![1.0; 12]).unwrap();
        assert!(Pca::decompose(&x, 0).is_err());
        assert!(Pca::decompose(&x, 5).is_err());
    }
}
