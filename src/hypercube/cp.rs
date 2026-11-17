//! Canonical Polyadic Alternating Least Squares (CP-ALS) tensor decomposition.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library, zero external linear algebra crates.

use crate::error::{DspError, DspResult};
use crate::math::{Matrix, Tensor};
use crate::synthetic::noise::Pcg32;
use crate::hypercube::pca::invert_matrix;

/// Canonical Polyadic (CP) decomposition container.
#[derive(Debug, Clone, PartialEq)]
pub struct CpModel {
    /// Factor matrices $A^{(0)}, A^{(1)}, \dots, A^{(N-1)}$ of dimensions $I_n \times R$.
    pub factors: Vec<Matrix<f64>>,
    /// Component scaling weights $\lambda_1, \dots, \lambda_R$.
    pub weights: Vec<f64>,
}

/// Canonical Polyadic Alternating Least Squares engine.
pub struct CpAls;

impl CpAls {
    /// Decomposes an N-dimensional tensor into rank-R CP model via regularized ALS.
    ///
    /// # Errors
    /// Returns `DspError::InvalidParameter` if `rank == 0`.
    /// Returns `DspError::InvalidParameter` if `max_iterations == 0`.
    pub fn decompose(
        tensor: &Tensor<f64>,
        rank: usize,
        max_iterations: usize,
    ) -> DspResult<CpModel> {
        let ndim = tensor.ndim();
        if ndim == 0 {
            return Err(DspError::EmptyInput);
        }

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

        let is_all_zeros = tensor.data().iter().all(|&v| v == 0.0);
        if is_all_zeros {
            let mut factors = Vec::with_capacity(ndim);
            for m in 0..ndim {
                let dim = tensor.shape()[m];
                factors.push(Matrix::zeros(dim, rank)?);
            }
            return Ok(CpModel {
                factors,
                weights: vec![0.0; rank],
            });
        }

        // Deterministic factor initialization using PCG32
        let mut rng = Pcg32::new(101, 77);
        let mut factors = Vec::with_capacity(ndim);

        for m in 0..ndim {
            let dim = tensor.shape()[m];
            let mut mat_data = Vec::with_capacity(dim * rank);
            for _ in 0..(dim * rank) {
                mat_data.push(0.1 + 0.8 * rng.next_f64());
            }
            let mut mat = Matrix::from_vec(dim, rank, mat_data)?;

            // Column normalize
            for r in 0..rank {
                let norm = (0..dim).map(|i| mat[(i, r)].powi(2)).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    for i in 0..dim {
                        mat[(i, r)] /= norm;
                    }
                }
            }
            factors.push(mat);
        }

        let mut weights = vec![1.0; rank];
        const TIKHONOV_LAMBDA: f64 = 1e-8;

        for _ in 0..max_iterations {
            for n in 0..ndim {
                let dim_n = tensor.shape()[n];

                // 1. Gram matrices: S^(m) = A^(m)T * A^(m) (R x R)
                // Compute Hadamard product V = product_{m != n} S^(m)
                let mut v = Matrix::zeros(rank, rank)?;
                for i in 0..rank {
                    for j in 0..rank {
                        v[(i, j)] = 1.0;
                    }
                }

                for m in 0..ndim {
                    if m == n {
                        continue;
                    }
                    let s_m = factors[m].transpose().matmul(&factors[m])?;
                    for i in 0..rank {
                        for j in 0..rank {
                            v[(i, j)] *= s_m[(i, j)];
                        }
                    }
                }

                // Add Tikhonov regularization to diagonal
                for r in 0..rank {
                    v[(r, r)] += TIKHONOV_LAMBDA;
                }

                let v_inv = invert_matrix(&v)?;

                // 2. MTTKRP: W = X_(n) * (A^(N-1) (x) ... (x) A^(0))
                // Compute directly via contraction to prevent Khatri-Rao memory blowup
                let unfolded = tensor.mode_unfold(n)?;
                let cols_unfolded = unfolded.cols();
                let mut w = Matrix::zeros(dim_n, rank)?;

                // Compute unfolding coordinate strides for other modes
                let mut other_modes = Vec::with_capacity(ndim - 1);
                for m in 0..ndim {
                    if m != n {
                        other_modes.push(m);
                    }
                }

                let mut other_strides = vec![1usize; other_modes.len()];
                for i in (0..other_modes.len().saturating_sub(1)).rev() {
                    let next_mode = other_modes[i + 1];
                    other_strides[i] = other_strides[i + 1] * tensor.shape()[next_mode];
                }

                for col_idx in 0..cols_unfolded {
                    // Extract other mode coordinates
                    let mut rem = col_idx;
                    let mut other_coords = Vec::with_capacity(other_modes.len());
                    for (&_m, &stride) in other_modes.iter().zip(other_strides.iter()) {
                        let c = rem / stride;
                        rem %= stride;
                        other_coords.push(c);
                    }

                    for r in 0..rank {
                        let mut kr_prod = 1.0;
                        for (&m, &c) in other_modes.iter().zip(other_coords.iter()) {
                            kr_prod *= factors[m][(c, r)];
                        }

                        for row_i in 0..dim_n {
                            let x_val = unfolded[(row_i, col_idx)];
                            w[(row_i, r)] += x_val * kr_prod;
                        }
                    }
                }

                // 3. Update factor matrix: A^(n) = W * V_inv
                let new_a = w.matmul(&v_inv)?;

                // Column normalization
                let mut norm_a = Matrix::zeros(dim_n, rank)?;
                for r in 0..rank {
                    let norm = (0..dim_n).map(|i| new_a[(i, r)].powi(2)).sum::<f64>().sqrt();
                    weights[r] = norm;
                    let inv_norm = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
                    for i in 0..dim_n {
                        norm_a[(i, r)] = new_a[(i, r)] * inv_norm;
                    }
                }
                factors[n] = norm_a;
            }
        }

        Ok(CpModel { factors, weights })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cp_rank_one() {
        let t = Tensor::from_vec(vec![2, 2, 2], vec![1.0; 8]).unwrap();
        let model = CpAls::decompose(&t, 1, 15).unwrap();
        assert_eq!(model.factors.len(), 3);
        assert_eq!(model.factors[0].cols(), 1);
    }

    #[test]
    fn test_cp_zeros() {
        let t = Tensor::from_vec(vec![3, 3, 3], vec![0.0; 27]).unwrap();
        let model = CpAls::decompose(&t, 1, 10).unwrap();
        assert_eq!(model.factors.len(), 3);
    }

    #[test]
    fn test_cp_singular_regularization() {
        let mut data = vec![0.0; 27];
        for i in 0..9 {
            data[i] = 1.0;
            data[i + 9] = 2.0;
            data[i + 18] = 3.0;
        }
        let t = Tensor::from_vec(vec![3, 3, 3], data).unwrap();
        let res = CpAls::decompose(&t, 2, 10);
        assert!(res.is_ok());
    }

    #[test]
    fn test_cp_invalid_parameters() {
        let t = Tensor::from_vec(vec![2, 2, 2], vec![1.0; 8]).unwrap();
        assert!(CpAls::decompose(&t, 0, 10).is_err());
        assert!(CpAls::decompose(&t, 1, 0).is_err());
    }
}
