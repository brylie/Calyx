//! First-principles multi-index strided tensor mathematics.
//!
//! Complies strictly with Demo Integrity Mode: zero external dependencies,
//! 100% pure standard library, safe Rust.
//! Implements Kolda & Bader (2009) mode matricization (unfolding) and mode products.

use crate::error::{DspError, DspResult};
use crate::math::matrix::Matrix;

/// An $N$-dimensional strided tensor stored in generalized row-major (C-order) contiguous layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T = f64> {
    /// Dimension sizes $(I_0, I_1, \dots, I_{N-1})$
    pub shape: Vec<usize>,
    /// Memory strides in units of elements
    pub strides: Vec<usize>,
    /// Flat row-major contiguous data elements
    pub data: Vec<T>,
}

impl Tensor<f64> {
    /// Computes generalized row-major (C-order) strides from a shape vector.
    ///
    /// The rightmost dimension varies fastest with stride 1.
    /// Strides accumulate right-to-left: $S_k = S_{k+1} \times I_{k+1}$.
    pub fn compute_c_strides(shape: &[usize]) -> Vec<usize> {
        let n = shape.len();
        if n == 0 {
            return Vec::new();
        }
        let mut strides = vec![1; n];
        for k in (0..n - 1).rev() {
            strides[k] = strides[k + 1] * shape[k + 1];
        }
        strides
    }

    /// Constructs a Tensor from shape and flat vector data.
    ///
    /// # Errors
    /// - Returns `DspError::EmptyInput` if `shape.is_empty()`.
    /// - Returns `DspError::InvalidParameter` if any dimension is 0 or if total elements overflow `usize`.
    /// - Returns `DspError::BufferLengthMismatch` if `data.len() != prod(shape)`.
    pub fn from_vec(shape: Vec<usize>, data: Vec<f64>) -> DspResult<Self> {
        if shape.is_empty() {
            return Err(DspError::EmptyInput);
        }
        let mut total_elements: usize = 1;
        for &dim in &shape {
            if dim == 0 {
                return Err(DspError::invalid_parameter(
                    "shape",
                    "Tensor dimension sizes must be positive non-zero integers",
                ));
            }
            total_elements = total_elements.checked_mul(dim).ok_or_else(|| {
                DspError::invalid_parameter("shape", "Total tensor element count overflowed usize")
            })?;
        }
        if data.len() != total_elements {
            return Err(DspError::buffer_length_mismatch(total_elements, data.len()));
        }
        let strides = Self::compute_c_strides(&shape);
        Ok(Self {
            shape,
            strides,
            data,
        })
    }

    /// Constructs a Tensor from shape and flat vector data.
    ///
    /// Symmetrical constructor alias to [`from_vec`](Self::from_vec).
    #[inline]
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> DspResult<Self> {
        Self::from_vec(shape, data)
    }

    /// Creates a tensor of the given shape filled with zeros.
    ///
    /// # Errors
    /// Returns `DspError::EmptyInput` if `shape.is_empty()`, or `DspError::InvalidParameter`
    /// if any dimension is 0.
    pub fn zeros(shape: Vec<usize>) -> DspResult<Self> {
        if shape.is_empty() {
            return Err(DspError::EmptyInput);
        }
        let mut total_elements: usize = 1;
        for &dim in &shape {
            if dim == 0 {
                return Err(DspError::invalid_parameter(
                    "shape",
                    "Tensor dimension sizes must be positive non-zero integers",
                ));
            }
            total_elements = total_elements.checked_mul(dim).ok_or_else(|| {
                DspError::invalid_parameter("shape", "Total tensor element count overflowed usize")
            })?;
        }
        let strides = Self::compute_c_strides(&shape);
        Ok(Self {
            shape,
            strides,
            data: vec![0.0; total_elements],
        })
    }

    /// Constructs a tensor using a generator closure $f(\&\text{indices})$.
    pub fn from_shape_fn<F>(shape: Vec<usize>, mut f: F) -> DspResult<Self>
    where
        F: FnMut(&[usize]) -> f64,
    {
        let mut tensor = Self::zeros(shape)?;
        let n_modes = tensor.shape.len();
        let mut current_idx = vec![0; n_modes];

        for item in &mut tensor.data {
            *item = f(&current_idx);
            for k in (0..n_modes).rev() {
                current_idx[k] += 1;
                if current_idx[k] < tensor.shape[k] {
                    break;
                }
                current_idx[k] = 0;
            }
        }
        Ok(tensor)
    }

    /// Tensor order / number of dimensions ($N$).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Reference to dimension sizes.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Reference to dimension strides.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if tensor contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a borrowed slice of the flat underlying data.
    #[inline]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Returns a mutable slice of the flat underlying data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Computes the flat 1D offset for a multi-index vector.
    ///
    /// # Errors
    /// Returns `DspError::DimensionMismatch` if the number of indices does not match
    /// tensor rank, or if any index is out of bounds for its respective axis.
    #[inline]
    pub fn flat_index(&self, indices: &[usize]) -> DspResult<usize> {
        if indices.len() != self.shape.len() {
            return Err(DspError::dimension_mismatch(
                self.shape.clone(),
                indices.to_vec(),
            ));
        }
        let mut offset = 0;
        for i in 0..indices.len() {
            let idx = indices[i];
            let dim = self.shape[i];
            if idx >= dim {
                return Err(DspError::dimension_mismatch(
                    self.shape.clone(),
                    indices.to_vec(),
                ));
            }
            offset += idx * self.strides[i];
        }
        Ok(offset)
    }

    /// Safe immutable multi-index access.
    #[inline]
    pub fn get(&self, indices: &[usize]) -> DspResult<f64> {
        let offset = self.flat_index(indices)?;
        Ok(self.data[offset])
    }

    /// Safe mutable multi-index access.
    #[inline]
    pub fn get_mut(&mut self, indices: &[usize]) -> DspResult<&mut f64> {
        let offset = self.flat_index(indices)?;
        Ok(&mut self.data[offset])
    }

    /// Safe element mutation.
    #[inline]
    pub fn set(&mut self, indices: &[usize], val: f64) -> DspResult<()> {
        let offset = self.flat_index(indices)?;
        self.data[offset] = val;
        Ok(())
    }

    /// Mode-$n$ matricization (unfolding) according to Kolda & Bader (2009) standard.
    ///
    /// Maps tensor $\mathcal{X} \in \mathbb{R}^{I_0 \times \dots \times I_{N-1}}$ into
    /// matrix $X_{(n)} \in \mathbb{R}^{I_n \times J}$, where $J = \prod_{k \neq n} I_k$.
    ///
    /// Column indices are computed via equation (2.2) of Kolda & Bader (2009):
    /// $$c = \sum_{p=0}^{N-2} i_{k_p} \cdot w_p, \quad w_0 = 1, \; w_p = \prod_{m=0}^{p-1} I_{k_m}$$
    pub fn mode_unfold(&self, mode: usize) -> DspResult<Matrix<f64>> {
        let n_modes = self.shape.len();
        if mode >= n_modes {
            return Err(DspError::invalid_parameter(
                "mode",
                format!("Mode {} out of bounds for tensor of order {}", mode, n_modes),
            ));
        }

        let rows = self.shape[mode];
        let cols: usize = self
            .shape
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != mode)
            .map(|(_, &d)| d)
            .product();

        let rem_dims: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != mode)
            .map(|(_, &d)| d)
            .collect();
        let rem_modes: Vec<usize> = (0..n_modes).filter(|&k| k != mode).collect();

        let mut col_weights = vec![1; rem_dims.len()];
        for p in (0..rem_dims.len().saturating_sub(1)).rev() {
            col_weights[p] = col_weights[p + 1] * rem_dims[p + 1];
        }

        let mut matrix_data = vec![0.0; rows * cols];
        let mut current_idx = vec![0; n_modes];

        for &val in &self.data {
            let r = current_idx[mode];
            let mut c = 0;
            for (p, &orig_mode) in rem_modes.iter().enumerate() {
                c += current_idx[orig_mode] * col_weights[p];
            }
            matrix_data[r * cols + c] = val;

            // Increment multi-index in C-order
            for k in (0..n_modes).rev() {
                current_idx[k] += 1;
                if current_idx[k] < self.shape[k] {
                    break;
                }
                current_idx[k] = 0;
            }
        }

        Matrix::from_vec(rows, cols, matrix_data)
    }

    /// Inverse matricization (folding).
    ///
    /// Reconstructs a tensor of the given `target_shape` from an unfolded mode-$n$ matrix.
    /// Invertible: `Tensor::fold(&t.mode_unfold(mode)?, mode, t.shape())? == t`.
    pub fn fold(matrix: &Matrix<f64>, mode: usize, target_shape: &[usize]) -> DspResult<Self> {
        let n_modes = target_shape.len();
        if mode >= n_modes {
            return Err(DspError::invalid_parameter(
                "mode",
                format!("Mode {} out of bounds for tensor of order {}", mode, n_modes),
            ));
        }
        let expected_rows = target_shape[mode];
        let expected_cols: usize = target_shape
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != mode)
            .map(|(_, &d)| d)
            .product();

        if matrix.rows != expected_rows || matrix.cols != expected_cols {
            return Err(DspError::dimension_mismatch(
                vec![expected_rows, expected_cols],
                vec![matrix.rows, matrix.cols],
            ));
        }

        let rem_dims: Vec<usize> = target_shape
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != mode)
            .map(|(_, &d)| d)
            .collect();
        let rem_modes: Vec<usize> = (0..n_modes).filter(|&k| k != mode).collect();

        let mut col_weights = vec![1; rem_dims.len()];
        for p in (0..rem_dims.len().saturating_sub(1)).rev() {
            col_weights[p] = col_weights[p + 1] * rem_dims[p + 1];
        }

        let total = expected_rows * expected_cols;
        let mut tensor_data = vec![0.0; total];
        let mut current_idx = vec![0; n_modes];

        for item in &mut tensor_data {
            let r = current_idx[mode];
            let mut c = 0;
            for (p, &orig_mode) in rem_modes.iter().enumerate() {
                c += current_idx[orig_mode] * col_weights[p];
            }
            *item = matrix.data[r * matrix.cols + c];

            for k in (0..n_modes).rev() {
                current_idx[k] += 1;
                if current_idx[k] < target_shape[k] {
                    break;
                }
                current_idx[k] = 0;
            }
        }

        Self::from_vec(target_shape.to_vec(), tensor_data)
    }

    /// Mode-$n$ tensor-matrix product: $\mathcal{Y} = \mathcal{X} \times_n M$.
    ///
    /// Replaces dimension `mode` (size $I_n$) with `matrix.rows` (size $K$).
    /// Implemented via the matricized identity $Y_{(n)} = M \cdot X_{(n)}$ (Kolda & Bader 2009).
    pub fn mode_product(&self, mode: usize, matrix: &Matrix<f64>) -> DspResult<Self> {
        let n_modes = self.shape.len();
        if mode >= n_modes {
            return Err(DspError::invalid_parameter(
                "mode",
                format!("Mode {} out of bounds for tensor of order {}", mode, n_modes),
            ));
        }
        if matrix.cols != self.shape[mode] {
            return Err(DspError::dimension_mismatch(
                vec![self.shape[mode]],
                vec![matrix.cols],
            ));
        }

        let x_n = self.mode_unfold(mode)?;
        let y_n = matrix.matmul(&x_n)?;

        let mut target_shape = self.shape.clone();
        target_shape[mode] = matrix.rows;

        Self::fold(&y_n, mode, &target_shape)
    }

    /// Computes the Frobenius norm of the tensor: $\sqrt{\sum x^2}$.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-12;

    #[test]
    fn test_tensor_strides_and_indexing() {
        // 3D: [2, 3, 4] -> strides [12, 4, 1]
        let strides_3d = Tensor::compute_c_strides(&[2, 3, 4]);
        assert_eq!(strides_3d, vec![12, 4, 1]);

        // 1D: [5] -> strides [1]
        let strides_1d = Tensor::compute_c_strides(&[5]);
        assert_eq!(strides_1d, vec![1]);

        let total = 2 * 3 * 4;
        let data: Vec<f64> = (0..total).map(|x| x as f64).collect();
        let mut t = Tensor::from_vec(vec![2, 3, 4], data).unwrap();

        assert_eq!(t.ndim(), 3);
        assert_eq!(t.len(), 24);
        assert!(!t.is_empty());

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let expected = (i * 12 + j * 4 + k) as f64;
                    assert_eq!(t.get(&[i, j, k]).unwrap(), expected);
                }
            }
        }

        t.set(&[1, 2, 3], 999.0).unwrap();
        assert_eq!(t.get(&[1, 2, 3]).unwrap(), 999.0);
    }

    #[test]
    fn test_tensor_bounds_validation() {
        assert_eq!(Tensor::zeros(vec![]), Err(DspError::EmptyInput));
        assert_eq!(
            Tensor::zeros(vec![2, 0, 3]),
            Err(DspError::invalid_parameter(
                "shape",
                "Tensor dimension sizes must be positive non-zero integers"
            ))
        );

        let t = Tensor::zeros(vec![2, 3, 4]).unwrap();
        // Wrong rank
        assert!(t.get(&[1, 1]).is_err());
        assert!(t.get(&[1, 1, 1, 1]).is_err());
        // Axis out of bounds
        assert!(t.get(&[2, 1, 1]).is_err());
        assert!(t.get(&[1, 3, 1]).is_err());
        assert!(t.get(&[1, 1, 4]).is_err());
    }

    #[test]
    fn test_tensor_mode_unfold_2d_equivalence() {
        // For a 2D tensor [M, N], mode_unfold(0) is the matrix itself (M x N),
        // and mode_unfold(1) is the transpose (N x M).
        let m = 3;
        let n = 4;
        let data: Vec<f64> = (0..m * n).map(|x| x as f64).collect();
        let t = Tensor::from_vec(vec![m, n], data.clone()).unwrap();

        let mat0 = t.mode_unfold(0).unwrap();
        assert_eq!(mat0.rows, m);
        assert_eq!(mat0.cols, n);
        assert_eq!(mat0.data, data);

        let mat1 = t.mode_unfold(1).unwrap();
        assert_eq!(mat1.rows, n);
        assert_eq!(mat1.cols, m);

        let mat_orig = Matrix::from_vec(m, n, data).unwrap();
        assert_eq!(mat1, mat_orig.transpose());
    }

    #[test]
    fn test_tensor_mode_unfold_3d_kolda_bader() {
        let shape = vec![2, 3, 4];
        let total = 2 * 3 * 4;
        let data: Vec<f64> = (0..total).map(|x| x as f64).collect();
        let t = Tensor::from_vec(shape, data).unwrap();

        // Mode 0: 2 x 12
        let u0 = t.mode_unfold(0).unwrap();
        assert_eq!(u0.rows, 2);
        assert_eq!(u0.cols, 12);

        // Mode 1: 3 x 8
        let u1 = t.mode_unfold(1).unwrap();
        assert_eq!(u1.rows, 3);
        assert_eq!(u1.cols, 8);

        // Mode 2: 4 x 6
        let u2 = t.mode_unfold(2).unwrap();
        assert_eq!(u2.rows, 4);
        assert_eq!(u2.cols, 6);

        // Verify exact column mapping for mode 1:
        // rem_modes = [0, 2], rem_dims = [2, 4], col_weights = [4, 1]
        // column index c = i_0 * 4 + i_2 * 1
        // element t(i_0, i_1, i_2) -> u1(i_1, i_0 * 4 + i_2)
        for i0 in 0..2 {
            for i1 in 0..3 {
                for i2 in 0..4 {
                    let val = t.get(&[i0, i1, i2]).unwrap();
                    let c = i0 * 4 + i2;
                    assert_eq!(u1.get(i1, c).unwrap(), val);
                }
            }
        }
    }

    #[test]
    fn test_tensor_fold_roundtrip() {
        // 3D tensor roundtrip
        let shape = vec![3, 4, 5];
        let t = Tensor::from_shape_fn(shape.clone(), |idx| {
            (idx[0] * 100 + idx[1] * 10 + idx[2]) as f64
        })
        .unwrap();

        for mode in 0..3 {
            let unfolded = t.mode_unfold(mode).unwrap();
            let folded = Tensor::fold(&unfolded, mode, &shape).unwrap();
            assert_eq!(folded, t);
        }

        // 5D hypercube tensor roundtrip
        let shape_5d = vec![2, 3, 2, 4, 2];
        let t_5d = Tensor::from_shape_fn(shape_5d.clone(), |idx| {
            idx.iter().enumerate().map(|(i, &x)| x * (i + 1) * 7).sum::<usize>() as f64
        })
        .unwrap();

        for mode in 0..5 {
            let unfolded = t_5d.mode_unfold(mode).unwrap();
            let folded = Tensor::fold(&unfolded, mode, &shape_5d).unwrap();
            assert_eq!(folded, t_5d);
        }
    }

    #[test]
    fn test_mode_product_identity() {
        let shape = vec![2, 3, 4];
        let t = Tensor::from_shape_fn(shape.clone(), |idx| {
            (idx[0] * 20 + idx[1] * 5 + idx[2]) as f64
        })
        .unwrap();

        for mode in 0..3 {
            let dim = shape[mode];
            let eye = Matrix::identity(dim).unwrap();
            let prod = t.mode_product(mode, &eye).unwrap();
            assert_eq!(prod, t);
        }
    }

    #[test]
    fn test_mode_product_dimension_change() {
        // Tensor [2, 3, 4] mode-1 product with matrix (5 x 3) -> Tensor [2, 5, 4]
        let t = Tensor::from_shape_fn(vec![2, 3, 4], |idx| {
            (idx[0] + idx[1] + idx[2]) as f64
        })
        .unwrap();
        let m = Matrix::from_shape_fn(5, 3, |r, c| (r * 3 + c + 1) as f64).unwrap();

        let y = t.mode_product(1, &m).unwrap();
        assert_eq!(y.shape(), &[2, 5, 4]);

        // Verify analytical value at [1, 2, 3]:
        // y[1, 2, 3] = sum_{k=0..2} t[1, k, 3] * m[2, k]
        let mut expected = 0.0;
        for k in 0..3 {
            expected += t.get(&[1, k, 3]).unwrap() * m.get(2, k).unwrap();
        }
        assert!((y.get(&[1, 2, 3]).unwrap() - expected).abs() < EPSILON);
    }

    #[test]
    fn test_mode_product_commutativity() {
        // X x_p A x_q B == X x_q B x_p A for distinct modes p != q
        let t = Tensor::from_shape_fn(vec![3, 4, 3], |idx| {
            (idx[0] * 12 + idx[1] * 3 + idx[2] + 1) as f64
        })
        .unwrap();

        // Mode 0 matrix: 2 x 3
        let a = Matrix::from_shape_fn(2, 3, |r, c| (r + c + 1) as f64).unwrap();
        // Mode 1 matrix: 5 x 4
        let b = Matrix::from_shape_fn(5, 4, |r, c| (r * 2 + c + 1) as f64).unwrap();

        let prod_ab = t.mode_product(0, &a).unwrap().mode_product(1, &b).unwrap();
        let prod_ba = t.mode_product(1, &b).unwrap().mode_product(0, &a).unwrap();

        assert_eq!(prod_ab.shape(), &[2, 5, 3]);
        assert_eq!(prod_ba.shape(), &[2, 5, 3]);

        for i in 0..prod_ab.len() {
            assert!((prod_ab.data[i] - prod_ba.data[i]).abs() < EPSILON);
        }
    }

    #[test]
    fn test_mode_product_associativity() {
        // X x_n A x_n B == X x_n (B * A)
        let t = Tensor::from_shape_fn(vec![2, 3, 2], |idx| {
            (idx[0] + idx[1] * 2 + idx[2]) as f64
        })
        .unwrap();

        // Mode 1: A is 4 x 3, B is 5 x 4 -> B * A is 5 x 3
        let a = Matrix::from_shape_fn(4, 3, |r, c| (r + c + 1) as f64).unwrap();
        let b = Matrix::from_shape_fn(5, 4, |r, c| (r * 2 + c + 1) as f64).unwrap();

        let chained = t.mode_product(1, &a).unwrap().mode_product(1, &b).unwrap();
        let ba = b.matmul(&a).unwrap();
        let direct = t.mode_product(1, &ba).unwrap();

        assert_eq!(chained.shape(), &[2, 5, 2]);
        assert_eq!(direct.shape(), &[2, 5, 2]);

        for i in 0..chained.len() {
            assert!((chained.data[i] - direct.data[i]).abs() < EPSILON);
        }
    }

    #[test]
    fn test_tensor_norm() {
        let t = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((t.norm() - (30.0_f64).sqrt()).abs() < EPSILON);
    }

    #[test]
    fn test_tensor_new_alias() {
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::new(vec![2, 3], raw.clone()).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.data(), raw.as_slice());
        assert_eq!(Tensor::new(vec![], vec![]), Err(DspError::EmptyInput));
        assert!(matches!(
            Tensor::new(vec![2, 0], vec![]),
            Err(DspError::InvalidParameter { name: "shape", .. })
        ));
    }
}
