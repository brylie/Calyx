//! First-principles dense row-major matrix mathematics.
//!
//! Complies strictly with Demo Integrity Mode: zero external dependencies,
//! 100% pure standard library, safe Rust.

use crate::error::{DspError, DspResult};

/// A dense 2D matrix stored in row-major (C-contiguous) order.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T = f64> {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Flat row-major contiguous data elements
    pub data: Vec<T>,
}

impl Matrix<f64> {
    /// Creates a new matrix filled with zeros.
    ///
    /// # Errors
    /// Returns `DspError::InvalidParameter` if `rows == 0` or `cols == 0`,
    /// or if the total element count overflows `usize`.
    pub fn zeros(rows: usize, cols: usize) -> DspResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(DspError::invalid_parameter(
                "dimensions",
                "Matrix dimensions must be non-zero positive integers",
            ));
        }
        let total = rows.checked_mul(cols).ok_or_else(|| {
            DspError::invalid_parameter("dimensions", "Matrix size overflowed usize")
        })?;
        Ok(Self {
            rows,
            cols,
            data: vec![0.0; total],
        })
    }

    /// Creates a matrix from a flat vector in row-major order.
    ///
    /// # Errors
    /// Returns `DspError::EmptyInput` if dimensions are zero and data is empty.
    /// Returns `DspError::InvalidParameter` if rows or cols is zero.
    /// Returns `DspError::BufferLengthMismatch` if `data.len() != rows * cols`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> DspResult<Self> {
        if rows == 0 || cols == 0 {
            if data.is_empty() {
                return Err(DspError::EmptyInput);
            }
            return Err(DspError::invalid_parameter(
                "dimensions",
                "Matrix dimensions must be non-zero positive integers",
            ));
        }
        let expected = rows.checked_mul(cols).ok_or_else(|| {
            DspError::invalid_parameter("dimensions", "Matrix size overflowed usize")
        })?;
        if data.len() != expected {
            return Err(DspError::buffer_length_mismatch(expected, data.len()));
        }
        Ok(Self { rows, cols, data })
    }

    /// Creates a matrix from a flat vector in row-major order.
    ///
    /// Symmetrical constructor alias to [`from_vec`](Self::from_vec).
    #[inline]
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> DspResult<Self> {
        Self::from_vec(rows, cols, data)
    }

    /// Creates a matrix from a borrowed slice in row-major order.
    pub fn from_slice(rows: usize, cols: usize, slice: &[f64]) -> DspResult<Self> {
        Self::from_vec(rows, cols, slice.to_vec())
    }

    /// Creates an identity matrix of size $N \times N$.
    pub fn identity(n: usize) -> DspResult<Self> {
        let mut mat = Self::zeros(n, n)?;
        for i in 0..n {
            mat.data[i * n + i] = 1.0;
        }
        Ok(mat)
    }

    /// Constructs a matrix using a generator closure $f(\text{row}, \text{col})$.
    pub fn from_shape_fn<F>(rows: usize, cols: usize, mut f: F) -> DspResult<Self>
    where
        F: FnMut(usize, usize) -> f64,
    {
        let mut mat = Self::zeros(rows, cols)?;
        for r in 0..rows {
            for c in 0..cols {
                mat.data[r * cols + c] = f(r, c);
            }
        }
        Ok(mat)
    }

    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Total number of elements ($rows \times cols$).
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if matrix contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a borrowed slice of the flat underlying data.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Returns a borrowed slice of the flat underlying data.
    ///
    /// Symmetrical alias to [`as_slice`](Self::as_slice) matching `Tensor::data`.
    #[inline]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Returns a mutable slice of the flat underlying data.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Checked immutable element access at row `r` and column `c`.
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> DspResult<f64> {
        if r >= self.rows || c >= self.cols {
            return Err(DspError::dimension_mismatch(
                vec![self.rows, self.cols],
                vec![r, c],
            ));
        }
        Ok(self.data[r * self.cols + c])
    }

    /// Checked mutable element access at row `r` and column `c`.
    #[inline]
    pub fn get_mut(&mut self, r: usize, c: usize) -> DspResult<&mut f64> {
        if r >= self.rows || c >= self.cols {
            return Err(DspError::dimension_mismatch(
                vec![self.rows, self.cols],
                vec![r, c],
            ));
        }
        let idx = r * self.cols + c;
        Ok(&mut self.data[idx])
    }

    /// Checked element mutation at row `r` and column `c`.
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f64) -> DspResult<()> {
        let elem = self.get_mut(r, c)?;
        *elem = val;
        Ok(())
    }

    /// Zero-copy immutable row slice extraction.
    pub fn row(&self, r: usize) -> DspResult<&[f64]> {
        if r >= self.rows {
            return Err(DspError::invalid_parameter(
                "row",
                format!("Row index {} out of bounds for matrix with {} rows", r, self.rows),
            ));
        }
        let start = r * self.cols;
        Ok(&self.data[start..start + self.cols])
    }

    /// Zero-copy mutable row slice extraction.
    pub fn row_mut(&mut self, r: usize) -> DspResult<&mut [f64]> {
        if r >= self.rows {
            return Err(DspError::invalid_parameter(
                "row",
                format!("Row index {} out of bounds for matrix with {} rows", r, self.rows),
            ));
        }
        let start = r * self.cols;
        let end = start + self.cols;
        Ok(&mut self.data[start..end])
    }

    /// Owned column vector extraction.
    pub fn col(&self, c: usize) -> DspResult<Vec<f64>> {
        if c >= self.cols {
            return Err(DspError::invalid_parameter(
                "col",
                format!("Col index {} out of bounds for matrix with {} cols", c, self.cols),
            ));
        }
        let mut col_data = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            col_data.push(self.data[r * self.cols + c]);
        }
        Ok(col_data)
    }

    /// Extracts a submatrix block `[row_start..row_end) x [col_start..col_end)`.
    pub fn submatrix(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> DspResult<Self> {
        if row_start >= row_end
            || row_end > self.rows
            || col_start >= col_end
            || col_end > self.cols
        {
            return Err(DspError::invalid_parameter(
                "submatrix_bounds",
                "Invalid submatrix boundary ranges",
            ));
        }
        let sub_rows = row_end - row_start;
        let sub_cols = col_end - col_start;
        let mut sub_data = Vec::with_capacity(sub_rows * sub_cols);
        for r in row_start..row_end {
            let row_slice = &self.data[r * self.cols + col_start..r * self.cols + col_end];
            sub_data.extend_from_slice(row_slice);
        }
        Self::from_vec(sub_rows, sub_cols, sub_data)
    }

    /// Transposes the matrix into a new $M \times N \to N \times M$ matrix.
    pub fn transpose(&self) -> Self {
        let mut trans_data = vec![0.0; self.data.len()];
        for r in 0..self.rows {
            for c in 0..self.cols {
                trans_data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            data: trans_data,
        }
    }

    /// Matrix multiplication: `self (M x K) * other (K x N) -> (M x N)`.
    ///
    /// Uses cache-friendly IKJ loop order for contiguous memory streaming
    /// and SIMD auto-vectorization without third-party BLAS.
    pub fn matmul(&self, other: &Self) -> DspResult<Self> {
        if self.cols != other.rows {
            return Err(DspError::dimension_mismatch(
                vec![self.cols],
                vec![other.rows],
            ));
        }
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        let total = m.checked_mul(n).ok_or_else(|| {
            DspError::invalid_parameter("dimensions", "Matrix size overflowed usize")
        })?;
        let mut out = vec![0.0; total];
        for i in 0..m {
            let a_row_offset = i * k;
            let c_row_offset = i * n;
            for p in 0..k {
                let a_val = self.data[a_row_offset + p];
                let b_row_offset = p * n;
                for j in 0..n {
                    out[c_row_offset + j] += a_val * other.data[b_row_offset + j];
                }
            }
        }
        Self::from_vec(m, n, out)
    }

    /// Elementwise matrix addition ($A + B$).
    pub fn add(&self, other: &Self) -> DspResult<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(DspError::dimension_mismatch(
                vec![self.rows, self.cols],
                vec![other.rows, other.cols],
            ));
        }
        let sum_data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| a + b)
            .collect();
        Self::from_vec(self.rows, self.cols, sum_data)
    }

    /// Elementwise matrix subtraction ($A - B$).
    pub fn sub(&self, other: &Self) -> DspResult<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(DspError::dimension_mismatch(
                vec![self.rows, self.cols],
                vec![other.rows, other.cols],
            ));
        }
        let diff_data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| a - b)
            .collect();
        Self::from_vec(self.rows, self.cols, diff_data)
    }

    /// Scalar multiplication ($A \cdot \alpha$).
    pub fn scale(&self, factor: f64) -> Self {
        let scaled_data: Vec<f64> = self.data.iter().map(|&x| x * factor).collect();
        Self {
            rows: self.rows,
            cols: self.cols,
            data: scaled_data,
        }
    }

    /// Elementwise Hadamard product ($A \odot B$), widely used in NMF multiplicative updates.
    pub fn hadamard(&self, other: &Self) -> DspResult<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(DspError::dimension_mismatch(
                vec![self.rows, self.cols],
                vec![other.rows, other.cols],
            ));
        }
        let prod_data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| a * b)
            .collect();
        Self::from_vec(self.rows, self.cols, prod_data)
    }

    /// Frobenius norm: $\sqrt{\sum_{i,j} A_{i,j}^2}$.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

impl std::ops::Index<(usize, usize)> for Matrix<f64> {
    type Output = f64;
    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (r, c) = index;
        assert!(
            r < self.rows && c < self.cols,
            "Matrix index out of bounds: ({}, {}) for size ({}, {})",
            r,
            c,
            self.rows,
            self.cols
        );
        &self.data[r * self.cols + c]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Matrix<f64> {
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (r, c) = index;
        assert!(
            r < self.rows && c < self.cols,
            "Matrix index out of bounds: ({}, {}) for size ({}, {})",
            r,
            c,
            self.rows,
            self.cols
        );
        &mut self.data[r * self.cols + c]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-12;

    #[test]
    fn test_matrix_creation_and_indexing() {
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let mut mat = Matrix::from_vec(3, 4, data).unwrap();
        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 4);
        assert_eq!(mat.len(), 12);
        assert!(!mat.is_empty());

        for r in 0..3 {
            for c in 0..4 {
                assert_eq!(mat.get(r, c).unwrap(), (r * 4 + c) as f64);
                assert_eq!(mat[(r, c)], (r * 4 + c) as f64);
            }
        }

        mat.set(1, 2, 99.0).unwrap();
        assert_eq!(mat.get(1, 2).unwrap(), 99.0);
        assert_eq!(mat[(1, 2)], 99.0);

        mat[(2, 3)] = 123.0;
        assert_eq!(mat.get(2, 3).unwrap(), 123.0);
    }

    #[test]
    fn test_matrix_bounds_and_error_handling() {
        assert_eq!(
            Matrix::zeros(0, 5),
            Err(DspError::invalid_parameter(
                "dimensions",
                "Matrix dimensions must be non-zero positive integers"
            ))
        );
        assert_eq!(
            Matrix::zeros(5, 0),
            Err(DspError::invalid_parameter(
                "dimensions",
                "Matrix dimensions must be non-zero positive integers"
            ))
        );

        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(
            Matrix::from_vec(2, 2, data),
            Err(DspError::buffer_length_mismatch(4, 3))
        );

        let mat = Matrix::zeros(3, 4).unwrap();
        assert!(mat.get(3, 0).is_err());
        assert!(mat.get(0, 4).is_err());
        assert!(mat.row(3).is_err());
        assert!(mat.col(4).is_err());
    }

    #[test]
    fn test_matrix_row_col_extraction() {
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let mut mat = Matrix::from_vec(3, 4, data).unwrap();

        assert_eq!(mat.row(1).unwrap(), &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(mat.col(2).unwrap(), vec![2.0, 6.0, 10.0]);

        mat.row_mut(0).unwrap()[1] = 42.0;
        assert_eq!(mat.get(0, 1).unwrap(), 42.0);

        let sub = mat.submatrix(0, 2, 1, 3).unwrap();
        assert_eq!(sub.rows(), 2);
        assert_eq!(sub.cols(), 2);
        assert_eq!(sub.row(0).unwrap(), &[42.0, 2.0]);
        assert_eq!(sub.row(1).unwrap(), &[5.0, 6.0]);
    }

    #[test]
    fn test_matrix_transpose() {
        let data: Vec<f64> = (0..6).map(|x| x as f64).collect();
        let mat = Matrix::from_vec(2, 3, data).unwrap();
        let t = mat.transpose();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(t.row(0).unwrap(), &[0.0, 3.0]);
        assert_eq!(t.row(1).unwrap(), &[1.0, 4.0]);
        assert_eq!(t.row(2).unwrap(), &[2.0, 5.0]);

        let roundtrip = t.transpose();
        assert_eq!(roundtrip, mat);
    }

    #[test]
    fn test_matrix_matmul_correctness() {
        // A: 2x3, B: 3x2
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        // B = [[7, 8],
        //      [9, 1],
        //      [2, 3]]
        // C = A * B:
        // C[0, 0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
        // C[0, 1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
        // C[1, 0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
        // C[1, 1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]).unwrap();
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);
        assert_eq!(c.row(0).unwrap(), &[31.0, 19.0]);
        assert_eq!(c.row(1).unwrap(), &[85.0, 55.0]);

        // Identity property: A * I = A
        let eye = Matrix::identity(3).unwrap();
        let a_eye = a.matmul(&eye).unwrap();
        assert_eq!(a_eye, a);

        // Dimension mismatch
        let bad_b = Matrix::from_vec(4, 2, vec![0.0; 8]).unwrap();
        assert!(a.matmul(&bad_b).is_err());
    }

    #[test]
    fn test_matrix_hadamard_add_sub_scale_and_norm() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        let sum = a.add(&b).unwrap();
        assert_eq!(sum.data, vec![6.0, 8.0, 10.0, 12.0]);

        let diff = b.sub(&a).unwrap();
        assert_eq!(diff.data, vec![4.0, 4.0, 4.0, 4.0]);

        let scaled = a.scale(2.5);
        assert_eq!(scaled.data, vec![2.5, 5.0, 7.5, 10.0]);

        let hadamard = a.hadamard(&b).unwrap();
        assert_eq!(hadamard.data, vec![5.0, 12.0, 21.0, 32.0]);

        // frobenius norm: sqrt(1 + 4 + 9 + 16) = sqrt(30)
        let norm = a.frobenius_norm();
        assert!((norm - (30.0_f64).sqrt()).abs() < EPSILON);
    }

    #[test]
    fn test_matrix_new_alias_and_data_accessor() {
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = Matrix::new(2, 3, raw.clone()).unwrap();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.data(), raw.as_slice());
        assert_eq!(m.as_slice(), raw.as_slice());

        // Error handling through new
        assert_eq!(Matrix::new(0, 5, vec![]), Err(DspError::EmptyInput));
        assert_eq!(Matrix::new(5, 0, vec![]), Err(DspError::EmptyInput));
        assert!(matches!(
            Matrix::new(2, 3, vec![1.0, 2.0]),
            Err(DspError::BufferLengthMismatch { .. })
        ));
    }
}
