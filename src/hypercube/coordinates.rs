//! Hypercube layout, 5-axis addressing (Time, Pitch, Dynamic, Articulation, Variation),
//! and coordinate indexing mathematics.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library.

use crate::error::{DspError, DspResult};

/// 5-axis acoustic hypercube layout specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HypercubeLayout {
    axes: Vec<usize>,
    strides: Vec<usize>,
}

impl HypercubeLayout {
    /// Constructs a new hypercube layout with given axis dimensions.
    ///
    /// # Errors
    /// Returns `DspError::EmptyInput` if `axes.is_empty()`.
    /// Returns `DspError::InvalidParameter` if any axis has dimension 0.
    pub fn new(axes: Vec<usize>) -> DspResult<Self> {
        if axes.is_empty() {
            return Err(DspError::EmptyInput);
        }

        for (i, &dim) in axes.iter().enumerate() {
            if dim == 0 {
                return Err(DspError::invalid_parameter(
                    "axes",
                    format!("Axis {} has invalid dimension 0; must be >= 1", i),
                ));
            }
        }

        let d = axes.len();
        let mut strides = vec![0; d];
        let mut s = 1usize;

        for i in (0..d).rev() {
            if axes[i] == 1 {
                strides[i] = 0;
            } else {
                strides[i] = s;
                s = s.checked_mul(axes[i]).ok_or_else(|| {
                    DspError::invalid_parameter("axes", "Hypercube total size overflowed usize")
                })?;
            }
        }

        Ok(Self { axes, strides })
    }

    /// Returns the coordinate strides for linear index mapping.
    ///
    /// If an axis has dimension 1, its stride is defined as 0.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the shape / axis dimensions.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.axes
    }

    /// Returns the axes vector.
    #[inline]
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }

    /// Total number of cells in the hypercube.
    #[inline]
    pub fn total_cells(&self) -> usize {
        self.axes.iter().product()
    }

    /// Computes the flat linear buffer index for a discrete coordinate tuple.
    ///
    /// # Errors
    /// Returns `DspError::DimensionMismatch` if coordinate length does not match axis count.
    /// Returns `DspError::InvalidParameter` if any coordinate exceeds its axis bound.
    pub fn flat_index(&self, coords: &[usize]) -> DspResult<usize> {
        if coords.len() != self.axes.len() {
            return Err(DspError::dimension_mismatch(
                self.axes.clone(),
                coords.to_vec(),
            ));
        }

        let mut idx = 0usize;
        for (d, (&c, (&dim, &stride))) in coords
            .iter()
            .zip(self.axes.iter().zip(self.strides.iter()))
            .enumerate()
        {
            if c >= dim {
                return Err(DspError::invalid_parameter(
                    "coords",
                    format!(
                        "Coordinate {} at axis {} out of bounds for dimension {}",
                        c, d, dim
                    ),
                ));
            }
            idx += c * stride;
        }

        Ok(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypercube_layout_valid() {
        let layout = HypercubeLayout::new(vec![2, 3, 4]).unwrap();
        assert_eq!(layout.shape(), &[2, 3, 4]);
        assert_eq!(layout.strides(), &[12, 4, 1]);
        assert_eq!(layout.total_cells(), 24);

        assert_eq!(layout.flat_index(&[0, 0, 0]).unwrap(), 0);
        assert_eq!(layout.flat_index(&[1, 2, 3]).unwrap(), 1 * 12 + 2 * 4 + 3);
    }

    #[test]
    fn test_hypercube_single_cell_axis() {
        let layout = HypercubeLayout::new(vec![1, 4, 4, 4, 4]).unwrap();
        assert_eq!(layout.strides()[0], 0);
        assert_eq!(layout.strides()[4], 1);
        assert_eq!(layout.strides()[3], 4);
        assert_eq!(layout.strides()[2], 16);
        assert_eq!(layout.strides()[1], 64);
    }

    #[test]
    fn test_hypercube_empty_and_zero_rejection() {
        assert!(matches!(
            HypercubeLayout::new(vec![]),
            Err(DspError::EmptyInput)
        ));
        assert!(matches!(
            HypercubeLayout::new(vec![3, 0, 4]),
            Err(DspError::InvalidParameter { name: "axes", .. })
        ));
    }

    #[test]
    fn test_hypercube_out_of_bounds_indexing() {
        let layout = HypercubeLayout::new(vec![2, 3]).unwrap();
        assert!(layout.flat_index(&[2, 0]).is_err());
        assert!(layout.flat_index(&[0, 3]).is_err());
        assert!(layout.flat_index(&[0]).is_err());
    }
}
