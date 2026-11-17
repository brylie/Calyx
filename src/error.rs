//! Typed error system for the Calyx DSP framework.
//!
//! Provides [`DspError`] and [`DspResult<T>`] covering domain errors,
//! dimension mismatches, parameter validations, and numerical failures.

use std::fmt;

/// Domain error type representing all failure modes in the Calyx DSP framework.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DspError {
    /// An input buffer, slice, or sequence is unexpectedly empty.
    EmptyInput,

    /// A parameter value violates algorithm preconditions or physical bounds.
    InvalidParameter {
        /// Name of the invalid parameter.
        name: &'static str,
        /// Detailed reason why the parameter is invalid.
        reason: String,
    },

    /// A 1D buffer or slice length does not match the expected length.
    BufferLengthMismatch {
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements provided.
        actual: usize,
    },

    /// Multi-dimensional tensor or matrix shapes are incompatible.
    DimensionMismatch {
        /// Expected dimension or shape vector.
        expected: Vec<usize>,
        /// Actual dimension or shape vector.
        actual: Vec<usize>,
    },

    /// An algorithm requires a power-of-two size (e.g. Radix-2 FFT), but received a non-power-of-two.
    NonPowerOfTwo {
        /// The invalid non-power-of-two size.
        size: usize,
    },

    /// Numerical instability detected during computation (e.g. NaN, infinity, ill-conditioned matrix, division by zero).
    NumericalInstability(String),

    /// An iterative algorithm failed to converge within the allowed tolerance or iteration limit.
    NonConvergence(String),
}

impl fmt::Display for DspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DspError::EmptyInput => {
                write!(f, "Input buffer or sequence is empty")
            }
            DspError::InvalidParameter { name, reason } => {
                write!(f, "Invalid parameter '{}': {}", name, reason)
            }
            DspError::BufferLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "Buffer length mismatch: expected {} elements, got {}",
                    expected, actual
                )
            }
            DspError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected shape {:?}, got {:?}",
                    expected, actual
                )
            }
            DspError::NonPowerOfTwo { size } => {
                write!(
                    f,
                    "Size {} is not a power of two (Radix-2 FFT requirement)",
                    size
                )
            }
            DspError::NumericalInstability(msg) => {
                write!(f, "Numerical instability: {}", msg)
            }
            DspError::NonConvergence(msg) => {
                write!(f, "Algorithm non-convergence: {}", msg)
            }
        }
    }
}

impl std::error::Error for DspError {}

/// Convenience result type alias for Calyx DSP operations.
pub type DspResult<T> = Result<T, DspError>;

impl DspError {
    /// Construct an `InvalidParameter` error with static parameter name and stringifiable reason.
    #[inline]
    pub fn invalid_parameter(name: &'static str, reason: impl Into<String>) -> Self {
        DspError::InvalidParameter {
            name,
            reason: reason.into(),
        }
    }

    /// Construct a `BufferLengthMismatch` error.
    #[inline]
    pub fn buffer_length_mismatch(expected: usize, actual: usize) -> Self {
        DspError::BufferLengthMismatch { expected, actual }
    }

    /// Construct a `DimensionMismatch` error from any types convertible into `Vec<usize>`.
    #[inline]
    pub fn dimension_mismatch(
        expected: impl Into<Vec<usize>>,
        actual: impl Into<Vec<usize>>,
    ) -> Self {
        DspError::DimensionMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Construct a `NonPowerOfTwo` error.
    #[inline]
    pub fn non_power_of_two(size: usize) -> Self {
        DspError::NonPowerOfTwo { size }
    }

    /// Construct a `NumericalInstability` error.
    #[inline]
    pub fn numerical_instability(msg: impl Into<String>) -> Self {
        DspError::NumericalInstability(msg.into())
    }

    /// Construct a `NonConvergence` error.
    #[inline]
    pub fn non_convergence(msg: impl Into<String>) -> Self {
        DspError::NonConvergence(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_formatting() {
        assert_eq!(
            DspError::EmptyInput.to_string(),
            "Input buffer or sequence is empty"
        );

        let err_param = DspError::InvalidParameter {
            name: "sample_rate",
            reason: "must be positive".to_string(),
        };
        assert_eq!(
            err_param.to_string(),
            "Invalid parameter 'sample_rate': must be positive"
        );

        let err_buf = DspError::BufferLengthMismatch {
            expected: 1024,
            actual: 512,
        };
        assert_eq!(
            err_buf.to_string(),
            "Buffer length mismatch: expected 1024 elements, got 512"
        );

        let err_dim = DspError::DimensionMismatch {
            expected: vec![4, 4],
            actual: vec![4, 8],
        };
        assert_eq!(
            err_dim.to_string(),
            "Dimension mismatch: expected shape [4, 4], got [4, 8]"
        );

        let err_np2 = DspError::NonPowerOfTwo { size: 300 };
        assert_eq!(
            err_np2.to_string(),
            "Size 300 is not a power of two (Radix-2 FFT requirement)"
        );

        let err_num = DspError::NumericalInstability("NaN detected in output".to_string());
        assert_eq!(
            err_num.to_string(),
            "Numerical instability: NaN detected in output"
        );

        let err_conv = DspError::NonConvergence("ALS exceeded 1000 iterations".to_string());
        assert_eq!(
            err_conv.to_string(),
            "Algorithm non-convergence: ALS exceeded 1000 iterations"
        );
    }

    #[test]
    fn test_helper_constructors() {
        assert_eq!(
            DspError::invalid_parameter("hop_size", "cannot be 0"),
            DspError::InvalidParameter {
                name: "hop_size",
                reason: "cannot be 0".to_string(),
            }
        );

        assert_eq!(
            DspError::buffer_length_mismatch(16, 8),
            DspError::BufferLengthMismatch {
                expected: 16,
                actual: 8,
            }
        );

        assert_eq!(
            DspError::dimension_mismatch(vec![2, 3], vec![3, 2]),
            DspError::DimensionMismatch {
                expected: vec![2, 3],
                actual: vec![3, 2],
            }
        );

        assert_eq!(
            DspError::non_power_of_two(100),
            DspError::NonPowerOfTwo { size: 100 }
        );

        assert_eq!(
            DspError::numerical_instability("zero divisor"),
            DspError::NumericalInstability("zero divisor".to_string())
        );

        assert_eq!(
            DspError::non_convergence("tolerance not reached"),
            DspError::NonConvergence("tolerance not reached".to_string())
        );
    }

    #[test]
    fn test_error_equality_and_cloning() {
        let err1 = DspError::BufferLengthMismatch {
            expected: 10,
            actual: 20,
        };
        let err2 = err1.clone();
        assert_eq!(err1, err2);

        let err3 = DspError::BufferLengthMismatch {
            expected: 10,
            actual: 21,
        };
        assert_ne!(err1, err3);
        assert_ne!(err1, DspError::EmptyInput);
    }

    #[test]
    fn test_std_error_trait_and_source() {
        let err: Box<dyn std::error::Error> =
            Box::new(DspError::numerical_instability("overflow"));
        assert!(err.source().is_none());
        assert_eq!(err.to_string(), "Numerical instability: overflow");
    }

    #[test]
    fn test_thread_safety_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DspError>();
    }

    #[test]
    fn test_edge_cases() {
        // Power of two with size 0
        let zero_np2 = DspError::non_power_of_two(0);
        assert_eq!(
            zero_np2.to_string(),
            "Size 0 is not a power of two (Radix-2 FFT requirement)"
        );

        // Empty dimension vectors
        let empty_dim = DspError::dimension_mismatch(vec![], vec![1]);
        assert_eq!(
            empty_dim.to_string(),
            "Dimension mismatch: expected shape [], got [1]"
        );

        // Empty reason string
        let empty_reason = DspError::invalid_parameter("foo", "");
        assert_eq!(empty_reason.to_string(), "Invalid parameter 'foo': ");
    }
}
