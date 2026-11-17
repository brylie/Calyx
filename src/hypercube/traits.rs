//! Core traits and container abstractions for the Hypercube decomposition subsystem.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure safe Rust standard library.

use crate::error::DspResult;
use crate::math::{Matrix, Tensor};
use crate::spectral::{SpectralFrame, SpectralTrajectory};
use crate::hypercube::storage::CompressedHypercube;
use crate::hypercube::tucker::TuckerDecomposition;

/// Trait for multi-dimensional spectral containers supporting coordinate access.
pub trait SpectralHypercube: Send + Sync {
    /// Retrieve a reference to a discrete frame by coordinate indices.
    fn get_frame(&self, coords: &[usize]) -> DspResult<&SpectralFrame>;

    /// Continuous multi-linear interpolation across normalized coordinates [0.0, 1.0]^D.
    fn interpolate(&self, coords: &[f64]) -> DspResult<SpectralFrame>;

    /// Dimensions / shape of the hypercube.
    fn shape(&self) -> &[usize];
}

/// Common trait for spectral trajectory decomposers (PCA, NMF, Tucker).
pub trait HypercubeDecomposer: Send + Sync {
    /// Decomposes a spectral trajectory into a compressed hypercube representation.
    fn decompose(&self, trajectory: &SpectralTrajectory) -> DspResult<CompressedHypercube>;

    /// Reconstructs a spectral trajectory from a compressed hypercube.
    fn reconstruct(&self, compressed: &CompressedHypercube) -> DspResult<SpectralTrajectory>;

    /// Trajectory reconstruction alias matching project scope.
    fn reconstruct_trajectory(&self, compressed: &CompressedHypercube) -> DspResult<SpectralTrajectory> {
        self.reconstruct(compressed)
    }
}

/// Modular trait for 2D matrix dimensionality reduction engines (e.g. PCA, NMF).
pub trait DimensionalityReducer: Send + Sync {
    /// Reduces a matrix into (weights, basis).
    fn reduce(&self, matrix: &Matrix<f64>, rank: usize) -> DspResult<(Matrix<f64>, Matrix<f64>)>;
}

/// Modular trait for N-dimensional tensor decomposition engines (e.g. Tucker, CP).
pub trait TensorDecomposer: Send + Sync {
    /// Decomposes an N-dimensional tensor into core tensor and factor matrices.
    fn decompose_tensor(&self, tensor: &Tensor<f64>, ranks: &[usize]) -> DspResult<TuckerDecomposition>;
}
