//! # Hypercube Decomposition Subsystem
//!
//! Provides 5-axis coordinate representation $(t, p, d, a, v)$, dense and sparse storage,
//! multi-linear continuous parameter interpolation, and first-principles decomposition
//! algorithms: Principal Component Analysis (PCA), Lee-Seung Non-Negative Matrix Factorization (NMF),
//! Higher-Order SVD (Tucker HOSVD), and Canonical Polyadic Alternating Least Squares (CP-ALS).
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library, zero external linear algebra crates.

pub mod coordinates;
pub mod cp;
pub mod nmf;
pub mod pca;
pub mod storage;
pub mod traits;
pub mod tucker;

pub use coordinates::*;
pub use cp::*;
pub use nmf::*;
pub use pca::*;
pub use storage::*;
pub use traits::*;
pub use tucker::*;

// Re-export QualityTier and QualityTierConfig for downstream integration
pub use crate::metrics::report::{QualityTier, QualityTierConfig};
