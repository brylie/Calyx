//! # Modular DSP Pipeline Subsystem
//!
//! Provides composable end-to-end DSP pipelines chaining spectral analysis,
//! parametric hypercube decomposition, and acoustic resynthesis across swappable backends
//! and quality tiers (`Reference`, `Production`, `Preview`).
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library safe Rust.

pub mod orchestrator;
pub mod traits;

pub use orchestrator::*;
pub use traits::*;

// Re-export QualityTierConfig and QualityTier for convenience
pub use crate::metrics::report::{QualityTier, QualityTierConfig};
