//! # Calyx DSP
//!
//! An experimental framework in Rust for exploratory prototyping, iteration,
//! and quantitative evaluation of Calyx DSP algorithm designs and architecture.
//!
//! Built strictly from first principles using pure `std` with zero external dependencies.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod experiments;
pub mod hypercube;
pub mod math;
pub mod metrics;
pub mod pipeline;
pub mod spectral;
pub mod synthesis;
pub mod synthetic;

// Re-export core error types at the crate root for ergonomics
pub use error::{DspError, DspResult};
