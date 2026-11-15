//! # Acoustic Resynthesis and Reconstruction Subsystem
//!
//! Provides additive sinusoidal oscillator banks with continuous phase tracking,
//! phase modeling (harmonic locking, vocoder integration, dispersion), stochastic noise
//! filterbank resynthesis, and Overlap-Add (OLA) streaming reconstruction with COLA normalization.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library safe Rust.

pub mod additive;
pub mod noise;
pub mod ola;
pub mod phase;
pub mod traits;

pub use additive::*;
pub use noise::*;
pub use ola::*;
pub use phase::*;
pub use traits::*;
