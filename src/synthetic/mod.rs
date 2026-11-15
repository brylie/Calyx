//! Pure Rust synthetic audio and spectral test signal generation without external audio dependencies.

pub mod buffer;
pub mod chirp;
pub mod harmonic;
pub mod impulse;
pub mod instrument;
pub mod noise;
pub mod sine;
pub mod traits;

// Re-exports
pub use buffer::{
    AnalyticalPartial, HarmonicPartial, KnownPartial, NoiseBandEnvelope, SignalBuffer, SignalKind,
    SignalMetadata, SpectralFrame,
};
pub use chirp::{ChirpGenerator, ChirpKind};
pub use harmonic::{HarmonicComplexGenerator, HarmonicRecipe};
pub use impulse::ImpulseGenerator;
pub use instrument::{Articulation, SyntheticInstrumentSource};
pub use noise::{NoiseColor, NoiseGenerator, Pcg32};
pub use sine::SineGenerator;
pub use traits::SignalGenerator;
