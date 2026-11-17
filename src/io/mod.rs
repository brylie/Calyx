//! Pure-Rust audio and data persistence I/O module.
//!
//! Provides RIFF/WAVE audio export and `.calyx` binary container format encoding and decoding
//! built strictly from first principles with zero external crate dependencies.

pub mod container;
pub mod wav;

// Re-export core container types and functions for ergonomics
pub use container::{
    adler32, AxisDefinition, CalyxContainer, CalyxContainerHeader, CpContainerData, MatrixData,
    QualityTier, ResidualContainerData, TuckerContainerData, CALYX_MAGIC, CALYX_MAJOR_VERSION,
    CALYX_MINOR_VERSION, CHUNK_AXES, CHUNK_CPMD, CHUNK_META, CHUNK_NOIS, CHUNK_TUCK,
    FLAG_HAS_CP, FLAG_HAS_RESIDUAL, FLAG_HAS_TUCKER, HEADER_SIZE,
};

// Re-export core WAV types and functions for ergonomics
pub use wav::{
    export_signal_buffer, export_wav_file, export_wav_file_float32, f64_to_pcm16, pcm16_to_f64,
    read_wav_float32, read_wav_header, read_wav_pcm16, write_wav_float32, write_wav_pcm16,
    WavFormat, WavHeader, WavSpec,
};
