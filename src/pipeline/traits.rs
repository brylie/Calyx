//! Composable DSP pipeline stage abstractions.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library safe Rust.

use crate::error::DspResult;

/// Generic DSP stage transforming an `Input` into an `Output`.
pub trait DspStage<Input, Output>: Send + Sync {
    /// Process input data and return transformed stage output.
    fn process(&mut self, input: Input) -> DspResult<Output>;

    /// Reset internal filter/buffer state if any.
    fn reset(&mut self) {}
}

/// Stateful streaming block processor for real-time and block-based DSP streams.
pub trait StreamProcessor: Send + Sync {
    /// Process a block of audio samples.
    fn process_block(&mut self, input: &[f64]) -> DspResult<Vec<f64>>;

    /// Flush any pending buffered audio.
    fn flush(&mut self) -> DspResult<Vec<f64>>;

    /// Reset the streaming processor state.
    fn reset(&mut self);
}
