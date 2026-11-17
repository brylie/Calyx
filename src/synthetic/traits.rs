//! Core traits for synthetic audio and spectral test signal generation.

use crate::error::{DspError, DspResult};
use crate::synthetic::buffer::SignalBuffer;

/// Core trait for deterministic synthetic audio signal generators.
pub trait SignalGenerator {
    /// Generates audio samples into a freshly allocated [`SignalBuffer`].
    ///
    /// If `num_samples == 0`, returns an empty buffer with `sample_rate` preserved.
    fn generate(&self, num_samples: usize, sample_rate: f64) -> SignalBuffer;

    /// Generates audio samples directly into an existing pre-allocated destination slice.
    ///
    /// This method performs zero heap allocations and is suitable for hot loops and real-time DSP.
    fn generate_into(&self, buffer: &mut [f64], sample_rate: f64);

    /// Generates an audio buffer with strict validation of parameters and sampling rate.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] if `sample_rate <= 0.0` or non-finite.
    fn generate_checked(&self, num_samples: usize, sample_rate: f64) -> DspResult<SignalBuffer> {
        if sample_rate <= 0.0 || !sample_rate.is_finite() {
            return Err(DspError::invalid_parameter(
                "sample_rate",
                format!("Sample rate must be positive and finite, got {}", sample_rate),
            ));
        }
        Ok(self.generate(num_samples, sample_rate))
    }
}
