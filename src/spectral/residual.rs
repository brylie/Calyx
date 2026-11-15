//! Stochastic residual noise extraction and critical filterbank analysis.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library,
//! zero external dependencies, safe Rust (#![deny(unsafe_code)]).

use crate::error::{DspError, DspResult};
use super::traits::{HarmonicPartial, NoiseBandEnvelope, ResidualAnalyzer, SpectralTrajectory};

/// Extractor for stochastic residual noise and perceptual critical-band envelopes.
#[derive(Debug, Clone, PartialEq)]
pub struct StochasticResidualExtractor {
    /// Number of perceptual critical bands.
    pub num_bands: usize,
}

impl StochasticResidualExtractor {
    /// Create a new stochastic residual extractor with requested number of bands.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] if `num_bands == 0`.
    pub fn new(num_bands: usize) -> DspResult<Self> {
        if num_bands == 0 {
            return Err(DspError::invalid_parameter(
                "num_bands",
                "Number of bands must be greater than zero",
            ));
        }
        Ok(Self { num_bands })
    }

    /// Associated function extracting residual envelope from matching original and deterministic spectra.
    ///
    /// Evaluates:
    /// $$|E\[i\]|^2 = \max(0.0, |S\[i\]|^2 - |D\[i\]|^2)$$
    /// Clamps negative values to zero and prevents division-by-zero on silence.
    pub fn extract_residual(orig_spectrum: &[f64], det_spectrum: &[f64]) -> NoiseBandEnvelope {
        let n = orig_spectrum.len().min(det_spectrum.len());
        if n == 0 {
            return NoiseBandEnvelope {
                band_energies: Vec::new(),
                noise_ratio: 0.0,
            };
        }

        let mut band_energies = Vec::with_capacity(n);
        let mut total_orig = 0.0;
        let mut total_res = 0.0;

        for i in 0..n {
            let orig = orig_spectrum[i];
            let det = det_spectrum[i];
            let orig_e = orig * orig;
            let det_e = det * det;
            let res_e = (orig_e - det_e).max(0.0);
            band_energies.push(res_e);
            total_orig += orig_e;
            total_res += res_e;
        }

        let noise_ratio = if total_orig > 1e-14 {
            (total_res / total_orig).min(1.0)
        } else {
            0.0
        };

        NoiseBandEnvelope {
            band_energies,
            noise_ratio,
        }
    }

    /// Extract critical band envelope from precomputed linear residual magnitude spectrum.
    ///
    /// Evaluates:
    /// $$|E[i]|^2 = |R[i]|^2$$
    /// directly from the linear time-domain residual spectrum without $2 S \Delta$ cross-term bleed.
    pub fn extract_from_residual(&self, orig_mag: &[f64], res_mag: &[f64]) -> NoiseBandEnvelope {
        if self.num_bands == 0 {
            return NoiseBandEnvelope {
                band_energies: Vec::new(),
                noise_ratio: 0.0,
            };
        }

        let n = orig_mag.len().min(res_mag.len());
        if n == 0 {
            return NoiseBandEnvelope {
                band_energies: vec![0.0; self.num_bands],
                noise_ratio: 0.0,
            };
        }

        let mut band_energies = vec![0.0; self.num_bands];
        let mut total_orig = 0.0;
        let mut total_res = 0.0;

        for (i, (&orig, &res)) in orig_mag.iter().take(n).zip(res_mag.iter().take(n)).enumerate() {
            let orig_e = orig * orig;
            let res_e = res * res;

            let band = if n == self.num_bands {
                i
            } else {
                ((i * self.num_bands) / n).min(self.num_bands - 1)
            };

            band_energies[band] += res_e;
            total_orig += orig_e;
            total_res += res_e;
        }

        let noise_ratio = if total_orig > 1e-14 {
            (total_res / total_orig).min(1.0)
        } else {
            0.0
        };

        NoiseBandEnvelope {
            band_energies,
            noise_ratio,
        }
    }

    /// Extract critical band envelope from original and deterministic magnitude spectra.
    pub fn extract(&self, orig_mag: &[f64], det_mag: &[f64]) -> NoiseBandEnvelope {
        if self.num_bands == 0 {
            return NoiseBandEnvelope {
                band_energies: Vec::new(),
                noise_ratio: 0.0,
            };
        }

        let n = orig_mag.len().min(det_mag.len());
        if n == 0 {
            return NoiseBandEnvelope {
                band_energies: vec![0.0; self.num_bands],
                noise_ratio: 0.0,
            };
        }

        let mut band_energies = vec![0.0; self.num_bands];
        let mut total_orig = 0.0;
        let mut total_res = 0.0;

        for (i, (&orig, &det)) in orig_mag.iter().take(n).zip(det_mag.iter().take(n)).enumerate() {
            let orig_e = orig * orig;
            let det_e = det * det;
            let res_e = (orig_e - det_e).max(0.0);

            let band = if n == self.num_bands {
                i
            } else {
                ((i * self.num_bands) / n).min(self.num_bands - 1)
            };

            band_energies[band] += res_e;
            total_orig += orig_e;
            total_res += res_e;
        }

        let noise_ratio = if total_orig > 1e-14 {
            (total_res / total_orig).min(1.0)
        } else {
            0.0
        };

        NoiseBandEnvelope {
            band_energies,
            noise_ratio,
        }
    }

    /// Extract averaged critical band energies across an entire spectral trajectory.
    ///
    /// Aggregates and averages the precomputed [`NoiseBandEnvelope`] band energies
    /// across all frames in `trajectory`. The `partials` parameter is retained for trait
    /// conformance with [`ResidualAnalyzer::extract_from_trajectory`].
    pub fn extract_from_trajectory(
        &self,
        trajectory: &SpectralTrajectory,
        partials: &[HarmonicPartial],
    ) -> Vec<f64> {
        let _ = partials;
        if trajectory.frames.is_empty() || self.num_bands == 0 {
            return vec![0.0; self.num_bands];
        }

        let mut avg_bands = vec![0.0; self.num_bands];
        let num_frames = trajectory.frames.len() as f64;

        for frame in &trajectory.frames {
            let frame_bands = &frame.residual.band_energies;
            let fb_len = frame_bands.len();

            if fb_len == self.num_bands {
                for (b, &e) in frame_bands.iter().enumerate() {
                    avg_bands[b] += e;
                }
            } else if fb_len > 0 {
                // Map/resample frame bands into target num_bands
                for (i, &e) in frame_bands.iter().enumerate() {
                    let b = ((i * self.num_bands) / fb_len).min(self.num_bands - 1);
                    avg_bands[b] += e;
                }
            }
        }

        for band in avg_bands.iter_mut().take(self.num_bands) {
            *band /= num_frames;
        }

        avg_bands
    }
}

impl ResidualAnalyzer for StochasticResidualExtractor {
    #[inline]
    fn extract_residual(&self, orig_mag: &[f64], det_mag: &[f64]) -> NoiseBandEnvelope {
        Self::extract_residual(orig_mag, det_mag)
    }

    #[inline]
    fn extract(&self, orig_mag: &[f64], det_mag: &[f64]) -> NoiseBandEnvelope {
        self.extract(orig_mag, det_mag)
    }

    #[inline]
    fn extract_from_trajectory(
        &self,
        trajectory: &SpectralTrajectory,
        partials: &[HarmonicPartial],
    ) -> Vec<f64> {
        self.extract_from_trajectory(trajectory, partials)
    }
}

impl ResidualAnalyzer for DspResult<StochasticResidualExtractor> {
    #[inline]
    fn extract_residual(&self, orig_mag: &[f64], det_mag: &[f64]) -> NoiseBandEnvelope {
        match self {
            Ok(extractor) => extractor.extract_residual(orig_mag, det_mag),
            Err(_) => NoiseBandEnvelope {
                band_energies: Vec::new(),
                noise_ratio: 0.0,
            },
        }
    }

    #[inline]
    fn extract(&self, orig_mag: &[f64], det_mag: &[f64]) -> NoiseBandEnvelope {
        match self {
            Ok(extractor) => extractor.extract(orig_mag, det_mag),
            Err(_) => NoiseBandEnvelope {
                band_energies: Vec::new(),
                noise_ratio: 0.0,
            },
        }
    }

    #[inline]
    fn extract_from_trajectory(
        &self,
        trajectory: &SpectralTrajectory,
        partials: &[HarmonicPartial],
    ) -> Vec<f64> {
        match self {
            Ok(extractor) => extractor.extract_from_trajectory(trajectory, partials),
            Err(_) => vec![0.0; 16],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual_zero_rejection() {
        assert!(matches!(
            StochasticResidualExtractor::new(0),
            Err(DspError::InvalidParameter { name: "num_bands", .. })
        ));
    }

    #[test]
    fn test_residual_perfect_deterministic_match() {
        let spec = vec![1.0; 16];
        let res = StochasticResidualExtractor::extract_residual(&spec, &spec);
        assert!(res.band_energies.iter().all(|&e| e.abs() < 1e-12));
        assert!(res.noise_ratio.abs() < 1e-12);
    }

    #[test]
    fn test_residual_overshoot_clamping() {
        let orig = vec![1.0; 16];
        let det = vec![1.5; 16];
        let res = StochasticResidualExtractor::extract_residual(&orig, &det);
        assert!(res.band_energies.iter().all(|&e| e >= 0.0));
    }

    #[test]
    fn test_residual_zero_energy_original() {
        let orig = vec![0.0; 16];
        let det = vec![0.0; 16];
        let res = StochasticResidualExtractor::extract_residual(&orig, &det);
        assert!(!res.noise_ratio.is_nan());
        assert_eq!(res.noise_ratio, 0.0);
    }

    #[test]
    fn test_residual_single_subband() {
        let extractor = StochasticResidualExtractor::new(1).unwrap();
        let orig = vec![2.0; 8];
        let det = vec![1.0; 8];
        let res = extractor.extract(&orig, &det);
        assert_eq!(res.band_energies.len(), 1);
        assert!(res.band_energies[0] > 0.0);
    }
}
