//! Sub-bin Spectral Peak Detection with Parabolic Interpolation.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library,
//! zero external dependencies, safe Rust (#![deny(unsafe_code)]).

use std::f64::consts::PI;
use crate::math::Complex;
use super::traits::SpectralPeak;

/// Spectral peak detector using 3-point parabolic interpolation on log-magnitude bins.
#[derive(Debug, Clone)]
pub struct PeakDetector {
    /// Magnitude floor threshold for candidate peaks (default: 1e-6).
    pub threshold: f64,
}

impl Default for PeakDetector {
    fn default() -> Self {
        Self { threshold: 1e-6 }
    }
}

impl PeakDetector {
    /// Construct a new peak detector with a specified threshold.
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Detect spectral peaks from a real-valued magnitude spectrum.
    ///
    /// Directly called by Calyx test harness.
    pub fn detect_peaks(spectrum: &[f64], sample_rate: f64) -> Vec<SpectralPeak> {
        Self::default().detect(spectrum, sample_rate)
    }

    /// Detect spectral peaks with sub-bin refinement and phase from a complex spectrum.
    pub fn detect_peaks_complex(spectrum: &[Complex<f64>], sample_rate: f64) -> Vec<SpectralPeak> {
        let mag: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
        let mut peaks = Self::default().detect(&mag, sample_rate);
        for peak in &mut peaks {
            if peak.bin_index < spectrum.len() {
                peak.phase_rad = spectrum[peak.bin_index].arg();
            }
        }
        peaks
    }

    /// Instance method detecting peaks from a magnitude spectrum.
    pub fn detect(&self, spectrum: &[f64], sample_rate: f64) -> Vec<SpectralPeak> {
        let n_bins = spectrum.len();
        if n_bins == 0 || sample_rate <= 0.0 {
            return Vec::new();
        }

        // FFT size N corresponding to one-sided spectrum: N = 2 * (n_bins - 1)
        let n_fft = if n_bins > 1 {
            2 * (n_bins - 1)
        } else {
            1
        };
        let df = sample_rate / (n_fft as f64);

        let mut peaks = Vec::new();
        const EPSILON: f64 = 1e-12;

        // 1. Check DC boundary bin (k = 0)
        if n_bins > 1
            && spectrum[0].is_finite()
            && spectrum[0] > self.threshold
            && spectrum[0] > spectrum[1]
        {
            peaks.push(SpectralPeak {
                bin_index: 0,
                frequency_hz: 0.0,
                amplitude: spectrum[0],
                phase_rad: 0.0,
            });
        }

        // 2. Interior bins (1 <= k < n_bins - 1)
        for k in 1..(n_bins.saturating_sub(1)) {
            let y_curr = spectrum[k];
            let y_prev = spectrum[k - 1];
            let y_next = spectrum[k + 1];

            if !y_curr.is_finite() || !y_prev.is_finite() || !y_next.is_finite() {
                continue;
            }

            // Local maximum condition (strict inequality on at least one side rejects flat plateaus)
            let is_local_max = y_curr > self.threshold
                && y_curr >= y_prev
                && y_curr >= y_next
                && (y_curr > y_prev || y_curr > y_next);

            if is_local_max {
                // Exact 2-bin spectral ratio interpolation for Hann window
                let (delta, amp) = if y_next > y_prev {
                    let r = y_next / (y_curr + EPSILON);
                    let d = ((2.0 * r - 1.0) / (r + 1.0)).clamp(0.0, 0.5);
                    let sinc_factor = if d > 1e-6 {
                        (PI * d * (1.0 - d * d)) / (PI * d).sin()
                    } else {
                        1.0
                    };
                    (d, y_curr * sinc_factor)
                } else if y_prev > y_next {
                    let r = y_prev / (y_curr + EPSILON);
                    let d = ((2.0 * r - 1.0) / (r + 1.0)).clamp(0.0, 0.5);
                    let sinc_factor = if d > 1e-6 {
                        (PI * d * (1.0 - d * d)) / (PI * d).sin()
                    } else {
                        1.0
                    };
                    (-d, y_curr * sinc_factor)
                } else {
                    (0.0, y_curr)
                };

                let freq = ((k as f64) + delta) * df;

                peaks.push(SpectralPeak {
                    bin_index: k,
                    frequency_hz: freq.max(0.0),
                    amplitude: amp,
                    phase_rad: 0.0,
                });
            }
        }

        // 3. Check Nyquist boundary bin (k = n_bins - 1)
        if n_bins > 1 {
            let last_idx = n_bins - 1;
            let y_last = spectrum[last_idx];
            let y_penultimate = spectrum[last_idx - 1];
            if y_last.is_finite()
                && y_penultimate.is_finite()
                && y_last > self.threshold
                && y_last > y_penultimate
            {
                peaks.push(SpectralPeak {
                    bin_index: last_idx,
                    frequency_hz: sample_rate / 2.0,
                    amplitude: y_last,
                    phase_rad: 0.0,
                });
            }
        }

        peaks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peak_detector_silence() {
        let spec = vec![0.0; 257];
        let peaks = PeakDetector::detect_peaks(&spec, 48000.0);
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_peak_detector_flat() {
        let spec = vec![1.0; 257];
        let peaks = PeakDetector::detect_peaks(&spec, 48000.0);
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_peak_detector_dc() {
        let mut spec = vec![0.0; 257];
        spec[0] = 10.0;
        let peaks = PeakDetector::detect_peaks(&spec, 48000.0);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].bin_index, 0);
        assert!((peaks[0].frequency_hz - 0.0).abs() < 1e-6);
        assert!((peaks[0].amplitude - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_peak_detector_nyquist() {
        let mut spec = vec![0.0; 257];
        spec[256] = 10.0;
        let peaks = PeakDetector::detect_peaks(&spec, 48000.0);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].bin_index, 256);
        assert!((peaks[0].frequency_hz - 24000.0).abs() < 1e-6);
        assert!((peaks[0].amplitude - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_peak_detector_equal_adjacent_peaks() {
        let mut spec = vec![0.0; 257];
        spec[50] = 10.0;
        spec[51] = 10.0;
        let peaks = PeakDetector::detect_peaks(&spec, 48000.0);
        assert!(!peaks.is_empty());
        for p in &peaks {
            assert!(p.frequency_hz.is_finite());
            assert!(p.amplitude.is_finite());
        }
    }
}
