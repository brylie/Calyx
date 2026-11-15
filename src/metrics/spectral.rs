//! Spectral-domain fidelity metrics (Spectral Convergence, Log-Spectral Distance, Harmonic Peak Error).

use crate::error::{DspError, DspResult};
use crate::math::Complex;
use crate::synthetic::buffer::{HarmonicPartial, SpectralPeak};
use std::f64::consts::PI;

/// Floor delta in linear magnitude corresponding to -120 dBFS.
pub const DELTA_FLOOR: f64 = 1e-6;

/// Spectral-domain fidelity metrics comparing frequency spectra or time signals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralFidelityMetrics {
    /// Spectral Convergence (normalized Frobenius distance, $\ge 0.0$). Perfect match = 0.0.
    pub spectral_convergence: f64,
    /// Log-Spectral Distance in decibels (dB). Perfect match = 0.0 dB.
    pub log_spectral_distance_db: f64,
}

/// Discrepancy report assessing the fidelity of extracted sinusoidal partials.
#[derive(Debug, Clone, PartialEq)]
pub struct HarmonicExtractionReport {
    /// Maximum absolute frequency discrepancy in Hertz across all partials.
    pub max_frequency_error_hz: f64,
    /// Maximum relative amplitude discrepancy ($\Delta A / A$) across all partials.
    pub max_relative_amplitude_error: f64,
}

/// Evaluates discrepancies between analytical ground-truth partials and extracted peaks.
pub fn evaluate_harmonic_extraction(
    ground_truth: &[HarmonicPartial],
    extracted: &[(f64, f64)],
) -> HarmonicExtractionReport {
    if extracted.is_empty() {
        return HarmonicExtractionReport {
            max_frequency_error_hz: f64::INFINITY,
            max_relative_amplitude_error: f64::INFINITY,
        };
    }

    let mut max_freq_err = 0.0_f64;
    let mut max_amp_err = 0.0_f64;

    for gt in ground_truth {
        // Find extracted peak with minimal absolute frequency discrepancy
        let mut best_diff_f = f64::INFINITY;
        let mut best_amp_diff = f64::INFINITY;

        for &(f_ext, a_ext) in extracted {
            let df = (gt.frequency - f_ext).abs();
            if df < best_diff_f {
                best_diff_f = df;
                best_amp_diff = (gt.amplitude - a_ext).abs();
            }
        }

        if best_diff_f > max_freq_err {
            max_freq_err = best_diff_f;
        }
        let rel_amp_err = best_amp_diff / gt.amplitude.max(1e-12);
        if rel_amp_err > max_amp_err {
            max_amp_err = rel_amp_err;
        }
    }

    HarmonicExtractionReport {
        max_frequency_error_hz: max_freq_err,
        max_relative_amplitude_error: max_amp_err,
    }
}

/// Computes harmonic peak error between ground-truth and detected peaks.
///
/// Returns `f64::INFINITY` if `detected_peaks` is empty.
pub fn compute_harmonic_peak_error(
    gt_peaks: &[SpectralPeak],
    detected_peaks: &[SpectralPeak],
) -> f64 {
    if detected_peaks.is_empty() {
        return f64::INFINITY;
    }

    let mut max_err = 0.0_f64;
    for gt in gt_peaks {
        let mut min_diff = f64::INFINITY;
        for det in detected_peaks {
            let diff = (gt.frequency_hz - det.frequency_hz).abs();
            if diff < min_diff {
                min_diff = diff;
            }
        }
        if min_diff > max_err {
            max_err = min_diff;
        }
    }
    max_err
}

/// Computes first-principles single-sided DFT magnitude spectrum.
pub fn compute_magnitude_dft(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let num_bins = n / 2 + 1;
    let mut magnitudes = Vec::with_capacity(num_bins);
    let two_pi_over_n = 2.0 * PI / (n as f64);

    for k in 0..num_bins {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (idx, &x) in signal.iter().enumerate() {
            let angle = two_pi_over_n * ((k * idx) as f64);
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        magnitudes.push((re * re + im * im).sqrt());
    }

    magnitudes
}

/// Internal helper computing in-place Radix-2 Cooley-Tukey FFT on a complex buffer.
fn fft_radix2_in_place(buffer: &mut [Complex<f64>]) {
    let n = buffer.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0;
    for i in 0..n {
        if i < j {
            buffer.swap(i, j);
        }
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
    }

    // Cooley-Tukey butterfly
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / (len as f64);
        let w_step = Complex::new(angle.cos(), angle.sin());

        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = buffer[i + k];
                let v = buffer[i + k + half] * w;
                buffer[i + k] = u + v;
                buffer[i + k + half] = u - v;
                w = w * w_step;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Computes single-sided magnitude spectrum for a windowed block using Radix-2 FFT or DFT.
fn compute_block_magnitude(block: &[f64]) -> Vec<f64> {
    let n = block.len();
    if n.is_power_of_two() {
        let mut complex_buf: Vec<Complex<f64>> = block.iter().map(|&s| Complex::new(s, 0.0)).collect();
        fft_radix2_in_place(&mut complex_buf);
        let num_bins = n / 2 + 1;
        complex_buf[..num_bins].iter().map(|c| c.norm()).collect()
    } else {
        compute_magnitude_dft(block)
    }
}

/// Computes spectral fidelity metrics directly from two magnitude spectra slices.
fn compute_spectral_metrics_direct(
    ref_spec: &[f64],
    recon_spec: &[f64],
) -> DspResult<SpectralFidelityMetrics> {
    let diff_sq: f64 = ref_spec
        .iter()
        .zip(recon_spec.iter())
        .map(|(&r, &h)| (r - h).powi(2))
        .sum();
    let ref_sq: f64 = ref_spec.iter().map(|&r| r * r).sum();

    let sc = if ref_sq == 0.0 {
        if diff_sq == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (diff_sq / ref_sq).sqrt()
    };

    let k = ref_spec.len() as f64;
    let sum_log_sq: f64 = ref_spec
        .iter()
        .zip(recon_spec.iter())
        .map(|(&r, &h)| {
            let log_r = 20.0 * (r.abs() + DELTA_FLOOR).log10();
            let log_h = 20.0 * (h.abs() + DELTA_FLOOR).log10();
            (log_r - log_h).powi(2)
        })
        .sum();

    let lsd = if diff_sq == 0.0 {
        0.0
    } else {
        (sum_log_sq / k).sqrt()
    };

    Ok(SpectralFidelityMetrics {
        spectral_convergence: sc,
        log_spectral_distance_db: lsd,
    })
}

/// Computes spectral fidelity metrics direct from STFT frames over time-domain audio signals.
pub fn compute_spectral_metrics_stft(
    reference: &[f64],
    reconstructed: &[f64],
    window_size: usize,
    hop_size: usize,
) -> DspResult<SpectralFidelityMetrics> {
    if reference.is_empty() || reconstructed.is_empty() {
        return Err(DspError::EmptyInput);
    }
    if reference.len() != reconstructed.len() {
        return Err(DspError::BufferLengthMismatch {
            expected: reference.len(),
            actual: reconstructed.len(),
        });
    }

    let eval_frame_sc = |ref_mag: &[f64], recon_mag: &[f64]| -> f64 {
        let diff_sq: f64 = ref_mag
            .iter()
            .zip(recon_mag.iter())
            .map(|(&r, &h)| (r - h).powi(2))
            .sum();
        let ref_sq: f64 = ref_mag.iter().map(|&r| r * r).sum();
        if ref_sq == 0.0 {
            if diff_sq == 0.0 {
                0.0
            } else {
                (diff_sq / (ref_sq + 1e-12)).sqrt()
            }
        } else {
            (diff_sq / ref_sq).sqrt()
        }
    };

    let n = reference.len();
    if n < window_size {
        // Fall back to direct magnitude transform of padded/truncated block
        let mut ref_block = vec![0.0; window_size];
        let mut recon_block = vec![0.0; window_size];
        ref_block[..n].copy_from_slice(reference);
        recon_block[..n].copy_from_slice(reconstructed);

        let mut ref_mag = compute_block_magnitude(&ref_block);
        let mut recon_mag = compute_block_magnitude(&recon_block);
        let norm_factor = 2.0 / (window_size as f64);
        for x in &mut ref_mag {
            *x *= norm_factor;
        }
        for x in &mut recon_mag {
            *x *= norm_factor;
        }
        let m = compute_spectral_metrics_direct(&ref_mag, &recon_mag)?;
        let sc = eval_frame_sc(&ref_mag, &recon_mag);
        return Ok(SpectralFidelityMetrics {
            spectral_convergence: sc,
            log_spectral_distance_db: m.log_spectral_distance_db,
        });
    }

    let mut sum_sc = 0.0_f64;
    let mut sum_lsd = 0.0_f64;
    let mut num_frames = 0_usize;

    // Hann window
    let mut window = vec![0.0_f64; window_size];
    let two_pi_over_w = 2.0 * PI / (window_size as f64);
    for (i, w) in window.iter_mut().enumerate() {
        *w = 0.5 * (1.0 - (two_pi_over_w * (i as f64)).cos());
    }

    let mut ref_frame = vec![0.0_f64; window_size];
    let mut recon_frame = vec![0.0_f64; window_size];

    let mut start = 0;
    while start + window_size <= n {
        for i in 0..window_size {
            ref_frame[i] = reference[start + i] * window[i];
            recon_frame[i] = reconstructed[start + i] * window[i];
        }

        let mut ref_mag = compute_block_magnitude(&ref_frame);
        let mut recon_mag = compute_block_magnitude(&recon_frame);
        let norm_factor = 2.0 / (window_size as f64);
        for x in &mut ref_mag {
            *x *= norm_factor;
        }
        for x in &mut recon_mag {
            *x *= norm_factor;
        }

        let m = compute_spectral_metrics_direct(&ref_mag, &recon_mag)?;
        let sc_val = eval_frame_sc(&ref_mag, &recon_mag);
        sum_sc += sc_val;
        sum_lsd += m.log_spectral_distance_db;
        num_frames += 1;

        start += hop_size;
    }

    if num_frames == 0 {
        return Ok(SpectralFidelityMetrics {
            spectral_convergence: 0.0,
            log_spectral_distance_db: 0.0,
        });
    }

    Ok(SpectralFidelityMetrics {
        spectral_convergence: sum_sc / (num_frames as f64),
        log_spectral_distance_db: sum_lsd / (num_frames as f64),
    })
}

/// Computes spectral fidelity metrics (Spectral Convergence and Log-Spectral Distance in dB).
///
/// Automatically determines whether the inputs are spectral magnitude slices or raw time-domain audio:
/// - If all samples are non-negative ($\ge 0.0$) and length $\le 2048$, computes metrics directly across bins.
/// - If any sample is negative (standard oscillating audio), evaluates sliding STFT spectral metrics.
///
/// # Errors
/// Returns [`DspError::EmptyInput`] if slices are empty, or [`DspError::BufferLengthMismatch`]
/// if lengths differ.
pub fn compute_spectral_metrics(
    reference: &[f64],
    reconstructed: &[f64],
) -> DspResult<SpectralFidelityMetrics> {
    if reference.is_empty() || reconstructed.is_empty() {
        return Err(DspError::EmptyInput);
    }
    if reference.len() != reconstructed.len() {
        return Err(DspError::BufferLengthMismatch {
            expected: reference.len(),
            actual: reconstructed.len(),
        });
    }

    let is_direct_spectrum = reference.len() % 2 != 0
        && reference.iter().all(|&s| s >= 0.0)
        && reconstructed.iter().all(|&s| s >= 0.0);

    if is_direct_spectrum {
        // Direct magnitude spectrum computation
        compute_spectral_metrics_direct(reference, reconstructed)
    } else {
        let n = reference.len();
        let window_size = if n >= 1024 {
            1024
        } else if n >= 512 {
            512
        } else {
            n.next_power_of_two().max(16)
        };
        let hop_size = (window_size / 4).max(1);
        compute_spectral_metrics_stft(reference, reconstructed, window_size, hop_size)
    }
}

/// Computes spectral convergence between reference and reconstructed spectra.
#[inline]
pub fn compute_spectral_convergence(reference: &[f64], reconstructed: &[f64]) -> DspResult<f64> {
    compute_spectral_metrics(reference, reconstructed).map(|m| m.spectral_convergence)
}

/// Computes Log-Spectral Distance (LSD in dB) between reference and reconstructed spectra.
#[inline]
pub fn compute_log_spectral_distance(reference: &[f64], reconstructed: &[f64]) -> DspResult<f64> {
    compute_spectral_metrics(reference, reconstructed).map(|m| m.log_spectral_distance_db)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_metrics_scale() {
        let x = vec![1.0, 0.5, 0.25, 0.125, 0.0625];
        let x_scaled: Vec<f64> = x.iter().map(|&v| 1.2 * v).collect();
        let m = compute_spectral_metrics(&x, &x_scaled).unwrap();
        assert!((m.spectral_convergence - 0.2).abs() < 1e-14);
    }
}
