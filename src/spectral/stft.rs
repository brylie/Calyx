//! Short-Time Fourier Transform (STFT) sliding-block engine.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library,
//! zero external dependencies, safe Rust (#![deny(unsafe_code)]).

use std::f64::consts::PI;
use crate::error::{DspError, DspResult};
use crate::math::{generate_window, Complex, Matrix, WindowType};
use super::fft::Fft;
use super::peaks::PeakDetector;
use super::residual::StochasticResidualExtractor;
use super::traits::{
    FourierTransform, HarmonicPartial, SpectralAnalyzer, SpectralFrame, SpectralTrajectory,
};

/// Short-Time Fourier Transform engine supporting sliding analysis, resynthesis, and trajectories.
#[derive(Debug, Clone)]
pub struct Stft {
    /// Window length in samples.
    pub window_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Analytical window function type.
    pub window_type: WindowType,
    /// FFT zero-padded size (power of two).
    pub fft_size: usize,
    /// Default sampling rate in Hz (default: 48000.0).
    pub sample_rate: f64,
    /// Number of subband noise residual filterbank bands (default: 16).
    pub num_noise_bands: usize,
    /// Precomputed window coefficients.
    window: Vec<f64>,
    /// Precomputed FFT execution plan.
    fft: Fft,
}

fn invert_matrix(a_mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let dim = a_mat.len();
    let mut a = a_mat.to_vec();
    let mut inv = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        inv[i][i] = 1.0;
    }

    for i in 0..dim {
        let mut max_row = i;
        let mut max_val = a[i][i].abs();
        for r in (i + 1)..dim {
            if a[r][i].abs() > max_val {
                max_val = a[r][i].abs();
                max_row = r;
            }
        }
        if max_row != i {
            a.swap(i, max_row);
            inv.swap(i, max_row);
        }

        let pivot = a[i][i];
        if pivot.abs() < 1e-12 {
            return None;
        }

        for j in 0..dim {
            a[i][j] /= pivot;
            inv[i][j] /= pivot;
        }

        for r in 0..dim {
            if r != i {
                let factor = a[r][i];
                for j in 0..dim {
                    a[r][j] -= factor * a[i][j];
                    inv[r][j] -= factor * inv[i][j];
                }
            }
        }
    }

    Some(inv)
}

struct HarmonicModel {
    f0: f64,
    f_ks: Vec<f64>,
    harmonic_indices: Vec<usize>,
    inv_m: Vec<Vec<f64>>,
}

impl HarmonicModel {
    fn detect(signal: &[f64], sample_rate: f64, window_size: usize) -> Option<Self> {
        let l = 2048.min(signal.len());
        if l < 512 || sample_rate <= 0.0 {
            return None;
        }

        let n_fft = l.next_power_of_two();
        let w_init = generate_window(WindowType::Hann, l).ok()?;
        let mut buf_init: Vec<Complex<f64>> = signal[..l]
            .iter()
            .zip(w_init.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();
        buf_init.resize(n_fft, Complex::new(0.0, 0.0));
        let fft_init = Fft::new(n_fft).ok()?;
        fft_init.forward_complex_inplace(&mut buf_init).ok()?;

        let mag_init: Vec<f64> = buf_init[..n_fft / 2 + 1].iter().map(|c| c.norm()).collect();
        let peak_detector = PeakDetector::default();
        let peaks_init = peak_detector.detect(&mag_init, sample_rate);
        let max_amp = peaks_init.iter().map(|p| p.amplitude).fold(0.0_f64, f64::max);
        if max_amp <= 0.0 {
            return None;
        }

        let mut sig_peaks: Vec<_> = peaks_init
            .into_iter()
            .filter(|p| p.amplitude > 0.02 * max_amp && p.frequency_hz > 20.0)
            .collect();
        sig_peaks.sort_by(|a, b| a.frequency_hz.partial_cmp(&b.frequency_hz).unwrap());
        if sig_peaks.is_empty() {
            return None;
        }

        // FM / Chirp detection: check if dominant frequency sweeps across time
        if signal.len() >= 2 * l {
            let test_offset = (signal.len() * 3 / 4).saturating_sub(l / 2);
            let test_slice = &signal[test_offset..test_offset + l];
            let mut buf_test: Vec<Complex<f64>> = test_slice
                .iter()
                .zip(w_init.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();
            buf_test.resize(n_fft, Complex::new(0.0, 0.0));
            if fft_init.forward_complex_inplace(&mut buf_test).is_ok() {
                let mag_test: Vec<f64> = buf_test[..n_fft / 2 + 1].iter().map(|c| c.norm()).collect();
                let peaks_test = peak_detector.detect(&mag_test, sample_rate);
                if let Some(p_max) = peaks_test.iter().max_by(|a, b| a.amplitude.partial_cmp(&b.amplitude).unwrap()) {
                    if (p_max.frequency_hz - sig_peaks[0].frequency_hz).abs() > 30.0 {
                        return None; // Dominant frequency moved by > 30 Hz -> chirp / FM
                    }
                }
            }
        }

        let f_cand = sig_peaks[0].frequency_hz;
        let mut matched = Vec::new();
        for p in &sig_peaks {
            let k = (p.frequency_hz / f_cand).round() as usize;
            if k >= 1 && k <= 32 {
                let exp_f = (k as f64) * f_cand;
                if (p.frequency_hz - exp_f).abs() < 0.35 * f_cand {
                    matched.push((k, p.frequency_hz));
                }
            }
        }

        if matched.is_empty() {
            return None;
        }

        // First-principles linear regression on (f_k / k)^2 = f0^2 + (f0^2 * B) * k^2
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_xx = 0.0_f64;
        let mut sum_xy = 0.0_f64;
        let mut n_pts = 0_usize;

        for &(k, fk) in &matched {
            if k >= 1 && k <= 12 {
                let k_f = k as f64;
                let x = k_f * k_f;
                let y = (fk / k_f).powi(2);
                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_xy += x * y;
                n_pts += 1;
            }
        }

        let (f0, b_est) = if n_pts >= 2 {
            let n_f = n_pts as f64;
            let denom = n_f * sum_xx - sum_x * sum_x;
            if denom.abs() > 1e-12 {
                let c1 = (n_f * sum_xy - sum_x * sum_y) / denom;
                let c0 = (sum_y - c1 * sum_x) / n_f;
                if c0 > 100.0 {
                    let f0_val = c0.sqrt();
                    let b_val = (c1 / c0).clamp(0.0, 0.01);
                    (f0_val, b_val)
                } else {
                    (f_cand, 0.0)
                }
            } else {
                (f_cand, 0.0)
            }
        } else {
            (f_cand, 0.0)
        };

        // Now find the maximum active harmonic up to 32 using the inharmonic series
        let mut k_last = 1usize;
        for k in 1..=32 {
            let k_f = k as f64;
            let exp_f = k_f * f0 * (1.0 + b_est * k_f * k_f).sqrt();
            if exp_f >= sample_rate / 2.0 {
                break;
            }
            if sig_peaks.iter().any(|p| (p.frequency_hz - exp_f).abs() < 0.35 * f0) {
                k_last = k;
            }
        }

        let mut harmonic_indices = Vec::new();
        let mut f_ks = Vec::new();
        for k in 1..=k_last {
            let k_f = k as f64;
            let f_k = k_f * f0 * (1.0 + b_est * k_f * k_f).sqrt();
            if f_k < sample_rate / 2.0 {
                harmonic_indices.push(k);
                f_ks.push(f_k);
            }
        }

        if f_ks.is_empty() {
            return None;
        }

        let k_len = f_ks.len();
        let dim = 2 * k_len;
        let mut m_mat = vec![vec![0.0; dim]; dim];
        for n in 0..window_size {
            let mut row = Vec::with_capacity(dim);
            for &fk in &f_ks {
                let omega = 2.0 * PI * fk * (n as f64) / sample_rate;
                row.push(omega.cos());
            }
            for &fk in &f_ks {
                let omega = 2.0 * PI * fk * (n as f64) / sample_rate;
                row.push(omega.sin());
            }
            for i in 0..dim {
                for j in 0..dim {
                    m_mat[i][j] += row[i] * row[j];
                }
            }
        }
        for i in 0..dim {
            m_mat[i][i] += 1e-6;
        }

        let inv_m = invert_matrix(&m_mat)?;
        Some(Self {
            f0,
            f_ks,
            harmonic_indices,
            inv_m,
        })
    }
}

impl Stft {
    /// Construct a new STFT instance with default fft_size equal to window_size.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] or [`DspError::NonPowerOfTwo`] if parameters violate preconditions.
    pub fn new(window_size: usize, hop_size: usize, window_type: WindowType) -> DspResult<Self> {
        Self::with_fft_size(window_size, hop_size, window_type, window_size)
    }

    /// Construct a new STFT instance with configurable FFT zero-padding size.
    ///
    /// # Errors
    /// - Returns [`DspError::InvalidParameter`] if `hop_size == 0`.
    /// - Returns [`DspError::InvalidParameter`] if `hop_size > window_size`.
    /// - Returns [`DspError::InvalidParameter`] if `window_size < 2`.
    /// - Returns [`DspError::InvalidParameter`] if `window_size > fft_size`.
    /// - Returns [`DspError::NonPowerOfTwo`] if `fft_size` is not a power of two.
    pub fn with_fft_size(
        window_size: usize,
        hop_size: usize,
        window_type: WindowType,
        fft_size: usize,
    ) -> DspResult<Self> {
        if hop_size == 0 {
            return Err(DspError::invalid_parameter(
                "hop_size",
                "Hop size must be greater than zero",
            ));
        }
        if hop_size > window_size {
            return Err(DspError::invalid_parameter(
                "hop_size",
                "Hop size cannot exceed window size",
            ));
        }
        if window_size < 2 {
            return Err(DspError::invalid_parameter(
                "window_size",
                "Window size must be at least 2",
            ));
        }
        if window_size > fft_size {
            return Err(DspError::invalid_parameter(
                "window_size",
                "Window size cannot exceed FFT size",
            ));
        }
        if !fft_size.is_power_of_two() {
            return Err(DspError::NonPowerOfTwo { size: fft_size });
        }

        let window = generate_window(window_type, window_size)?;
        let fft = Fft::new(fft_size)?;

        Ok(Self {
            window_size,
            hop_size,
            window_type,
            fft_size,
            sample_rate: 48000.0,
            num_noise_bands: 16,
            window,
            fft,
        })
    }

    /// Set default sample rate on STFT instance.
    pub fn with_sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set number of subband noise residual filterbank bands.
    pub fn with_noise_bands(mut self, num_noise_bands: usize) -> Self {
        self.num_noise_bands = num_noise_bands;
        self
    }

    /// Forward STFT producing a vector of complex spectral frames.
    ///
    /// # Errors
    /// Returns [`DspError::EmptyInput`] if `signal` is empty.
    pub fn forward(&self, signal: &[f64]) -> DspResult<Vec<Vec<Complex<f64>>>> {
        if signal.is_empty() {
            return Err(DspError::EmptyInput);
        }

        for (i, &s) in signal.iter().enumerate() {
            if !s.is_finite() {
                return Err(DspError::numerical_instability(format!(
                    "STFT input sample at index {} is not finite: {}",
                    i, s
                )));
            }
        }

        let sig_len = signal.len();
        let mut frames = Vec::new();
        let mut offset = 0;

        // Loop extracts sliding blocks; handles sub-window audio via zero-padding
        while offset < sig_len || (offset == 0 && sig_len < self.window_size) {
            let mut frame_buf = vec![Complex::zero(); self.fft_size];

            for (n, item) in frame_buf.iter_mut().enumerate().take(self.window_size) {
                let sample_idx = offset + n;
                let sample = if sample_idx < sig_len {
                    signal[sample_idx]
                } else {
                    0.0
                };
                *item = Complex::new(sample * self.window[n], 0.0);
            }

            self.fft.forward_complex_inplace(&mut frame_buf)?;
            frames.push(frame_buf);

            offset += self.hop_size;
        }

        Ok(frames)
    }

    /// Forward STFT producing a real magnitude Matrix (rows = frames, cols = half_bins).
    pub fn forward_magnitude_matrix(&self, signal: &[f64]) -> DspResult<Matrix<f64>> {
        let frames = self.forward(signal)?;
        let num_frames = frames.len();
        let half_bins = self.fft_size / 2 + 1;
        let mut flat_data = Vec::with_capacity(num_frames * half_bins);
        for frame in frames {
            for c in &frame[0..half_bins] {
                flat_data.push(c.norm());
            }
        }
        Matrix::from_vec(num_frames, half_bins, flat_data)
    }

    /// Inverse STFT via Weighted Overlap-Add (WOLA).
    ///
    /// # Errors
    /// Returns [`DspError::EmptyInput`] if `spectrogram` is empty.
    pub fn inverse(
        &self,
        spectrogram: &[Vec<Complex<f64>>],
        target_len: Option<usize>,
    ) -> DspResult<Vec<f64>> {
        if spectrogram.is_empty() {
            return Err(DspError::EmptyInput);
        }

        let num_frames = spectrogram.len();
        let total_output_len = (num_frames - 1) * self.hop_size + self.window_size;

        let mut output = vec![0.0; total_output_len];
        let mut window_weight_sum = vec![0.0; total_output_len];

        for (m, frame) in spectrogram.iter().enumerate() {
            if frame.len() != self.fft_size {
                return Err(DspError::BufferLengthMismatch {
                    expected: self.fft_size,
                    actual: frame.len(),
                });
            }

            let mut time_buf = frame.clone();
            self.fft.inverse_complex_inplace(&mut time_buf)?;

            let start = m * self.hop_size;
            for (n, sample_c) in time_buf.iter().enumerate().take(self.window_size) {
                let out_idx = start + n;
                let sample = sample_c.re * self.window[n];
                output[out_idx] += sample;
                window_weight_sum[out_idx] += self.window[n] * self.window[n];
            }
        }

        // Normalize by accumulated window power
        const EPSILON: f64 = 1e-12;
        for i in 0..total_output_len {
            if window_weight_sum[i] > EPSILON {
                output[i] /= window_weight_sum[i];
            }
        }

        if let Some(desired_len) = target_len {
            output.resize(desired_len, 0.0);
        }

        Ok(output)
    }

    /// High-level analysis returning SpectralTrajectory using the configured sample rate.
    pub fn process(&self, signal: &[f64]) -> DspResult<SpectralTrajectory> {
        self.process_with_sample_rate(signal, self.sample_rate)
    }

    /// High-level analysis returning SpectralTrajectory with custom sample rate.
    pub fn process_with_sample_rate(
        &self,
        signal: &[f64],
        sample_rate: f64,
    ) -> DspResult<SpectralTrajectory> {
        if signal.is_empty() {
            return Err(DspError::EmptyInput);
        }
        if sample_rate <= 0.0 || !sample_rate.is_finite() {
            return Err(DspError::invalid_parameter(
                "sample_rate",
                format!("Sample rate must be positive and finite, got {}", sample_rate),
            ));
        }

        let frames_complex = self.forward(signal)?;
        let half_bins = self.fft_size / 2 + 1;
        let mut traj_frames: Vec<SpectralFrame> = Vec::with_capacity(frames_complex.len());

        let peak_detector = PeakDetector::default();
        let residual_extractor = StochasticResidualExtractor::new(self.num_noise_bands.max(1))?;
        let harmonic_model = HarmonicModel::detect(signal, sample_rate, self.window_size);

        for (m, frame_c) in frames_complex.into_iter().enumerate() {
            let timestamp = (m as f64) * (self.hop_size as f64) / sample_rate;
            let offset = m * self.hop_size;
            let frame_len = self.window_size.min(signal.len().saturating_sub(offset));
            let frame_slice = &signal[offset..offset + frame_len];

            // One-sided magnitude spectrum
            let mag_spec: Vec<f64> = frame_c[0..half_bins].iter().map(|c| c.norm()).collect();

            if let Some(model) = &harmonic_model {
                let k_len = model.f_ks.len();
                let dim = 2 * k_len;
                let mut partials = Vec::with_capacity(k_len);

                if frame_len == self.window_size {
                    let mut rhs = vec![0.0_f64; dim];
                    for (n, &s) in frame_slice.iter().enumerate() {
                        let n_f = n as f64;
                        for (k, &fk) in model.f_ks.iter().enumerate() {
                            let omega: f64 = 2.0 * PI * fk * n_f / sample_rate;
                            rhs[k] += s * omega.cos();
                            rhs[k_len + k] += s * omega.sin();
                        }
                    }

                    let mut coeffs = vec![0.0_f64; dim];
                    for i in 0..dim {
                        for j in 0..dim {
                            coeffs[i] += model.inv_m[i][j] * rhs[j];
                        }
                    }

                    let mut max_p_amp = 0.0_f64;
                    let mut temp = Vec::with_capacity(k_len);
                    for k in 0..k_len {
                        let c_k: f64 = coeffs[k];
                        let s_k: f64 = coeffs[k_len + k];
                        let amp: f64 = (c_k * c_k + s_k * s_k).sqrt();
                        let phase: f64 = (-s_k).atan2(c_k);
                        if amp > max_p_amp {
                            max_p_amp = amp;
                        }
                        temp.push((amp, phase));
                    }

                    let threshold = (0.005 * max_p_amp).max(1e-5);
                    for (k, &(amp, phase)) in temp.iter().enumerate() {
                        if amp > threshold {
                            partials.push(HarmonicPartial {
                                harmonic_index: model.harmonic_indices[k],
                                frequency: model.f_ks[k],
                                amplitude: amp,
                                phase,
                            });
                        }
                    }
                } else if let Some(last_frame) = traj_frames.last() {
                    // Tail frame: advance phases analytically
                    let hop_dt = (self.hop_size as f64) / sample_rate;
                    for p in &last_frame.partials {
                        let omega = 2.0 * PI * p.frequency;
                        partials.push(HarmonicPartial {
                            harmonic_index: p.harmonic_index,
                            frequency: p.frequency,
                            amplitude: p.amplitude,
                            phase: p.phase + omega * hop_dt,
                        });
                    }
                }

                let mut det_signal = vec![0.0_f64; self.window_size];
                for p in &partials {
                    let omega = 2.0 * PI * p.frequency / sample_rate;
                    for (n, s) in det_signal.iter_mut().enumerate() {
                        *s += p.amplitude * (omega * (n as f64) + p.phase).cos();
                    }
                }
                let mut res_signal = vec![0.0_f64; self.window_size];
                for n in 0..frame_len {
                    res_signal[n] = (frame_slice[n] - det_signal[n]) * self.window[n];
                }
                let mut res_complex: Vec<Complex<f64>> = res_signal.iter().map(|&s| Complex::new(s, 0.0)).collect();
                res_complex.resize(self.fft_size, Complex::zero());
                let _ = self.fft.forward_complex_inplace(&mut res_complex);
                let res_mag: Vec<f64> = res_complex[0..half_bins].iter().map(|c| c.norm()).collect();
                let residual = residual_extractor.extract_from_residual(&mag_spec, &res_mag);
                traj_frames.push(SpectralFrame {
                    timestamp,
                    partials,
                    residual,
                    fundamental_freq: model.f0,
                });
                continue;
            }

            // Fallback for dynamic / chirp / unmodeled signals
            if frame_len < self.window_size / 3 && !traj_frames.is_empty() {
                let last_frame = traj_frames.last().unwrap();
                let hop_dt = (self.hop_size as f64) / sample_rate;
                let partials: Vec<HarmonicPartial> = last_frame
                    .partials
                    .iter()
                    .map(|p| HarmonicPartial {
                        harmonic_index: p.harmonic_index,
                        frequency: p.frequency,
                        amplitude: p.amplitude,
                        phase: p.phase + 2.0 * PI * p.frequency * hop_dt,
                    })
                    .collect();
                traj_frames.push(SpectralFrame {
                    timestamp,
                    fundamental_freq: 0.0,
                    partials,
                    residual: last_frame.residual.clone(),
                });
                continue;
            }

            let max_amp = mag_spec.iter().fold(0.0_f64, |a, &b| a.max(b));
            let peak_thresh = (0.05 * max_amp).max(1e-4);
            let detected_peaks: Vec<_> = peak_detector
                .detect(&mag_spec, sample_rate)
                .into_iter()
                .filter(|p| p.amplitude > peak_thresh)
                .collect();

            let frame_rms = if frame_len > 0 {
                (frame_slice.iter().map(|&s| s * s).sum::<f64>() / frame_len as f64).sqrt()
            } else {
                0.0
            };
            let single_component_amp = if detected_peaks.len() == 1 && frame_rms > 1e-4 {
                Some((frame_rms * 2.0_f64.sqrt()) * (self.window_size as f64 / 4.0))
            } else {
                None
            };

            let partials: Vec<HarmonicPartial> = detected_peaks
                .into_iter()
                .enumerate()
                .map(|(idx, p)| {
                    let amp = if let Some(true_amp) = single_component_amp {
                        true_amp.max(p.amplitude)
                    } else {
                        p.amplitude
                    };
                    let phase = if p.bin_index < frame_c.len() {
                        frame_c[p.bin_index].arg()
                    } else {
                        0.0
                    };
                    HarmonicPartial {
                        harmonic_index: idx + 1,
                        frequency: p.frequency_hz,
                        amplitude: amp,
                        phase,
                    }
                })
                .collect();

            let mut det_signal = vec![0.0_f64; self.window_size];
            let df = sample_rate / (self.fft_size as f64);
            let win_size_f = self.window_size as f64;
            for p in &partials {
                let phys_amp = 4.0 * p.amplitude / win_size_f;
                let k_bin = (p.frequency / df).round();
                let delta_k = (p.frequency - k_bin * df) / df;
                let phase_0 = p.phase - PI * delta_k * (win_size_f - 1.0) / win_size_f;
                let omega = 2.0 * PI * p.frequency / sample_rate;
                for (n, s) in det_signal.iter_mut().enumerate() {
                    *s += phys_amp * (omega * (n as f64) + phase_0).cos();
                }
            }
            for (n, s) in det_signal.iter_mut().enumerate() {
                *s *= self.window[n];
            }
            let mut det_complex: Vec<Complex<f64>> = det_signal.iter().map(|&s| Complex::new(s, 0.0)).collect();
            det_complex.resize(self.fft_size, Complex::zero());
            let _ = self.fft.forward_complex_inplace(&mut det_complex);
            let det_mag: Vec<f64> = det_complex[0..half_bins].iter().map(|c| c.norm()).collect();

            let residual = residual_extractor.extract(&mag_spec, &det_mag);
            traj_frames.push(SpectralFrame {
                timestamp,
                fundamental_freq: 0.0,
                partials,
                residual,
            });
        }

        Ok(SpectralTrajectory {
            sample_rate,
            hop_size: self.hop_size,
            frames: traj_frames,
        })
    }
}

impl SpectralAnalyzer for Stft {
    #[inline]
    fn analyze(&self, signal: &[f64], sample_rate: f64) -> Result<SpectralTrajectory, DspError> {
        self.process_with_sample_rate(signal, sample_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stft_parameter_validation() {
        assert!(Stft::new(512, 0, WindowType::Hann).is_err());
        assert!(Stft::new(256, 512, WindowType::Hann).is_err());
        assert!(Stft::new(1, 1, WindowType::Hann).is_err());
        assert!(Stft::with_fft_size(512, 256, WindowType::Hann, 500).is_err());
    }

    #[test]
    fn test_stft_empty_rejection() {
        let stft = Stft::new(512, 256, WindowType::Hann).unwrap();
        assert_eq!(stft.process(&[]), Err(DspError::EmptyInput));
        assert_eq!(stft.forward(&[]), Err(DspError::EmptyInput));
    }

    #[test]
    fn test_stft_short_audio() {
        let short_audio = vec![0.5; 200];
        let stft = Stft::new(512, 256, WindowType::Hann).unwrap();
        let traj = stft.process(&short_audio);
        assert!(traj.is_ok());
    }

    #[test]
    fn test_stft_remainder_audio() {
        let audio = vec![0.1; 1000];
        let stft = Stft::new(256, 128, WindowType::Hann).unwrap();
        let traj = stft.process(&audio).unwrap();
        assert!(!traj.frames.is_empty());
    }

    #[test]
    fn test_stft_reconstruction_wola() {
        let n = 2048;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).sin()).collect();
        let stft = Stft::new(512, 128, WindowType::Hann).unwrap();
        let spec = stft.forward(&x).unwrap();
        let recon = stft.inverse(&spec, Some(n)).unwrap();

        // Check steady-state middle segment reconstruction
        let start = 512;
        let end = n - 512;
        for i in start..end {
            assert!(
                (x[i] - recon[i]).abs() < 1e-6,
                "Mismatch at {}: orig={}, recon={}",
                i,
                x[i],
                recon[i]
            );
        }
    }

    #[test]
    fn test_stft_trajectory_and_partial_tracking_e2e() {
        use std::f64::consts::PI;
        use super::super::tracking::HarmonicPartialTracker;
        use super::super::traits::ResidualAnalyzer;

        let fs = 48000.0;
        let duration = 0.5;
        let n_samples = (fs * duration) as usize;
        let f0 = 110.0;
        let b = 0.0004;
        let num_partials = 24;

        // Synthesize inharmonic multi-tone signal
        let mut signal = vec![0.0; n_samples];
        for k in 1..=num_partials {
            let k_f = k as f64;
            let freq = k_f * f0 * (1.0 + b * k_f * k_f).sqrt();
            let amp = 0.8 / k_f;
            for (i, sample) in signal.iter_mut().enumerate() {
                let t = i as f64 / fs;
                *sample += amp * (2.0 * PI * freq * t).sin();
            }
        }

        let stft = Stft::with_fft_size(2048, 512, WindowType::Hann, 4096).unwrap();
        let trajectory = stft.process(&signal).unwrap();
        assert!(!trajectory.frames.is_empty());

        let tracker = HarmonicPartialTracker::new(f0, num_partials, b);
        let extracted_partials = tracker.track_trajectory(&trajectory);
        assert_eq!(extracted_partials.len(), num_partials);

        for (k_idx, partial) in extracted_partials.iter().enumerate() {
            let k = (k_idx + 1) as f64;
            let expected_f = k * f0 * (1.0 + b * k.powi(2)).sqrt();
            let err = (partial.frequency - expected_f).abs();
            assert!(
                err < 0.5,
                "Partial {} error: expected {:.2} Hz, got {:.2} Hz (err={:.3} Hz)",
                k,
                expected_f,
                partial.frequency,
                err
            );
        }

        let extractor = StochasticResidualExtractor::new(16);
        let residual_bands = extractor.extract_from_trajectory(&trajectory, &extracted_partials);
        assert_eq!(residual_bands.len(), 16);
    }
}
