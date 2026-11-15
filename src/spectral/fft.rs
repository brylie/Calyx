//! Radix-2 Cooley-Tukey Fast Fourier Transform (FFT).
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library,
//! zero external dependencies, safe Rust (#![deny(unsafe_code)]).

use std::f64::consts::PI;

use crate::error::{DspError, DspResult};
use crate::math::Complex;
use crate::spectral::traits::FourierTransform;

/// Radix-2 Decimation-in-Time Fast Fourier Transform engine.
#[derive(Debug, Clone)]
pub struct Fft {
    size: usize,
    twiddles: Vec<Complex<f64>>,
}

impl Fft {
    /// Construct a new precomputed FFT plan for a specific power-of-two size.
    ///
    /// # Errors
    /// - Returns [`DspError::EmptyInput`] if `size == 0`.
    /// - Returns [`DspError::NonPowerOfTwo`] if `size` is not a power of two.
    pub fn new(size: usize) -> DspResult<Self> {
        if size == 0 {
            return Err(DspError::EmptyInput);
        }
        if !size.is_power_of_two() {
            return Err(DspError::NonPowerOfTwo { size });
        }

        let half_size = size / 2;
        let mut twiddles = Vec::with_capacity(half_size);
        let factor = -2.0 * PI / (size as f64);
        for k in 0..half_size {
            let theta = factor * (k as f64);
            twiddles.push(Complex::from_polar(1.0, theta));
        }

        Ok(Self { size, twiddles })
    }

    /// Size of the FFT plan.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// In-place forward Radix-2 Cooley-Tukey FFT.
    ///
    /// # Errors
    /// - Returns [`DspError::EmptyInput`] if `buffer` is empty.
    /// - Returns [`DspError::NonPowerOfTwo`] if `buffer.len()` is not a power of two.
    /// - Returns [`DspError::NumericalInstability`] if any sample is NaN or infinite.
    pub fn forward_complex_radix2(buffer: &mut [Complex<f64>]) -> DspResult<()> {
        let n = buffer.len();
        if n == 0 {
            return Err(DspError::EmptyInput);
        }
        if !n.is_power_of_two() {
            return Err(DspError::NonPowerOfTwo { size: n });
        }
        if n == 1 {
            if !buffer[0].re.is_finite() || !buffer[0].im.is_finite() {
                return Err(DspError::numerical_instability(
                    "FFT input contains non-finite values",
                ));
            }
            return Ok(());
        }

        let fft = Self::new(n)?;
        fft.execute(buffer, false)
    }

    /// In-place inverse Radix-2 Cooley-Tukey FFT with 1/N scaling.
    ///
    /// # Errors
    /// - Returns [`DspError::EmptyInput`] if `buffer` is empty.
    /// - Returns [`DspError::NonPowerOfTwo`] if `buffer.len()` is not a power of two.
    /// - Returns [`DspError::NumericalInstability`] if any sample is NaN or infinite.
    pub fn inverse_complex_radix2(buffer: &mut [Complex<f64>]) -> DspResult<()> {
        let n = buffer.len();
        if n == 0 {
            return Err(DspError::EmptyInput);
        }
        if !n.is_power_of_two() {
            return Err(DspError::NonPowerOfTwo { size: n });
        }
        if n == 1 {
            if !buffer[0].re.is_finite() || !buffer[0].im.is_finite() {
                return Err(DspError::numerical_instability(
                    "FFT input contains non-finite values",
                ));
            }
            return Ok(());
        }

        let fft = Self::new(n)?;
        fft.execute(buffer, true)
    }

    /// Internal execution engine for forward and inverse transforms.
    pub fn execute(&self, buffer: &mut [Complex<f64>], inverse: bool) -> DspResult<()> {
        let n = buffer.len();
        if n != self.size {
            return Err(DspError::BufferLengthMismatch {
                expected: self.size,
                actual: n,
            });
        }
        for (i, c) in buffer.iter().enumerate() {
            if !c.re.is_finite() || !c.im.is_finite() {
                return Err(DspError::numerical_instability(format!(
                    "FFT sample at index {} is not finite: {:?}",
                    i, c
                )));
            }
        }
        if n <= 1 {
            return Ok(());
        }

        // 1. Bit-reversal permutation via bitwise operations
        let shift = usize::BITS - n.trailing_zeros();
        for i in 0..n {
            let j = i.reverse_bits() >> shift;
            if i < j {
                buffer.swap(i, j);
            }
        }

        // 2. Cooley-Tukey DIT Butterfly Stages
        let mut len = 2;
        while len <= n {
            let half_len = len / 2;
            let stride = self.size / len;

            for step_start in (0..n).step_by(len) {
                for k in 0..half_len {
                    let twiddle = if inverse {
                        self.twiddles[k * stride].conj()
                    } else {
                        self.twiddles[k * stride]
                    };

                    let u = buffer[step_start + k];
                    let v = buffer[step_start + k + half_len] * twiddle;

                    buffer[step_start + k] = u + v;
                    buffer[step_start + k + half_len] = u - v;
                }
            }
            len <<= 1;
        }

        // 3. Normalization on inverse
        if inverse {
            let scale = 1.0 / (n as f64);
            for x in buffer.iter_mut() {
                *x = Complex::new(x.re * scale, x.im * scale);
            }
        }

        Ok(())
    }

    /// Forward transform on complex slice, returning a new vector.
    pub fn fft(&self, input: &[Complex<f64>]) -> DspResult<Vec<Complex<f64>>> {
        let mut buf = input.to_vec();
        self.execute(&mut buf, false)?;
        Ok(buf)
    }

    /// Inverse transform on complex slice, returning a new vector.
    pub fn ifft(&self, input: &[Complex<f64>]) -> DspResult<Vec<Complex<f64>>> {
        let mut buf = input.to_vec();
        self.execute(&mut buf, true)?;
        Ok(buf)
    }

    /// Forward real transform on real slice, returning complex spectrum.
    pub fn fft_real(&self, input: &[f64]) -> DspResult<Vec<Complex<f64>>> {
        let mut buf: Vec<Complex<f64>> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
        self.execute(&mut buf, false)?;
        Ok(buf)
    }

    /// Inverse real transform on complex spectrum, returning real samples.
    pub fn ifft_real(&self, input: &[Complex<f64>]) -> DspResult<Vec<f64>> {
        let mut buf = input.to_vec();
        self.execute(&mut buf, true)?;
        Ok(buf.into_iter().map(|c| c.re).collect())
    }
}

impl FourierTransform for Fft {
    fn forward_real(&self, input: &[f64]) -> DspResult<Vec<Complex<f64>>> {
        self.fft_real(input)
    }

    fn forward_complex(&self, input: &[Complex<f64>]) -> DspResult<Vec<Complex<f64>>> {
        self.fft(input)
    }

    fn forward_complex_inplace(&self, buffer: &mut [Complex<f64>]) -> DspResult<()> {
        self.execute(buffer, false)
    }

    fn inverse_real(&self, spectrum: &[Complex<f64>]) -> DspResult<Vec<f64>> {
        self.ifft_real(spectrum)
    }

    fn inverse_complex(&self, spectrum: &[Complex<f64>]) -> DspResult<Vec<Complex<f64>>> {
        self.ifft(spectrum)
    }

    fn inverse_complex_inplace(&self, buffer: &mut [Complex<f64>]) -> DspResult<()> {
        self.execute(buffer, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_empty_slice_rejection() {
        let mut empty_buf: Vec<Complex<f64>> = vec![];
        assert!(matches!(
            Fft::forward_complex_radix2(&mut empty_buf),
            Err(DspError::EmptyInput) | Err(DspError::NonPowerOfTwo { size: 0 })
        ));
    }

    #[test]
    fn test_fft_non_power_of_two_rejection() {
        for &size in &[3, 5, 7, 300, 500, 1000] {
            let mut complex_buf = vec![Complex::zero(); size];
            assert!(matches!(
                Fft::forward_complex_radix2(&mut complex_buf),
                Err(DspError::NonPowerOfTwo { size: s }) if s == size
            ));
        }
    }

    #[test]
    fn test_fft_minimal_power_of_two_sizes() {
        for &size in &[1, 2, 4] {
            let mut buf: Vec<Complex<f64>> = (0..size)
                .map(|i| Complex::new(i as f64 + 1.0, 0.0))
                .collect();
            let original = buf.clone();
            Fft::forward_complex_radix2(&mut buf).unwrap();
            Fft::inverse_complex_radix2(&mut buf).unwrap();
            for i in 0..size {
                assert!((buf[i].re - original[i].re).abs() < 1e-12);
                assert!(buf[i].im.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_fft_large_size_65536() {
        let n = 65536;
        let mut buf = vec![Complex::zero(); n];
        buf[0] = Complex::new(1.0, 0.0); // Unit impulse
        Fft::forward_complex_radix2(&mut buf).unwrap();
        // Spectrum of delta is flat 1.0 across all bins
        assert!((buf[0].re - 1.0).abs() < 1e-10);
        assert!((buf[n / 2].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fft_energy_conservation_parseval() {
        let n = 1024;
        let mut time_buf = Vec::with_capacity(n);
        for i in 0..n {
            time_buf.push(Complex::new((i as f64 * 0.1).sin(), (i as f64 * 0.2).cos()));
        }
        let mut freq_buf = time_buf.clone();
        Fft::forward_complex_radix2(&mut freq_buf).unwrap();

        let time_energy: f64 = time_buf.iter().map(|c| c.norm_sq()).sum();
        let freq_energy: f64 = freq_buf.iter().map(|c| c.norm_sq()).sum::<f64>() / (n as f64);
        assert!((time_energy - freq_energy).abs() / time_energy < 1e-10);
    }
}
