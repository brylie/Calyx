//! Direct $O(N^2)$ Discrete Fourier Transform.
//!
//! Complies strictly with Demo Integrity Mode: zero external dependencies,
//! 100% pure standard library, safe Rust (#![deny(unsafe_code)]).
//!
//! Computes exact forward and inverse Discrete Fourier Transforms for arbitrary
//! sequence lengths $N \ge 1$ without requiring power-of-two constraints.
//! Serves as the numerical ground-truth reference for validating fast Fourier transform algorithms.

use std::f64::consts::PI;

use crate::error::{DspError, DspResult};
use crate::math::Complex;
use crate::spectral::traits::FourierTransform;

/// First-principles $O(N^2)$ direct Discrete Fourier Transform engine.
///
/// Computes the exact forward transform:
/// $$X\[k\] = \sum_{n=0}^{N-1} x\[n\] \cdot e^{-i \frac{2\pi}{N} k n}, \quad k \in \{0, \dots, N-1\}$$
/// and exact inverse transform:
/// $$x\[n\] = \frac{1}{N} \sum_{k=0}^{N-1} X\[k\] \cdot e^{+i \frac{2\pi}{N} k n}, \quad n \in \{0, \dots, N-1\}$$
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Dft;

impl Dft {
    /// Construct a new direct DFT instance.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Perform a forward Discrete Fourier Transform on real-valued time-domain samples,
    /// returning the full two-sided complex frequency spectrum of length $N$.
    ///
    /// # Mathematical Formulation
    /// For $k \in \{0, \dots, N-1\}$:
    /// $$\text{Re}(X\[k\]) = \sum_{n=0}^{N-1} x\[n\] \cos\left(\frac{2\pi k n}{N}\right)$$
    /// $$\text{Im}(X\[k\]) = -\sum_{n=0}^{N-1} x\[n\] \sin\left(\frac{2\pi k n}{N}\right)$$
    ///
    /// # Errors
    /// - Returns [`DspError::EmptyInput`] if `input` is empty.
    /// - Returns [`DspError::NumericalInstability`] if any input sample is NaN or infinite.
    pub fn forward_real(input: &[f64]) -> DspResult<Vec<Complex<f64>>> {
        let n = input.len();
        if n == 0 {
            return Err(DspError::EmptyInput);
        }

        for (i, &val) in input.iter().enumerate() {
            if !val.is_finite() {
                return Err(DspError::numerical_instability(format!(
                    "DFT input sample at index {} is not finite: {}",
                    i, val
                )));
            }
        }

        let mut spectrum = Vec::with_capacity(n);
        let two_pi_over_n = 2.0 * PI / (n as f64);

        for k in 0..n {
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            for (idx, &x) in input.iter().enumerate() {
                let angle = two_pi_over_n * ((k * idx) as f64);
                let (sin_a, cos_a) = angle.sin_cos();
                sum_re += x * cos_a;
                sum_im -= x * sin_a;
            }
            spectrum.push(Complex::new(sum_re, sum_im));
        }

        Ok(spectrum)
    }

    /// Perform a forward Discrete Fourier Transform on complex-valued time-domain samples,
    /// returning the complex frequency spectrum of length $N$.
    ///
    /// # Errors
    /// - Returns [`DspError::EmptyInput`] if `input` is empty.
    /// - Returns [`DspError::NumericalInstability`] if any real or imaginary component is not finite.
    pub fn forward_complex(input: &[Complex<f64>]) -> DspResult<Vec<Complex<f64>>> {
        let n = input.len();
        if n == 0 {
            return Err(DspError::EmptyInput);
        }

        for (i, c) in input.iter().enumerate() {
            if !c.re.is_finite() || !c.im.is_finite() {
                return Err(DspError::numerical_instability(format!(
                    "DFT complex input at index {} is not finite: {:?}",
                    i, c
                )));
            }
        }

        let mut spectrum = Vec::with_capacity(n);
        let two_pi_over_n = 2.0 * PI / (n as f64);

        for k in 0..n {
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            for (idx, c) in input.iter().enumerate() {
                let angle = two_pi_over_n * ((k * idx) as f64);
                let (sin_a, cos_a) = angle.sin_cos();
                sum_re += c.re * cos_a + c.im * sin_a;
                sum_im += c.im * cos_a - c.re * sin_a;
            }
            spectrum.push(Complex::new(sum_re, sum_im));
        }

        Ok(spectrum)
    }

    /// Perform an in-place forward Discrete Fourier Transform on a mutable slice of complex numbers.
    ///
    /// # Errors
    /// - Returns [`DspError::EmptyInput`] if `buffer` is empty.
    /// - Returns [`DspError::NumericalInstability`] if any component is not finite.
    pub fn forward_complex_inplace(buffer: &mut [Complex<f64>]) -> DspResult<()> {
        let transformed = Self::forward_complex(buffer)?;
        buffer.copy_from_slice(&transformed);
        Ok(())
    }

    /// Perform an inverse Discrete Fourier Transform from a complex spectrum,
    /// reconstructing the real-valued time-domain signal of length $N$.
    ///
    /// Computes:
    /// $$x\[n\] = \frac{1}{N} \sum_{k=0}^{N-1} \left(\text{Re}(X\[k\]) \cos\left(\frac{2\pi k n}{N}\right) - \text{Im}(X\[k\]) \sin\left(\frac{2\pi k n}{N}\right)\right)$$
    ///
    /// # Errors
    /// - Returns [`DspError::EmptyInput`] if `spectrum` is empty.
    /// - Returns [`DspError::NumericalInstability`] if any spectrum bin is not finite.
    pub fn inverse_real(spectrum: &[Complex<f64>]) -> DspResult<Vec<f64>> {
        let n = spectrum.len();
        if n == 0 {
            return Err(DspError::EmptyInput);
        }

        for (i, c) in spectrum.iter().enumerate() {
            if !c.re.is_finite() || !c.im.is_finite() {
                return Err(DspError::numerical_instability(format!(
                    "DFT spectrum bin at index {} is not finite: {:?}",
                    i, c
                )));
            }
        }

        let inv_n = 1.0 / (n as f64);
        let two_pi_over_n = 2.0 * PI / (n as f64);
        let mut output = Vec::with_capacity(n);

        for idx in 0..n {
            let mut sum = 0.0;
            for (k, c) in spectrum.iter().enumerate() {
                let angle = two_pi_over_n * ((k * idx) as f64);
                let (sin_a, cos_a) = angle.sin_cos();
                sum += c.re * cos_a - c.im * sin_a;
            }
            output.push(sum * inv_n);
        }

        Ok(output)
    }

    /// Perform an inverse Discrete Fourier Transform on complex frequency bins,
    /// reconstructing complex time-domain samples of length $N$ scaled by $1/N$.
    ///
    /// # Errors
    /// - Returns [`DspError::EmptyInput`] if `spectrum` is empty.
    /// - Returns [`DspError::NumericalInstability`] if any spectrum bin is not finite.
    pub fn inverse_complex(spectrum: &[Complex<f64>]) -> DspResult<Vec<Complex<f64>>> {
        let n = spectrum.len();
        if n == 0 {
            return Err(DspError::EmptyInput);
        }

        for (i, c) in spectrum.iter().enumerate() {
            if !c.re.is_finite() || !c.im.is_finite() {
                return Err(DspError::numerical_instability(format!(
                    "DFT spectrum bin at index {} is not finite: {:?}",
                    i, c
                )));
            }
        }

        let inv_n = 1.0 / (n as f64);
        let two_pi_over_n = 2.0 * PI / (n as f64);
        let mut output = Vec::with_capacity(n);

        for idx in 0..n {
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            for (k, c) in spectrum.iter().enumerate() {
                let angle = two_pi_over_n * ((k * idx) as f64);
                let (sin_a, cos_a) = angle.sin_cos();
                sum_re += c.re * cos_a - c.im * sin_a;
                sum_im += c.re * sin_a + c.im * cos_a;
            }
            output.push(Complex::new(sum_re * inv_n, sum_im * inv_n));
        }

        Ok(output)
    }

    /// Perform an in-place inverse Discrete Fourier Transform on a mutable slice of complex numbers.
    ///
    /// # Errors
    /// - Returns [`DspError::EmptyInput`] if `buffer` is empty.
    /// - Returns [`DspError::NumericalInstability`] if any sample is not finite.
    pub fn inverse_complex_inplace(buffer: &mut [Complex<f64>]) -> DspResult<()> {
        let transformed = Self::inverse_complex(buffer)?;
        buffer.copy_from_slice(&transformed);
        Ok(())
    }
}

impl FourierTransform for Dft {
    #[inline]
    fn forward_real(&self, input: &[f64]) -> DspResult<Vec<Complex<f64>>> {
        Self::forward_real(input)
    }

    #[inline]
    fn forward_complex(&self, input: &[Complex<f64>]) -> DspResult<Vec<Complex<f64>>> {
        Self::forward_complex(input)
    }

    #[inline]
    fn forward_complex_inplace(&self, buffer: &mut [Complex<f64>]) -> DspResult<()> {
        Self::forward_complex_inplace(buffer)
    }

    #[inline]
    fn inverse_real(&self, spectrum: &[Complex<f64>]) -> DspResult<Vec<f64>> {
        Self::inverse_real(spectrum)
    }

    #[inline]
    fn inverse_complex(&self, spectrum: &[Complex<f64>]) -> DspResult<Vec<Complex<f64>>> {
        Self::inverse_complex(spectrum)
    }

    #[inline]
    fn inverse_complex_inplace(&self, buffer: &mut [Complex<f64>]) -> DspResult<()> {
        Self::inverse_complex_inplace(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dft_empty_input_rejection() {
        assert_eq!(Dft::forward_real(&[]), Err(DspError::EmptyInput));
        assert_eq!(Dft::forward_complex(&[]), Err(DspError::EmptyInput));
        assert_eq!(Dft::inverse_real(&[]), Err(DspError::EmptyInput));
        assert_eq!(Dft::inverse_complex(&[]), Err(DspError::EmptyInput));

        let mut empty = [];
        assert_eq!(Dft::forward_complex_inplace(&mut empty), Err(DspError::EmptyInput));
        assert_eq!(Dft::inverse_complex_inplace(&mut empty), Err(DspError::EmptyInput));
    }

    #[test]
    fn test_dft_length_one() {
        let input = [3.5];
        let spec = Dft::forward_real(&input).unwrap();
        assert_eq!(spec.len(), 1);
        assert!((spec[0].re - 3.5).abs() < 1e-12);
        assert!(spec[0].im.abs() < 1e-12);

        let recon = Dft::inverse_real(&spec).unwrap();
        assert_eq!(recon.len(), 1);
        assert!((recon[0] - 3.5).abs() < 1e-12);
    }

    #[test]
    fn test_dft_dc_isolation() {
        let n = 128;
        let dc = vec![1.0; n];
        let spec = Dft::forward_real(&dc).unwrap();
        assert!((spec[0].re - (n as f64)).abs() < 1e-10);
        assert!(spec[0].im.abs() < 1e-10);

        for (k, bin) in spec.iter().enumerate().skip(1) {
            assert!(bin.norm() < 1e-10, "Bin {} had non-zero energy", k);
        }
    }

    #[test]
    fn test_dft_prime_lengths_roundtrip() {
        for &n in &[3, 7, 11, 13] {
            let x: Vec<f64> = (0..n).map(|v| (v + 1) as f64).collect();
            let spectrum = Dft::forward_real(&x).unwrap();
            let reconstructed = Dft::inverse_real(&spectrum).unwrap();

            for idx in 0..n {
                assert!(
                    (x[idx] - reconstructed[idx]).abs() < 1e-12,
                    "Mismatch at n={}, idx={}",
                    n,
                    idx
                );
            }
        }
    }

    #[test]
    fn test_dft_parseval_energy_identity() {
        for &n in &[16, 64, 127] {
            let x: Vec<f64> = (0..n).map(|v| (v as f64 * 0.1).sin()).collect();
            let spec = Dft::forward_real(&x).unwrap();

            let time_energy: f64 = x.iter().map(|&s| s * s).sum();
            let freq_energy: f64 = spec.iter().map(|c| c.norm_sq()).sum::<f64>() / (n as f64);
            let diff = (time_energy - freq_energy).abs();
            assert!(
                diff < 1e-10,
                "Parseval violation for N={}: diff={:.2e}",
                n,
                diff
            );
        }
    }

    #[test]
    fn test_dft_complex_roundtrip() {
        let input = vec![
            Complex::new(1.0, 2.0),
            Complex::new(-0.5, 1.5),
            Complex::new(3.0, -1.0),
            Complex::new(0.0, 4.0),
        ];

        let spec = Dft::forward_complex(&input).unwrap();
        let recon = Dft::inverse_complex(&spec).unwrap();

        for i in 0..input.len() {
            assert!((input[i].re - recon[i].re).abs() < 1e-12);
            assert!((input[i].im - recon[i].im).abs() < 1e-12);
        }
    }

    #[test]
    fn test_dft_nan_inf_rejection() {
        let nan_input = [1.0, f64::NAN, 3.0];
        assert!(matches!(
            Dft::forward_real(&nan_input),
            Err(DspError::NumericalInstability(_))
        ));

        let inf_input = [1.0, f64::INFINITY, 3.0];
        assert!(matches!(
            Dft::forward_real(&inf_input),
            Err(DspError::NumericalInstability(_))
        ));
    }

    #[test]
    fn test_dft_trait_polymorphism() {
        let dft = Dft::new();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let spec = dft.forward_real(&x).unwrap();
        let recon = dft.inverse_real(&spec).unwrap();
        for i in 0..4 {
            assert!((x[i] - recon[i]).abs() < 1e-12);
        }
    }
}
