//! Adversarial Stress & Correctness Challenge Suite for Calyx Spectral Engine (`Dft` & `Fft`).
//!
//! Evaluates:
//! 1. Exact mathematical equivalence between direct ground-truth $O(N^2)$ DFT and $O(N \log N)$ Radix-2 FFT.
//! 2. Roundtrip reconstruction precision $\|x - \text{IFFT}(\text{FFT}(x))\|_\infty$ on pathological signals
//!    (unit impulses at all offsets, DC signals, Nyquist signals, extreme amplitude scalings $10^{\pm 150}$, subnormals).
//! 3. Parseval's energy theorem error across random signals on arbitrary prime/odd/even lengths for DFT and powers-of-two for FFT.
//! 4. Rigorous boundary conditions: $N=1, 2, 4$, large $N=65536$, non-power-of-two rejection, empty slice handling.
//! 5. Robustness against NaN/Inf contamination across all forward, inverse, in-place, and trait interfaces.
//! 6. Fundamental Fourier identities: linearity, circular convolution, time-shift, frequency-shift, and Hermitian symmetry.
//! 7. Trait polymorphism and performance/memory stability under repeated execution.

#![allow(clippy::needless_range_loop)]

use std::f64::consts::PI;

use calyx_dsp::error::DspError;
use calyx_dsp::math::Complex;
use calyx_dsp::spectral::{Dft, Fft, FourierTransform};

const EPS_RECON: f64 = 1e-11;
const EPS_PARSEVAL: f64 = 1e-10;

/// Deterministic Linear Congruential PRNG (Knuth 64-bit parameters)
struct DeterministicPrng {
    state: u64,
}

impl DeterministicPrng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(0x14057B7EF767814F) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64_unit(&mut self) -> f64 {
        let val = (self.next_u64() >> 11) as f64;
        val / ((1u64 << 53) as f64)
    }

    fn next_f64_symmetric(&mut self) -> f64 {
        self.next_f64_unit() * 2.0 - 1.0
    }

    fn next_gaussian(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.next_f64_unit().max(1e-15);
        let u2 = self.next_f64_unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn max_complex_diff(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x.re - y.re).hypot(x.im - y.im))
        .fold(0.0, f64::max)
}

// ============================================================================
// Section 1: Exact Equivalence: DFT Ground Truth vs FFT
// ============================================================================

#[test]
fn challenge_dft_fft_exact_equivalence_all_power_of_two_sizes() {
    let sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
    let mut prng = DeterministicPrng::new(0xABCDEF0123456789);

    for &n in &sizes {
        let mut x_complex = Vec::with_capacity(n);
        let mut x_real = Vec::with_capacity(n);
        for _ in 0..n {
            let re = prng.next_f64_symmetric();
            let im = prng.next_f64_symmetric();
            x_complex.push(Complex::new(re, im));
            x_real.push(re);
        }

        let fft_plan = Fft::new(n).unwrap();

        // 1. Complex forward
        let dft_spec = Dft::forward_complex(&x_complex).unwrap();
        let fft_spec = fft_plan.fft(&x_complex).unwrap();
        let diff_fwd = max_complex_diff(&dft_spec, &fft_spec);
        assert!(
            diff_fwd < (n as f64) * 1e-13 + 1e-10,
            "Complex forward mismatch at N={}: diff={:.2e}",
            n,
            diff_fwd
        );

        // 2. Complex inverse
        let dft_recon = Dft::inverse_complex(&dft_spec).unwrap();
        let fft_recon = fft_plan.ifft(&fft_spec).unwrap();
        let diff_inv = max_complex_diff(&dft_recon, &fft_recon);
        assert!(
            diff_inv < (n as f64) * 1e-13 + 1e-10,
            "Complex inverse mismatch at N={}: diff={:.2e}",
            n,
            diff_inv
        );

        // 3. Real forward
        let dft_real_spec = Dft::forward_real(&x_real).unwrap();
        let fft_real_spec = fft_plan.fft_real(&x_real).unwrap();
        let diff_real_fwd = max_complex_diff(&dft_real_spec, &fft_real_spec);
        assert!(
            diff_real_fwd < (n as f64) * 1e-13 + 1e-10,
            "Real forward mismatch at N={}: diff={:.2e}",
            n,
            diff_real_fwd
        );

        // 4. Real inverse
        let dft_real_recon = Dft::inverse_real(&dft_real_spec).unwrap();
        let fft_real_recon = fft_plan.ifft_real(&fft_real_spec).unwrap();
        let diff_real_inv = max_abs_diff(&dft_real_recon, &fft_real_recon);
        assert!(
            diff_real_inv < (n as f64) * 1e-13 + 1e-10,
            "Real inverse mismatch at N={}: diff={:.2e}",
            n,
            diff_real_inv
        );
    }
}

// ============================================================================
// Section 2: Pathological Signals & Roundtrip Precision
// ============================================================================

#[test]
fn challenge_unit_impulse_all_offsets_roundtrip() {
    let n = 64;
    let fft = Fft::new(n).unwrap();

    // Test impulse at EVERY single offset k from 0 to N-1
    for offset in 0..n {
        let mut impulse = vec![0.0; n];
        impulse[offset] = 1.0;

        let spectrum = fft.fft_real(&impulse).unwrap();

        // Analytical property: |X[k]| == 1.0 for all frequency bins
        for (bin, c) in spectrum.iter().enumerate() {
            let mag = c.norm();
            assert!(
                (mag - 1.0).abs() < 1e-12,
                "Offset {}: bin {} magnitude expected 1.0, got {:.15}",
                offset,
                bin,
                mag
            );
            // Analytical phase: phase = -2*PI*k*offset / N
            let expected_angle = -2.0 * PI * (bin as f64) * (offset as f64) / (n as f64);
            let expected_c = Complex::from_polar(1.0, expected_angle);
            let c_diff = (c.re - expected_c.re).hypot(c.im - expected_c.im);
            assert!(
                c_diff < 1e-11,
                "Offset {}: bin {} phase mismatch: got {:?}, expected {:?}",
                offset,
                bin,
                c,
                expected_c
            );
        }

        // Exact roundtrip reconstruction
        let reconstructed = fft.ifft_real(&spectrum).unwrap();
        let err = max_abs_diff(&impulse, &reconstructed);
        assert!(
            err < 1e-14,
            "Offset {}: impulse roundtrip error {:.2e} exceeds tolerance",
            offset,
            err
        );
    }
}

#[test]
fn challenge_dc_signals_energy_isolation_and_scaling() {
    let test_sizes = [1, 2, 8, 32, 128, 512];
    let dc_values = [1.0, -100.0, 1e-6, 1e7, -PI];

    for &n in &test_sizes {
        let fft = Fft::new(n).unwrap();
        for &dc_val in &dc_values {
            let signal = vec![dc_val; n];
            let spec = fft.fft_real(&signal).unwrap();

            // DC bin must equal n * dc_val
            let expected_dc = (n as f64) * dc_val;
            assert!(
                (spec[0].re - expected_dc).abs() < (dc_val.abs().max(1.0)) * 1e-11,
                "DC bin 0 mismatch for N={}, dc={}: got {}, expected {}",
                n,
                dc_val,
                spec[0].re,
                expected_dc
            );
            assert!(
                spec[0].im.abs() < (dc_val.abs().max(1.0)) * 1e-11,
                "DC bin 0 imaginary non-zero: {}",
                spec[0].im
            );

            // All AC bins (k >= 1) must be strictly 0
            for k in 1..n {
                let ac_mag = spec[k].norm();
                assert!(
                    ac_mag < (dc_val.abs().max(1.0)) * 1e-10,
                    "AC bin {} had leakage for N={}, dc={}: mag={:.2e}",
                    k,
                    n,
                    dc_val,
                    ac_mag
                );
            }

            // Roundtrip
            let recon = fft.ifft_real(&spec).unwrap();
            let err = max_abs_diff(&signal, &recon);
            assert!(
                err < (dc_val.abs().max(1.0)) * 1e-12,
                "DC roundtrip error {:.2e} exceeds tolerance",
                err
            );
        }
    }
}

#[test]
fn challenge_nyquist_signals_energy_isolation() {
    let test_sizes = [2, 4, 16, 64, 256, 1024];
    let amplitudes = [1.0, -50.0, 0.001, 42.42];

    for &n in &test_sizes {
        let fft = Fft::new(n).unwrap();
        for &amp in &amplitudes {
            // Signal oscillating at Nyquist: x[n] = amp * (-1)^n
            let signal: Vec<f64> = (0..n)
                .map(|i| if i % 2 == 0 { amp } else { -amp })
                .collect();

            let spec = fft.fft_real(&signal).unwrap();
            let nyquist_bin = n / 2;

            // Nyquist bin must equal n * amp
            let expected_nyquist = (n as f64) * amp;
            assert!(
                (spec[nyquist_bin].re - expected_nyquist).abs() < (amp.abs().max(1.0)) * 1e-10,
                "Nyquist bin mismatch at N={}: got {}, expected {}",
                n,
                spec[nyquist_bin].re,
                expected_nyquist
            );
            assert!(
                spec[nyquist_bin].im.abs() < (amp.abs().max(1.0)) * 1e-10,
                "Nyquist bin had imaginary component: {}",
                spec[nyquist_bin].im
            );

            // All other bins must be strictly 0
            for k in 0..n {
                if k != nyquist_bin {
                    let mag = spec[k].norm();
                    assert!(
                        mag < (amp.abs().max(1.0)) * 1e-10,
                        "Non-Nyquist bin {} had leakage at N={}: mag={:.2e}",
                        k,
                        n,
                        mag
                    );
                }
            }

            // Roundtrip
            let recon = fft.ifft_real(&spec).unwrap();
            let err = max_abs_diff(&signal, &recon);
            assert!(
                err < (amp.abs().max(1.0)) * 1e-12,
                "Nyquist roundtrip error {:.2e}",
                err
            );
        }
    }
}

#[test]
fn challenge_extreme_amplitude_scaling_roundtrip() {
    let n = 128;
    let fft = Fft::new(n).unwrap();

    // Giant amplitude: 1e150
    let scale_huge = 1e150;
    let x_huge: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.1).sin() * scale_huge).collect();
    let spec_huge = fft.fft_real(&x_huge).unwrap();
    let recon_huge = fft.ifft_real(&spec_huge).unwrap();
    let rel_err_huge = max_abs_diff(&x_huge, &recon_huge) / scale_huge;
    assert!(
        rel_err_huge < 1e-14,
        "Extreme scale 1e150 relative roundtrip error: {:.2e}",
        rel_err_huge
    );

    // Tiny amplitude: 1e-150
    let scale_tiny = 1e-150;
    let x_tiny: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.1).sin() * scale_tiny).collect();
    let spec_tiny = fft.fft_real(&x_tiny).unwrap();
    let recon_tiny = fft.ifft_real(&spec_tiny).unwrap();
    let rel_err_tiny = max_abs_diff(&x_tiny, &recon_tiny) / scale_tiny;
    assert!(
        rel_err_tiny < 1e-14,
        "Extreme scale 1e-150 relative roundtrip error: {:.2e}",
        rel_err_tiny
    );

    // Subnormal amplitude: 1e-310 (IEEE-754 double subnormal range)
    let scale_subnormal = 1e-310;
    let x_sub: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { scale_subnormal } else { 0.0 }).collect();
    let spec_sub = fft.fft_real(&x_sub).unwrap();
    for c in &spec_sub {
        assert!(!c.re.is_nan() && !c.im.is_nan());
    }
    let recon_sub = fft.ifft_real(&spec_sub).unwrap();
    for &val in &recon_sub {
        assert!(!val.is_nan());
    }
}

#[test]
fn challenge_gaussian_and_uniform_noise_roundtrip() {
    let mut prng = DeterministicPrng::new(42);
    let test_sizes = [16, 64, 256, 1024, 4096];

    for &n in &test_sizes {
        let fft = Fft::new(n).unwrap();

        // 1. Uniform noise
        let uniform: Vec<f64> = (0..n).map(|_| prng.next_f64_symmetric()).collect();
        let spec_u = fft.fft_real(&uniform).unwrap();
        let recon_u = fft.ifft_real(&spec_u).unwrap();
        let err_u = max_abs_diff(&uniform, &recon_u);
        assert!(
            err_u < EPS_RECON,
            "Uniform noise roundtrip error at N={}: {:.2e}",
            n,
            err_u
        );

        // 2. Gaussian noise
        let gaussian: Vec<f64> = (0..n).map(|_| prng.next_gaussian()).collect();
        let spec_g = fft.fft_real(&gaussian).unwrap();
        let recon_g = fft.ifft_real(&spec_g).unwrap();
        let err_g = max_abs_diff(&gaussian, &recon_g);
        assert!(
            err_g < EPS_RECON,
            "Gaussian noise roundtrip error at N={}: {:.2e}",
            n,
            err_g
        );
    }
}

// ============================================================================
// Section 3: Parseval's Energy Conservation Theorem
// ============================================================================

#[test]
fn challenge_dft_parseval_arbitrary_odd_even_prime_lengths() {
    let lengths = [1, 2, 3, 5, 7, 11, 13, 16, 17, 31, 32, 63, 64, 127, 128, 257];
    let mut prng = DeterministicPrng::new(0x1337C0DE);

    for &n in &lengths {
        for _ in 0..5 {
            let mut x_complex = Vec::with_capacity(n);
            let mut time_energy = 0.0;
            for _ in 0..n {
                let c = Complex::new(prng.next_gaussian(), prng.next_gaussian());
                time_energy += c.norm_sq();
                x_complex.push(c);
            }

            let spectrum = Dft::forward_complex(&x_complex).unwrap();
            let freq_energy: f64 = spectrum.iter().map(|c| c.norm_sq()).sum::<f64>() / (n as f64);

            let rel_diff = (time_energy - freq_energy).abs() / time_energy;
            assert!(
                rel_diff < EPS_PARSEVAL,
                "DFT Parseval violated at N={}: rel_diff={:.2e}",
                n,
                rel_diff
            );
        }
    }
}

#[test]
fn challenge_fft_parseval_all_power_of_two_sizes() {
    let sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
    let mut prng = DeterministicPrng::new(0xFEEDFACE);

    for &n in &sizes {
        let fft = Fft::new(n).unwrap();
        for _ in 0..5 {
            let mut x_complex = Vec::with_capacity(n);
            let mut time_energy = 0.0;
            for _ in 0..n {
                let c = Complex::new(prng.next_f64_symmetric(), prng.next_f64_symmetric());
                time_energy += c.norm_sq();
                x_complex.push(c);
            }

            let spectrum = fft.fft(&x_complex).unwrap();
            let freq_energy: f64 = spectrum.iter().map(|c| c.norm_sq()).sum::<f64>() / (n as f64);

            let rel_diff = (time_energy - freq_energy).abs() / time_energy;
            assert!(
                rel_diff < EPS_PARSEVAL,
                "FFT Parseval violated at N={}: rel_diff={:.2e}",
                n,
                rel_diff
            );
        }
    }
}

// ============================================================================
// Section 4: Edge Cases & Boundary Conditions
// ============================================================================

#[test]
fn challenge_n_equals_one_exact_semantics() {
    // 1. Real DFT
    let dft_spec = Dft::forward_real(&[7.5]).unwrap();
    assert_eq!(dft_spec.len(), 1);
    assert_eq!(dft_spec[0], Complex::new(7.5, 0.0));
    let dft_recon = Dft::inverse_real(&dft_spec).unwrap();
    assert_eq!(dft_recon, vec![7.5]);

    // 2. Complex DFT
    let c = Complex::new(-3.0, 4.0);
    let dft_c_spec = Dft::forward_complex(&[c]).unwrap();
    assert_eq!(dft_c_spec[0], c);
    let dft_c_recon = Dft::inverse_complex(&dft_c_spec).unwrap();
    assert_eq!(dft_c_recon[0], c);

    // 3. FFT N=1
    let fft1 = Fft::new(1).unwrap();
    assert_eq!(fft1.size(), 1);
    let fft_spec = fft1.fft_real(&[7.5]).unwrap();
    assert_eq!(fft_spec[0], Complex::new(7.5, 0.0));
    let fft_recon = fft1.ifft_real(&fft_spec).unwrap();
    assert_eq!(fft_recon, vec![7.5]);

    let fft_c_spec = fft1.fft(&[c]).unwrap();
    assert_eq!(fft_c_spec[0], c);
    let fft_c_recon = fft1.ifft(&fft_c_spec).unwrap();
    assert_eq!(fft_c_recon[0], c);

    // 4. In-place static radix2 N=1
    let mut buf = [c];
    Fft::forward_complex_radix2(&mut buf).unwrap();
    assert_eq!(buf[0], c);
    Fft::inverse_complex_radix2(&mut buf).unwrap();
    assert_eq!(buf[0], c);
}

#[test]
fn challenge_n_equals_two_exact_butterflies() {
    let a = Complex::new(3.0, 1.0);
    let b = Complex::new(5.0, -2.0);
    let input = [a, b];

    // Analytical forward: [a + b, a - b]
    let exp_fwd = [a + b, a - b];

    let fft = Fft::new(2).unwrap();
    let spec = fft.fft(&input).unwrap();
    assert_eq!(spec[0], exp_fwd[0]);
    assert_eq!(spec[1], exp_fwd[1]);

    let recon = fft.ifft(&spec).unwrap();
    assert!((recon[0].re - a.re).abs() < 1e-14);
    assert!((recon[0].im - a.im).abs() < 1e-14);
    assert!((recon[1].re - b.re).abs() < 1e-14);
    assert!((recon[1].im - b.im).abs() < 1e-14);
}

#[test]
fn challenge_n_equals_four_analytical_butterflies() {
    let a = 1.0;
    let b = 2.0;
    let c = 3.0;
    let d = 4.0;
    let x = [a, b, c, d];

    // Theoretical 4-point DFT:
    // X[0] = a + b + c + d = 10
    // X[1] = (a - c) - i*(b - d) = -2 + 2i
    // X[2] = a - b + c - d = -2
    // X[3] = (a - c) + i*(b - d) = -2 - 2i
    let fft = Fft::new(4).unwrap();
    let spec = fft.fft_real(&x).unwrap();

    assert!((spec[0].re - 10.0).abs() < 1e-14);
    assert!(spec[0].im.abs() < 1e-14);
    assert!((spec[1].re - (-2.0)).abs() < 1e-14);
    assert!((spec[1].im - 2.0).abs() < 1e-14);
    assert!((spec[2].re - (-2.0)).abs() < 1e-14);
    assert!(spec[2].im.abs() < 1e-14);
    assert!((spec[3].re - (-2.0)).abs() < 1e-14);
    assert!((spec[3].im - (-2.0)).abs() < 1e-14);

    let recon = fft.ifft_real(&spec).unwrap();
    for i in 0..4 {
        assert!((recon[i] - x[i]).abs() < 1e-14);
    }
}

#[test]
fn challenge_large_size_65536_stress_and_precision() {
    let n = 65536;
    let fft = Fft::new(n).unwrap();
    assert_eq!(fft.size(), n);

    // Multi-frequency chirping waveform
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f64) / (n as f64);
        let val = (2.0 * PI * 100.0 * t * t).sin() + 0.5 * (2.0 * PI * 1000.0 * t).cos();
        x.push(val);
    }

    let time_energy: f64 = x.iter().map(|&v| v * v).sum();

    let spec = fft.fft_real(&x).unwrap();
    let freq_energy: f64 = spec.iter().map(|c| c.norm_sq()).sum::<f64>() / (n as f64);
    let parseval_err = (time_energy - freq_energy).abs() / time_energy;
    assert!(
        parseval_err < 1e-11,
        "N=65536 Parseval error: {:.2e}",
        parseval_err
    );

    let recon = fft.ifft_real(&spec).unwrap();
    let max_err = max_abs_diff(&x, &recon);
    assert!(
        max_err < 1e-11,
        "N=65536 roundtrip max error: {:.2e}",
        max_err
    );
}

#[test]
fn challenge_exhaustive_non_power_of_two_rejection() {
    // Non-power of two sizes up to 135
    for size in 0usize..=135usize {
        if size == 0 {
            assert_eq!(Fft::new(0).err(), Some(DspError::EmptyInput));
            let mut buf = vec![];
            assert_eq!(
                Fft::forward_complex_radix2(&mut buf).err(),
                Some(DspError::EmptyInput)
            );
        } else if !size.is_power_of_two() {
            assert_eq!(
                Fft::new(size).err(),
                Some(DspError::NonPowerOfTwo { size })
            );
            let mut buf = vec![Complex::zero(); size];
            assert_eq!(
                Fft::forward_complex_radix2(&mut buf).err(),
                Some(DspError::NonPowerOfTwo { size })
            );
            assert_eq!(
                Fft::inverse_complex_radix2(&mut buf).err(),
                Some(DspError::NonPowerOfTwo { size })
            );
        } else {
            assert!(Fft::new(size).is_ok());
        }
    }

    // High non-powers of two
    for &size in &[255, 257, 500, 1000, 30000, 65535, 65537] {
        assert_eq!(
            Fft::new(size).err(),
            Some(DspError::NonPowerOfTwo { size })
        );
    }
}

#[test]
fn challenge_empty_slice_strict_rejection() {
    // DFT
    assert_eq!(Dft::forward_real(&[]).err(), Some(DspError::EmptyInput));
    assert_eq!(Dft::forward_complex(&[]).err(), Some(DspError::EmptyInput));
    assert_eq!(Dft::inverse_real(&[]).err(), Some(DspError::EmptyInput));
    assert_eq!(Dft::inverse_complex(&[]).err(), Some(DspError::EmptyInput));
    assert_eq!(
        Dft::forward_complex_inplace(&mut []).err(),
        Some(DspError::EmptyInput)
    );
    assert_eq!(
        Dft::inverse_complex_inplace(&mut []).err(),
        Some(DspError::EmptyInput)
    );

    // FFT static
    let mut empty_complex: [Complex<f64>; 0] = [];
    assert_eq!(
        Fft::forward_complex_radix2(&mut empty_complex).err(),
        Some(DspError::EmptyInput)
    );
    assert_eq!(
        Fft::inverse_complex_radix2(&mut empty_complex).err(),
        Some(DspError::EmptyInput)
    );

    // FFT instance dimension mismatch on empty input
    let fft = Fft::new(8).unwrap();
    assert_eq!(
        fft.fft(&[]).err(),
        Some(DspError::BufferLengthMismatch {
            expected: 8,
            actual: 0
        })
    );
    assert_eq!(
        fft.ifft(&[]).err(),
        Some(DspError::BufferLengthMismatch {
            expected: 8,
            actual: 0
        })
    );
    assert_eq!(
        fft.fft_real(&[]).err(),
        Some(DspError::BufferLengthMismatch {
            expected: 8,
            actual: 0
        })
    );
    assert_eq!(
        fft.ifft_real(&[]).err(),
        Some(DspError::BufferLengthMismatch {
            expected: 8,
            actual: 0
        })
    );
}

// ============================================================================
// Section 5: Robustness Against NaN / Inf Contamination
// ============================================================================

#[test]
fn challenge_dft_nan_inf_rejection_all_interfaces() {
    let mut nan_real = vec![1.0, 2.0, 3.0, 4.0];
    nan_real[1] = f64::NAN;
    assert!(matches!(
        Dft::forward_real(&nan_real),
        Err(DspError::NumericalInstability(_))
    ));

    let mut inf_real = vec![1.0, 2.0, 3.0, 4.0];
    inf_real[3] = f64::INFINITY;
    assert!(matches!(
        Dft::forward_real(&inf_real),
        Err(DspError::NumericalInstability(_))
    ));

    let mut neg_inf_real = vec![1.0, 2.0, 3.0, 4.0];
    neg_inf_real[0] = f64::NEG_INFINITY;
    assert!(matches!(
        Dft::forward_real(&neg_inf_real),
        Err(DspError::NumericalInstability(_))
    ));

    // Complex forward with NaN
    let mut nan_c = vec![Complex::zero(); 4];
    nan_c[2].im = f64::NAN;
    assert!(matches!(
        Dft::forward_complex(&nan_c),
        Err(DspError::NumericalInstability(_))
    ));

    // Inverse real and complex with Inf
    let mut inf_c = vec![Complex::zero(); 4];
    inf_c[0].re = f64::INFINITY;
    assert!(matches!(
        Dft::inverse_real(&inf_c),
        Err(DspError::NumericalInstability(_))
    ));
    assert!(matches!(
        Dft::inverse_complex(&inf_c),
        Err(DspError::NumericalInstability(_))
    ));
}

#[test]
fn challenge_fft_nan_inf_rejection_all_interfaces() {
    let test_sizes = [1, 2, 4, 16];

    for &n in &test_sizes {
        let fft = Fft::new(n).unwrap();

        // 1. Static in-place forward with NaN
        let mut buf_nan = vec![Complex::zero(); n];
        buf_nan[0] = Complex::new(f64::NAN, 0.0);
        assert!(matches!(
            Fft::forward_complex_radix2(&mut buf_nan),
            Err(DspError::NumericalInstability(_))
        ));

        // 2. Static in-place inverse with Inf
        let mut buf_inf = vec![Complex::zero(); n];
        buf_inf[n - 1] = Complex::new(0.0, f64::INFINITY);
        assert!(matches!(
            Fft::inverse_complex_radix2(&mut buf_inf),
            Err(DspError::NumericalInstability(_))
        ));

        // 3. FFT instance method with NaN
        let res_fft = fft.fft(&buf_nan);
        assert!(matches!(res_fft, Err(DspError::NumericalInstability(_))));

        let res_ifft = fft.ifft(&buf_inf);
        assert!(matches!(res_ifft, Err(DspError::NumericalInstability(_))));

        // 4. Real forward with NaN
        let mut real_nan = vec![0.0; n];
        real_nan[0] = f64::NAN;
        assert!(matches!(
            fft.fft_real(&real_nan),
            Err(DspError::NumericalInstability(_))
        ));

        // 5. In-place execute with -Inf
        let mut buf_neginf = vec![Complex::zero(); n];
        buf_neginf[n / 2] = Complex::new(f64::NEG_INFINITY, 0.0);
        assert!(matches!(
            fft.execute(&mut buf_neginf, false),
            Err(DspError::NumericalInstability(_))
        ));
    }
}

// ============================================================================
// Section 6: Fundamental Fourier Identities & Symmetries
// ============================================================================

#[test]
fn challenge_linearity_identity() {
    let n = 64;
    let fft = Fft::new(n).unwrap();
    let mut prng = DeterministicPrng::new(0xCAFEBABE);

    let alpha = 2.5;
    let beta = -1.75;

    let x: Vec<f64> = (0..n).map(|_| prng.next_gaussian()).collect();
    let y: Vec<f64> = (0..n).map(|_| prng.next_gaussian()).collect();

    let linear_combo: Vec<f64> = (0..n).map(|i| alpha * x[i] + beta * y[i]).collect();

    let x_spec = fft.fft_real(&x).unwrap();
    let y_spec = fft.fft_real(&y).unwrap();
    let combo_spec = fft.fft_real(&linear_combo).unwrap();

    // FFT(alpha*x + beta*y) == alpha*FFT(x) + beta*FFT(y)
    for i in 0..n {
        let expected = Complex::new(
            alpha * x_spec[i].re + beta * y_spec[i].re,
            alpha * x_spec[i].im + beta * y_spec[i].im,
        );
        let diff = (combo_spec[i].re - expected.re).hypot(combo_spec[i].im - expected.im);
        assert!(
            diff < 1e-12,
            "Linearity violated at bin {}: diff={:.2e}",
            i,
            diff
        );
    }
}

#[test]
fn challenge_circular_convolution_theorem() {
    let n = 32;
    let fft = Fft::new(n).unwrap();
    let mut prng = DeterministicPrng::new(0x77777777);

    let x: Vec<f64> = (0..n).map(|_| prng.next_f64_symmetric()).collect();
    let h: Vec<f64> = (0..n).map(|_| prng.next_f64_symmetric()).collect();

    // 1. Direct circular convolution in time domain: y[n] = sum_k x[k] * h[(n - k) mod N]
    let mut y_direct = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..n {
            let h_idx = (i + n - k) % n;
            sum += x[k] * h[h_idx];
        }
        y_direct[i] = sum;
    }

    // 2. Convolution via frequency domain multiplication: y_fft = IFFT(FFT(x) * FFT(h))
    let x_spec = fft.fft_real(&x).unwrap();
    let h_spec = fft.fft_real(&h).unwrap();
    let mut prod_spec = Vec::with_capacity(n);
    for i in 0..n {
        prod_spec.push(x_spec[i] * h_spec[i]);
    }
    let y_fft = fft.ifft_real(&prod_spec).unwrap();

    let diff = max_abs_diff(&y_direct, &y_fft);
    assert!(
        diff < 1e-11,
        "Circular convolution theorem error: {:.2e}",
        diff
    );
}

#[test]
fn challenge_hermitian_symmetry_for_real_signals() {
    let n = 128;
    let fft = Fft::new(n).unwrap();
    let mut prng = DeterministicPrng::new(0x12344321);

    let signal: Vec<f64> = (0..n).map(|_| prng.next_gaussian()).collect();
    let spec = fft.fft_real(&signal).unwrap();

    // Bin 0 must be purely real
    assert!(spec[0].im.abs() < 1e-12, "DC bin has imaginary part: {}", spec[0].im);

    // Nyquist bin N/2 must be purely real
    assert!(
        spec[n / 2].im.abs() < 1e-12,
        "Nyquist bin has imaginary part: {}",
        spec[n / 2].im
    );

    // Hermitian symmetry: X[N - k] == X[k]*
    for k in 1..(n / 2) {
        let mirrored = spec[n - k];
        let original = spec[k];

        assert!(
            (mirrored.re - original.re).abs() < 1e-12,
            "Real part symmetry failure at k={}: orig={}, mirrored={}",
            k,
            original.re,
            mirrored.re
        );
        assert!(
            (mirrored.im - (-original.im)).abs() < 1e-12,
            "Imaginary part antisymmetry failure at k={}: orig={}, mirrored={}",
            k,
            original.im,
            mirrored.im
        );
    }
}

#[test]
fn challenge_circular_time_shift_theorem() {
    let n = 64;
    let fft = Fft::new(n).unwrap();
    let mut prng = DeterministicPrng::new(0x99887766);

    let x: Vec<f64> = (0..n).map(|_| prng.next_gaussian()).collect();
    let x_spec = fft.fft_real(&x).unwrap();

    for shift in [1, 5, 17, 32, 63] {
        // Circularly shift x by `shift` samples: y[n] = x[(n - shift) mod n]
        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = x[(i + n - (shift % n)) % n];
        }

        let y_spec = fft.fft_real(&y).unwrap();

        // Spectrum of shifted signal: Y[k] = X[k] * e^{-i * 2*PI*k*shift / N}
        for k in 0..n {
            let angle = -2.0 * PI * (k as f64) * (shift as f64) / (n as f64);
            let twiddle = Complex::from_polar(1.0, angle);
            let expected = x_spec[k] * twiddle;

            let diff = (y_spec[k].re - expected.re).hypot(y_spec[k].im - expected.im);
            assert!(
                diff < 1e-11,
                "Time shift theorem violated at shift={}, k={}: diff={:.2e}",
                shift,
                k,
                diff
            );
        }
    }
}

// ============================================================================
// Section 7: Trait Polymorphism and In-Place Equivalence
// ============================================================================

#[test]
fn challenge_trait_polymorphism_and_inplace_equivalence() {
    let n = 128;
    let mut prng = DeterministicPrng::new(0x5555AAAA);

    let original: Vec<Complex<f64>> = (0..n)
        .map(|_| Complex::new(prng.next_gaussian(), prng.next_gaussian()))
        .collect();

    let fft = Fft::new(n).unwrap();

    // 1. Out-of-place via instance
    let out1 = fft.fft(&original).unwrap();

    // 2. In-place via instance execute
    let mut buf2 = original.clone();
    fft.execute(&mut buf2, false).unwrap();
    assert_eq!(out1, buf2);

    // 3. In-place via static method
    let mut buf3 = original.clone();
    Fft::forward_complex_radix2(&mut buf3).unwrap();
    assert_eq!(out1, buf3);

    // 4. Via dyn FourierTransform trait object
    let trait_obj: &dyn FourierTransform = &fft;
    let out4 = trait_obj.forward_complex(&original).unwrap();
    assert_eq!(out1, out4);

    let mut buf5 = original.clone();
    trait_obj.forward_complex_inplace(&mut buf5).unwrap();
    assert_eq!(out1, buf5);

    // 5. Inverse trait object
    let recon_trait = trait_obj.inverse_complex(&out4).unwrap();
    for i in 0..n {
        assert!((recon_trait[i].re - original[i].re).abs() < 1e-12);
        assert!((recon_trait[i].im - original[i].im).abs() < 1e-12);
    }
}

// ============================================================================
// Section 8: Memory Stability & Stress Harness
// ============================================================================

#[test]
fn challenge_repeated_execution_memory_stability() {
    let n = 1024;
    let fft = Fft::new(n).unwrap();
    let mut prng = DeterministicPrng::new(0xDEADC0DE);

    let mut buf: Vec<Complex<f64>> = (0..n)
        .map(|_| Complex::new(prng.next_f64_symmetric(), prng.next_f64_symmetric()))
        .collect();
    let original = buf.clone();

    // 1,000 back-to-back forward and inverse transforms
    for _ in 0..1000 {
        fft.execute(&mut buf, false).unwrap();
        fft.execute(&mut buf, true).unwrap();
    }

    // After 1,000 roundtrips, numerical drift should still be well-bounded
    let err = max_complex_diff(&original, &buf);
    assert!(
        err < 1e-9,
        "Drift after 1,000 FFT/IFFT roundtrips: {:.2e}",
        err
    );
}

// ============================================================================
// Section 9: Analytical Sine & Cosine Phase and Peak Verification
// ============================================================================

#[test]
fn challenge_pure_sinusoid_analytical_phase_and_magnitude() {
    let n = 256;
    let fft = Fft::new(n).unwrap();
    let k0 = 7; // Frequency bin 7

    // 1. Pure cosine: x[n] = cos(2*pi*k0*n / N)
    // Spectrum should have X[k0] = N/2 + 0i (phase 0)
    let cosine: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * (k0 as f64) * (i as f64) / (n as f64)).cos())
        .collect();
    let cos_spec = fft.fft_real(&cosine).unwrap();

    let expected_cos_peak = (n as f64) / 2.0;
    assert!(
        (cos_spec[k0].re - expected_cos_peak).abs() < 1e-11,
        "Cosine peak real part mismatch: got {}, expected {}",
        cos_spec[k0].re,
        expected_cos_peak
    );
    assert!(
        cos_spec[k0].im.abs() < 1e-11,
        "Cosine peak imaginary part non-zero: {}",
        cos_spec[k0].im
    );
    assert!(
        cos_spec[k0].arg().abs() < 1e-11,
        "Cosine phase must be 0, got {}",
        cos_spec[k0].arg()
    );

    // 2. Pure sine: x[n] = sin(2*pi*k0*n / N)
    // Spectrum should have X[k0] = 0 - i*(N/2) (phase -pi/2)
    let sine: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * (k0 as f64) * (i as f64) / (n as f64)).sin())
        .collect();
    let sin_spec = fft.fft_real(&sine).unwrap();

    assert!(
        sin_spec[k0].re.abs() < 1e-11,
        "Sine peak real part non-zero: {}",
        sin_spec[k0].re
    );
    assert!(
        (sin_spec[k0].im - (-expected_cos_peak)).abs() < 1e-11,
        "Sine peak imag part mismatch: got {}, expected {}",
        sin_spec[k0].im,
        -expected_cos_peak
    );
    let expected_phase = -PI / 2.0;
    assert!(
        (sin_spec[k0].arg() - expected_phase).abs() < 1e-11,
        "Sine phase must be -pi/2, got {}",
        sin_spec[k0].arg()
    );
}

// ============================================================================
// Section 10: High-Dynamic-Range Exponential Decay Signals
// ============================================================================

#[test]
fn challenge_exponential_decay_transient_roundtrip() {
    let n = 512;
    let fft = Fft::new(n).unwrap();

    // Signal with steep exponential decay spanning 100 dB of dynamic range
    // x[n] = exp(-n / 20) * cos(2*pi*5*n / N)
    let signal: Vec<f64> = (0..n)
        .map(|i| (-(i as f64) / 20.0).exp() * (2.0 * PI * 5.0 * (i as f64) / (n as f64)).cos())
        .collect();

    let spec = fft.fft_real(&signal).unwrap();
    let recon = fft.ifft_real(&spec).unwrap();

    let err = max_abs_diff(&signal, &recon);
    assert!(
        err < 1e-13,
        "Exponential decay transient roundtrip error: {:.2e}",
        err
    );

    // Parseval check on transient
    let time_energy: f64 = signal.iter().map(|&v| v * v).sum();
    let freq_energy: f64 = spec.iter().map(|c| c.norm_sq()).sum::<f64>() / (n as f64);
    let parseval_err = (time_energy - freq_energy).abs() / time_energy;
    assert!(
        parseval_err < 1e-11,
        "Parseval error on exponential decay: {:.2e}",
        parseval_err
    );
}

// ============================================================================
// Section 11: N=65536 Pathological Delta Impulses at Boundaries
// ============================================================================

#[test]
fn challenge_large_n_65536_impulses_at_boundaries() {
    let n = 65536;
    let fft = Fft::new(n).unwrap();

    let offsets_to_test = [0, 1, n / 2, n - 1];

    for &offset in &offsets_to_test {
        let mut impulse = vec![0.0; n];
        impulse[offset] = 1.0;

        let spec = fft.fft_real(&impulse).unwrap();

        // Spectrum of impulse must have magnitude 1.0 everywhere
        for (bin, c) in spec.iter().enumerate() {
            let mag = c.norm();
            assert!(
                (mag - 1.0).abs() < 1e-10,
                "Offset {}: bin {} magnitude error at N=65536: got {}",
                offset,
                bin,
                mag
            );
        }

        let recon = fft.ifft_real(&spec).unwrap();
        let err = max_abs_diff(&impulse, &recon);
        assert!(
            err < 1e-12,
            "Offset {}: N=65536 impulse roundtrip error: {:.2e}",
            offset,
            err
        );
    }
}

// ============================================================================
// Section 12: Thread-Safety (Send + Sync) and Concurrent Execution
// ============================================================================

#[test]
fn challenge_thread_safety_and_concurrent_execution() {
    use std::sync::Arc;
    use std::thread;

    // Static assertion of Send + Sync
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Dft>();
    assert_send_sync::<Fft>();

    let n = 512;
    let fft = Arc::new(Fft::new(n).unwrap());
    let mut handles = Vec::new();

    for t in 0..8 {
        let fft_clone = Arc::clone(&fft);
        handles.push(thread::spawn(move || {
            let mut prng = DeterministicPrng::new(t as u64 + 100);
            for _ in 0..50 {
                let x: Vec<f64> = (0..n).map(|_| prng.next_gaussian()).collect();
                let spec = fft_clone.fft_real(&x).unwrap();
                let recon = fft_clone.ifft_real(&spec).unwrap();
                let err = max_abs_diff(&x, &recon);
                assert!(err < 1e-11);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

// ============================================================================
// Section 13: DFT Pathological Signals on Arbitrary and Prime Lengths
// ============================================================================

#[test]
fn challenge_dft_pathological_signals_arbitrary_lengths() {
    let prime_lengths = [3, 7, 13, 29, 61];

    for &n in &prime_lengths {
        // 1. Impulse at all offsets for prime N
        for offset in 0..n {
            let mut impulse = vec![0.0; n];
            impulse[offset] = 1.0;

            let spec = Dft::forward_real(&impulse).unwrap();
            for c in &spec {
                assert!((c.norm() - 1.0).abs() < 1e-12);
            }

            let recon = Dft::inverse_real(&spec).unwrap();
            assert!(max_abs_diff(&impulse, &recon) < 1e-12);
        }

        // 2. DC on prime length
        let dc = vec![4.2; n];
        let spec_dc = Dft::forward_real(&dc).unwrap();
        assert!((spec_dc[0].re - (n as f64) * 4.2).abs() < 1e-11);
        for k in 1..n {
            assert!(spec_dc[k].norm() < 1e-11);
        }
        let recon_dc = Dft::inverse_real(&spec_dc).unwrap();
        assert!(max_abs_diff(&dc, &recon_dc) < 1e-12);
    }
}

