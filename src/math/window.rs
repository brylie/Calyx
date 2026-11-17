//! Standard analytical window functions and Constant Overlap-Add (COLA) verification.
//!
//! Complies strictly with Demo Integrity Mode: zero external dependencies,
//! 100% pure standard library, safe Rust.

use std::f64::consts::PI;
use crate::error::{DspError, DspResult};

/// Supported analytical window function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WindowType {
    /// Hann (Hanning) window: $w\[n\] = 0.5 - 0.5 \cos(2\pi n / N)$.
    Hann,
    /// Hamming window: $w\[n\] = 0.54 - 0.46 \cos(2\pi n / N)$.
    Hamming,
    /// Blackman window (exact 3-term): $w\[n\] = 0.42 - 0.50 \cos(2\pi n / N) + 0.08 \cos(4\pi n / N)$.
    Blackman,
    /// Bartlett (triangular) window: $w\[n\] = 1.0 - |2n - N| / N$.
    Bartlett,
    /// Uniform rectangular window: $w\[n\] = 1.0$.
    Rectangular,
}

/// Generate window coefficients for a specified window type and length.
///
/// Produces periodic (DFT-even) windows suitable for STFT sliding-block analysis
/// and Overlap-Add (OLA) reconstruction that strictly satisfy Constant Overlap-Add (COLA).
///
/// # Errors
/// - Returns `DspError::EmptyInput` if `length == 0`.
/// - Returns `DspError::InvalidParameter` if `length < 2`.
pub fn generate_window(window_type: WindowType, length: usize) -> DspResult<Vec<f64>> {
    if length == 0 {
        return Err(DspError::EmptyInput);
    }
    if length < 2 {
        return Err(DspError::invalid_parameter(
            "length",
            format!("Window length must be at least 2, got {}", length),
        ));
    }

    let n_f64 = length as f64;
    let mut window = Vec::with_capacity(length);

    match window_type {
        WindowType::Rectangular => {
            window.resize(length, 1.0);
        }
        WindowType::Hann => {
            for n in 0..length {
                let phi = 2.0 * PI * (n as f64) / n_f64;
                window.push(0.5 * (1.0 - phi.cos()));
            }
        }
        WindowType::Hamming => {
            for n in 0..length {
                let phi = 2.0 * PI * (n as f64) / n_f64;
                window.push(0.54 - 0.46 * phi.cos());
            }
        }
        WindowType::Blackman => {
            for n in 0..length {
                let phi = 2.0 * PI * (n as f64) / n_f64;
                let val = 0.42 - 0.50 * phi.cos() + 0.08 * (2.0 * phi).cos();
                window.push(val);
            }
        }
        WindowType::Bartlett => {
            for n in 0..length {
                let val = 1.0 - (2.0 * (n as f64) - n_f64).abs() / n_f64;
                window.push(val);
            }
        }
    }

    Ok(window)
}

/// Generate symmetric window coefficients (designed for linear-phase FIR filter design).
///
/// Divides by $N - 1$, guaranteeing exact mirror symmetry around $(N-1)/2$ and zero endpoints.
///
/// # Errors
/// - Returns `DspError::EmptyInput` if `length == 0`.
/// - Returns `DspError::InvalidParameter` if `length < 2`.
pub fn generate_window_symmetric(window_type: WindowType, length: usize) -> DspResult<Vec<f64>> {
    if length == 0 {
        return Err(DspError::EmptyInput);
    }
    if length < 2 {
        return Err(DspError::invalid_parameter(
            "length",
            format!("Window length must be at least 2, got {}", length),
        ));
    }

    let denom = (length - 1) as f64;
    let mut window = Vec::with_capacity(length);

    match window_type {
        WindowType::Rectangular => {
            window.resize(length, 1.0);
        }
        WindowType::Hann => {
            for n in 0..length {
                let phi = 2.0 * PI * (n as f64) / denom;
                window.push(0.5 * (1.0 - phi.cos()));
            }
        }
        WindowType::Hamming => {
            for n in 0..length {
                let phi = 2.0 * PI * (n as f64) / denom;
                window.push(0.54 - 0.46 * phi.cos());
            }
        }
        WindowType::Blackman => {
            for n in 0..length {
                let phi = 2.0 * PI * (n as f64) / denom;
                let val = 0.42 - 0.50 * phi.cos() + 0.08 * (2.0 * phi).cos();
                window.push(val);
            }
        }
        WindowType::Bartlett => {
            for n in 0..length {
                let val = 1.0 - (2.0 * (n as f64) - denom).abs() / denom;
                window.push(val);
            }
        }
    }

    Ok(window)
}

/// Verify whether a window satisfies the Constant Overlap-Add (COLA) condition
/// for a specified hop size within an absolute tolerance $\epsilon$.
///
/// Evaluates:
/// $$S\[p\] = \sum_{k} w\[p + k \cdot R\], \quad \forall p \in \{0, 1, \dots, R - 1\}$$
/// Returns `true` if $\max(S) - \min(S) \le \text{tolerance}$ and $\min(S) > 0.0$.
///
/// Returns `false` on invalid input (empty window, non-finite samples, hop size of 0,
/// hop size exceeding length, negative or non-finite tolerance, or degenerate non-positive sum).
pub fn verify_cola(window: &[f64], hop_size: usize, tolerance: f64) -> bool {
    if window.is_empty()
        || hop_size == 0
        || hop_size > window.len()
        || !tolerance.is_finite()
        || tolerance < 0.0
    {
        return false;
    }

    if window.iter().any(|&x| !x.is_finite()) {
        return false;
    }

    let mut colas = vec![0.0; hop_size];
    for (n, &val) in window.iter().enumerate() {
        colas[n % hop_size] += val;
    }

    let min_val = colas.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = colas.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if !min_val.is_finite() || !max_val.is_finite() || min_val <= 0.0 {
        return false;
    }

    (max_val - min_val) <= tolerance
}

/// Compute the coherent gain (DC scaling factor) of a window:
/// $$G_c = \frac{1}{N} \sum_{n=0}^{N-1} w\[n\]$$
pub fn coherent_gain(window: &[f64]) -> f64 {
    if window.is_empty() {
        return 0.0;
    }
    window.iter().sum::<f64>() / (window.len() as f64)
}

/// Compute the energy gain (power scaling factor) of a window:
/// $$G_e = \frac{1}{N} \sum_{n=0}^{N-1} w^2\[n\]$$
pub fn energy_gain(window: &[f64]) -> f64 {
    if window.is_empty() {
        return 0.0;
    }
    window.iter().map(|&x| x * x).sum::<f64>() / (window.len() as f64)
}

/// Compute the Equivalent Noise Bandwidth (ENBW) in DFT bins:
/// $$\text{ENBW} = \frac{G_e}{G_c^2}$$
pub fn equivalent_noise_bandwidth(window: &[f64]) -> f64 {
    let gc = coherent_gain(window);
    if gc == 0.0 {
        return 0.0;
    }
    let ge = energy_gain(window);
    ge / (gc * gc)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-12;

    #[test]
    fn test_invalid_lengths() {
        assert_eq!(generate_window(WindowType::Hann, 0), Err(DspError::EmptyInput));
        match generate_window(WindowType::Hann, 1) {
            Err(DspError::InvalidParameter { name, .. }) => assert_eq!(name, "length"),
            other => panic!("Expected InvalidParameter, got {:?}", other),
        }

        assert_eq!(generate_window_symmetric(WindowType::Hann, 0), Err(DspError::EmptyInput));
        match generate_window_symmetric(WindowType::Hann, 1) {
            Err(DspError::InvalidParameter { name, .. }) => assert_eq!(name, "length"),
            other => panic!("Expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn test_rectangular_window() {
        let w = generate_window(WindowType::Rectangular, 64).unwrap();
        assert_eq!(w.len(), 64);
        assert!(w.iter().all(|&x| (x - 1.0).abs() < EPSILON));
        assert!(verify_cola(&w, 32, 1e-5));
        assert!(verify_cola(&w, 64, 1e-5));
        assert!((coherent_gain(&w) - 1.0).abs() < EPSILON);
        assert!((energy_gain(&w) - 1.0).abs() < EPSILON);
        assert!((equivalent_noise_bandwidth(&w) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_hann_cola_invariant() {
        let n = 1024;
        let w = generate_window(WindowType::Hann, n).unwrap();
        assert_eq!(w.len(), n);

        // Project requirement: Hann at R = N/2 sums to exactly 1.0
        assert!(verify_cola(&w, n / 2, 1e-5));
        assert!(verify_cola(&w, n / 4, 1e-5));
        assert!(verify_cola(&w, n / 8, 1e-5));

        // Non-integer ratio fails COLA
        assert!(!verify_cola(&w, 300, 1e-5));

        // Gains
        assert!((coherent_gain(&w) - 0.50).abs() < 1e-4);
        assert!((energy_gain(&w) - 0.375).abs() < 1e-4);
        assert!((equivalent_noise_bandwidth(&w) - 1.50).abs() < 1e-4);
    }

    #[test]
    fn test_hamming_cola() {
        let n = 512;
        let w = generate_window(WindowType::Hamming, n).unwrap();
        assert!(verify_cola(&w, n / 2, 1e-5));
        assert!(verify_cola(&w, n / 4, 1e-5));
        assert!((coherent_gain(&w) - 0.54).abs() < 1e-4);
        assert!((energy_gain(&w) - 0.3974).abs() < 1e-4);
    }

    #[test]
    fn test_blackman_cola() {
        let n = 512;
        let w = generate_window(WindowType::Blackman, n).unwrap();
        // Blackman requires R <= N/3 due to second harmonic; fails at R = N/2
        assert!(!verify_cola(&w, n / 2, 1e-5));
        // Passes at R = N/4
        assert!(verify_cola(&w, n / 4, 1e-5));
        assert!((coherent_gain(&w) - 0.42).abs() < 1e-4);
    }

    #[test]
    fn test_bartlett_cola() {
        let n = 256;
        let w = generate_window(WindowType::Bartlett, n).unwrap();
        assert!(verify_cola(&w, n / 2, 1e-5));
        assert!(verify_cola(&w, n / 4, 1e-5));
        assert!((coherent_gain(&w) - 0.50).abs() < 1e-4);
    }

    #[test]
    fn test_symmetric_windows() {
        let n = 64;
        let w_sym = generate_window_symmetric(WindowType::Hann, n).unwrap();
        assert_eq!(w_sym.len(), n);
        // Endpoints should be 0.0
        assert!(w_sym[0].abs() < EPSILON);
        assert!(w_sym[n - 1].abs() < EPSILON);
        // Symmetry check
        for i in 0..n / 2 {
            assert!((w_sym[i] - w_sym[n - 1 - i]).abs() < EPSILON);
        }

        let w_bart = generate_window_symmetric(WindowType::Bartlett, n).unwrap();
        assert!(w_bart[0].abs() < EPSILON);
        assert!(w_bart[n - 1].abs() < EPSILON);
        for i in 0..n / 2 {
            assert!((w_bart[i] - w_bart[n - 1 - i]).abs() < EPSILON);
        }
    }

    #[test]
    fn test_verify_cola_edge_cases() {
        let w = vec![1.0; 16];
        assert!(!verify_cola(&[], 4, 1e-5));
        assert!(!verify_cola(&w, 0, 1e-5));
        assert!(!verify_cola(&w, 32, 1e-5));
        assert!(!verify_cola(&w, 4, -1e-5));

        // Zero window
        let zero_w = vec![0.0; 16];
        assert!(!verify_cola(&zero_w, 4, 1e-5));

        // Coherent gain of empty window
        assert_eq!(coherent_gain(&[]), 0.0);
        assert_eq!(energy_gain(&[]), 0.0);
        assert_eq!(equivalent_noise_bandwidth(&[]), 0.0);
    }

    #[test]
    fn test_verify_cola_nan_inf_and_loose_tolerance() {
        // All-NaN slice must be rejected
        let nan_slice = vec![f64::NAN; 64];
        assert!(!verify_cola(&nan_slice, 16, 1e-5));

        // Partial-NaN slice must be rejected
        let mut partial_nan = vec![1.0; 64];
        partial_nan[5] = f64::NAN;
        assert!(!verify_cola(&partial_nan, 16, 1e-5));

        // +inf and -inf slices must be rejected
        let inf_slice = vec![f64::INFINITY; 64];
        assert!(!verify_cola(&inf_slice, 16, 1e-5));
        let neg_inf_slice = vec![f64::NEG_INFINITY; 64];
        assert!(!verify_cola(&neg_inf_slice, 16, 1e-5));

        // Partial inf slice must be rejected
        let mut partial_inf = vec![1.0; 64];
        partial_inf[10] = f64::INFINITY;
        assert!(!verify_cola(&partial_inf, 16, 1e-5));

        // Non-finite tolerance must be rejected
        let w = vec![1.0; 64];
        assert!(!verify_cola(&w, 16, f64::NAN));
        assert!(!verify_cola(&w, 16, f64::INFINITY));
        assert!(!verify_cola(&w, 16, f64::NEG_INFINITY));

        // Acceptance of loose tolerance (tolerance >= min_val)
        // Standard Hann window (periodic) has constant overlap sum S[p] = 1.0 at R = 128.
        let hann = generate_window(WindowType::Hann, 256).unwrap();
        assert!(verify_cola(&hann, 128, 1e-5));
        // Loose tolerance (1.5 >= min_val = 1.0) must NOT falsely reject
        assert!(verify_cola(&hann, 128, 1.5));
        assert!(verify_cola(&hann, 128, 1.0));
    }
}
