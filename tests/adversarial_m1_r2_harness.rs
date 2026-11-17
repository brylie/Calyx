//! tests/adversarial_m1_r2_harness.rs
//!
//! Empirical Adversarial Verification Suite for Milestone 1 Iteration 2 Hardening.
//! Targets: `src/math/complex.rs` and `src/math/window.rs`.

use calyx_dsp::math::complex::Complex;
use calyx_dsp::math::window::{
    coherent_gain, energy_gain, equivalent_noise_bandwidth, generate_window,
    generate_window_symmetric, verify_cola, WindowType,
};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

const EPSILON: f64 = 1e-12;

// =========================================================================
// 1. EMPIRICAL VERIFICATION: verify_cola Vulnerabilities Fixes
// =========================================================================

#[test]
fn test_verify_cola_all_nan_rejected() {
    let nan_window = vec![f64::NAN; 64];
    assert!(
        !verify_cola(&nan_window, 16, 1e-5),
        "verify_cola MUST return false for all-NaN buffer"
    );
}

#[test]
fn test_verify_cola_partial_nan_slices_rejected() {
    for pos in [0, 1, 15, 16, 31, 32, 47, 63] {
        let mut window = vec![1.0; 64];
        window[pos] = f64::NAN;
        assert!(
            !verify_cola(&window, 16, 1e-5),
            "verify_cola MUST return false when NaN is at index {}",
            pos
        );
    }

    // Alternating NaNs
    let mut alternating = vec![1.0; 64];
    for i in (0..64).step_by(2) {
        alternating[i] = f64::NAN;
    }
    assert!(
        !verify_cola(&alternating, 16, 1e-5),
        "verify_cola MUST return false for alternating NaNs"
    );
}

#[test]
fn test_verify_cola_inf_slices_rejected() {
    let pos_inf = vec![f64::INFINITY; 64];
    assert!(
        !verify_cola(&pos_inf, 16, 1e-5),
        "verify_cola MUST return false for +inf window"
    );

    let neg_inf = vec![f64::NEG_INFINITY; 64];
    assert!(
        !verify_cola(&neg_inf, 16, 1e-5),
        "verify_cola MUST return false for -inf window"
    );

    // Partial infinities
    for &inf_val in &[f64::INFINITY, f64::NEG_INFINITY] {
        for pos in [0, 15, 32, 63] {
            let mut window = vec![1.0; 64];
            window[pos] = inf_val;
            assert!(
                !verify_cola(&window, 16, 1e-5),
                "verify_cola MUST return false when inf is at index {}",
                pos
            );
        }
    }

    // Mix of NaN and Inf
    let mut mixed = vec![1.0; 64];
    mixed[5] = f64::NAN;
    mixed[10] = f64::INFINITY;
    mixed[20] = f64::NEG_INFINITY;
    assert!(
        !verify_cola(&mixed, 16, 1e-5),
        "verify_cola MUST return false for mixed NaN/Inf window"
    );
}

#[test]
fn test_verify_cola_non_finite_and_negative_tolerances_rejected() {
    let hann = generate_window(WindowType::Hann, 256).unwrap();

    assert!(
        !verify_cola(&hann, 128, f64::NAN),
        "verify_cola MUST return false for NaN tolerance"
    );
    assert!(
        !verify_cola(&hann, 128, f64::INFINITY),
        "verify_cola MUST return false for +inf tolerance"
    );
    assert!(
        !verify_cola(&hann, 128, f64::NEG_INFINITY),
        "verify_cola MUST return false for -inf tolerance"
    );
    assert!(
        !verify_cola(&hann, 128, -1e-5),
        "verify_cola MUST return false for negative tolerance"
    );
}

#[test]
fn test_verify_cola_loose_tolerance_accepted() {
    // Standard periodic Hann window: overlap sum S[p] = 1.0 for all p at R = 128.
    // min_val = 1.0, max_val = 1.0.
    let hann = generate_window(WindowType::Hann, 256).unwrap();

    // Verify tight tolerance works
    assert!(verify_cola(&hann, 128, 1e-5));

    // Verify loose tolerance: tolerance >= min_val (1.0)
    // Previously, `if min_val <= tolerance` caused false rejection when tolerance >= 1.0!
    assert!(
        verify_cola(&hann, 128, 1.0),
        "verify_cola MUST accept when tolerance == min_val"
    );
    assert!(
        verify_cola(&hann, 128, 1.5),
        "verify_cola MUST accept when tolerance (1.5) > min_val (1.0)"
    );
    assert!(
        verify_cola(&hann, 128, 10.0),
        "verify_cola MUST accept when tolerance (10.0) >> min_val (1.0)"
    );
    assert!(
        verify_cola(&hann, 128, 1000.0),
        "verify_cola MUST accept for very large tolerance"
    );
}

#[test]
fn test_verify_cola_degenerate_zero_and_negative_windows_rejected_even_with_loose_tolerance() {
    // All-zeros window has min_val = 0.0 <= 0.0 -> must be rejected even with huge tolerance!
    let zeros = vec![0.0; 64];
    assert!(
        !verify_cola(&zeros, 16, 1000.0),
        "verify_cola MUST reject all-zeros window even with huge tolerance"
    );

    // Negative window has min_val < 0.0 -> must be rejected even with huge tolerance!
    let negative = vec![-1.0; 64];
    assert!(
        !verify_cola(&negative, 16, 1000.0),
        "verify_cola MUST reject negative window even with huge tolerance"
    );

    // Window with one zero bin in overlap sum
    let mut dip = vec![1.0; 64];
    // zero out phase 3
    for k in 0..4 {
        dip[3 + k * 16] = 0.0;
    }
    assert!(
        !verify_cola(&dip, 16, 1000.0),
        "verify_cola MUST reject window with zero-sum phase even with huge tolerance"
    );
}

#[test]
fn test_verify_cola_hop_size_and_empty_boundaries() {
    let w = vec![1.0; 64];
    assert!(!verify_cola(&[], 16, 1e-5), "Empty window must return false");
    assert!(!verify_cola(&w, 0, 1e-5), "Hop size 0 must return false");
    assert!(!verify_cola(&w, 65, 1e-5), "Hop size > len must return false");

    // Exact length hop size:
    // Rectangular window has all 1.0s, so R = N has min_val = 1.0, max_val = 1.0 -> true
    assert!(verify_cola(&w, 64, 1e-5));

    // Periodic Hann window at R = N has w[0] = 0.0, so min_val = 0.0 <= 0.0 -> false
    let hann = generate_window(WindowType::Hann, 64).unwrap();
    assert!(!verify_cola(&hann, 64, 1e-5));
}

// =========================================================================
// 2. EMPIRICAL VERIFICATION: Complex::exp(800.0, 0.0) & Boundary Invariants
// =========================================================================

#[test]
fn test_complex_exp_large_real_no_nan() {
    let z800 = Complex::new(800.0, 0.0).exp();
    assert!(
        z800.re.is_infinite() && z800.re > 0.0,
        "exp(800 + 0i).re must be +inf, got {}",
        z800.re
    );
    assert_eq!(
        z800.im, 0.0,
        "exp(800 + 0i).im must be 0.0, got {}",
        z800.im
    );
    assert!(
        !z800.im.is_nan(),
        "exp(800 + 0i).im must NOT be NaN"
    );
    assert!(
        z800.im.is_sign_positive(),
        "exp(800 + 0i).im must be +0.0"
    );

    // Range of large real values across and above the double-precision overflow threshold (~709.78)
    for &re in &[709.0, 709.78, 709.79, 710.0, 800.0, 1000.0, 1e10, f64::INFINITY] {
        let z = Complex::new(re, 0.0).exp();
        assert!(
            !z.im.is_nan(),
            "exp({} + 0i).im must not be NaN",
            re
        );
        assert_eq!(
            z.im, 0.0,
            "exp({} + 0i).im must be strictly 0.0",
            re
        );
    }
}

#[test]
fn test_complex_exp_signed_zero_preservation() {
    let z_neg_zero = Complex::new(800.0, -0.0).exp();
    assert!(z_neg_zero.re.is_infinite() && z_neg_zero.re > 0.0);
    assert_eq!(z_neg_zero.im, 0.0);
    assert!(
        z_neg_zero.im.is_sign_negative(),
        "exp(800 - 0i).im must preserve negative signed zero"
    );

    let z_neg_zero_normal = Complex::new(2.0, -0.0).exp();
    assert!((z_neg_zero_normal.re - 2.0f64.exp()).abs() < EPSILON);
    assert_eq!(z_neg_zero_normal.im, 0.0);
    assert!(
        z_neg_zero_normal.im.is_sign_negative(),
        "exp(2 - 0i).im must preserve negative signed zero"
    );
}

#[test]
fn test_complex_exp_negative_infinity_real() {
    // e^(-inf + iy) = 0.0 + 0.0i for any y
    let z1 = Complex::new(f64::NEG_INFINITY, 0.0).exp();
    assert_eq!(z1, Complex::ZERO);

    let z2 = Complex::new(f64::NEG_INFINITY, 1.0).exp();
    assert_eq!(z2, Complex::ZERO);

    let z3 = Complex::new(f64::NEG_INFINITY, f64::INFINITY).exp();
    assert_eq!(z3, Complex::ZERO);

    let z4 = Complex::new(f64::NEG_INFINITY, f64::NAN).exp();
    assert_eq!(z4, Complex::ZERO);
}

// =========================================================================
// 3. EMPIRICAL VERIFICATION: Extreme Inputs, Denormals, Scaled Hypot in ln()
// =========================================================================

#[test]
fn test_complex_ln_scaled_hypot_avoids_overflow() {
    // z = (1.5e308, 1.5e308)
    // Direct hypot would overflow to +inf, causing ln() to return +inf.
    // Scaled hypot should compute ln(|z|) ≈ 709.948247 without overflow.
    let z = Complex::new(1.5e308, 1.5e308);
    let log_z = z.ln();

    assert!(
        log_z.re.is_finite(),
        "ln(1.5e308, 1.5e308).re must be finite, got {}",
        log_z.re
    );
    assert!(
        (log_z.re - 709.948247).abs() < 1e-4,
        "ln(1.5e308, 1.5e308).re expected ≈ 709.948247, got {}",
        log_z.re
    );
    assert!(
        (log_z.im - FRAC_PI_4).abs() < 1e-12,
        "ln(1.5e308, 1.5e308).im expected ≈ pi/4, got {}",
        log_z.im
    );

    // Near maximum representable f64 values
    let z_max = Complex::new(1.79e308, 1.79e308);
    let log_max = z_max.ln();
    assert!(
        log_max.re.is_finite(),
        "ln(1.79e308, 1.79e308).re must be finite, got {}",
        log_max.re
    );
    assert!((log_max.re - 710.125).abs() < 0.1);
    assert!((log_max.im - FRAC_PI_4).abs() < 1e-12);

    // Mixed quadrants
    let q2 = Complex::new(-1.5e308, 1.5e308).ln();
    assert!(q2.re.is_finite());
    assert!((q2.im - (3.0 * FRAC_PI_4)).abs() < 1e-12);

    let q3 = Complex::new(-1.5e308, -1.5e308).ln();
    assert!(q3.re.is_finite());
    assert!((q3.im - (-3.0 * FRAC_PI_4)).abs() < 1e-12);

    let q4 = Complex::new(1.5e308, -1.5e308).ln();
    assert!(q4.re.is_finite());
    assert!((q4.im - (-FRAC_PI_4)).abs() < 1e-12);
}

#[test]
fn test_complex_ln_denormals_and_branch_cuts() {
    // Denormal input: magnitude ≈ 1.414e-320
    let tiny = Complex::new(1e-320, 1e-320);
    let log_tiny = tiny.ln();
    assert!(
        log_tiny.re.is_finite(),
        "ln(1e-320, 1e-320).re must be finite, got {}",
        log_tiny.re
    );
    // ln(1.414e-320) ≈ ln(1.414) - 320 * ln(10) ≈ 0.3465 - 736.827 ≈ -736.48
    assert!((log_tiny.re - (-736.48)).abs() < 0.1);
    assert!((log_tiny.im - FRAC_PI_4).abs() < 1e-12);

    // Branch cut along negative real axis
    let cut_pos = Complex::new(-1.0, 0.0).ln();
    assert!((cut_pos.re - 0.0).abs() < EPSILON);
    assert!((cut_pos.im - PI).abs() < EPSILON, "ln(-1 + 0i).im must be +pi");

    let cut_neg = Complex::new(-1.0, -0.0).ln();
    assert!((cut_neg.re - 0.0).abs() < EPSILON);
    assert!(
        (cut_neg.im - (-PI)).abs() < EPSILON,
        "ln(-1 - 0i).im must be -pi (preserving signed zero branch cut)"
    );

    // Natural log of zero
    let log_zero = Complex::ZERO.ln();
    assert!(
        log_zero.re.is_infinite() && log_zero.re < 0.0,
        "ln(0).re must be -inf"
    );
    assert_eq!(log_zero.im, 0.0);
}

// =========================================================================
// 4. EMPIRICAL VERIFICATION: Extreme Operations (Polar, Powf, Powc, Div)
// =========================================================================

#[test]
fn test_complex_from_polar_and_powf_overflow_protections() {
    // Polar with infinite radius at theta = 0 must yield (inf, 0.0) without NaN
    let pol_inf = Complex::from_polar(f64::INFINITY, 0.0);
    assert!(pol_inf.re.is_infinite() && pol_inf.re > 0.0);
    assert_eq!(pol_inf.im, 0.0);
    assert!(!pol_inf.im.is_nan());

    // powf with large positive base that overflows double precision
    let pow_overflow = Complex::new(10.0, 0.0).powf(400.0);
    assert!(pow_overflow.re.is_infinite() && pow_overflow.re > 0.0);
    assert_eq!(pow_overflow.im, 0.0);
    assert!(!pow_overflow.im.is_nan());

    // (-1)^0.5 = i
    let sqrt_neg_one = Complex::new(-1.0, 0.0).sqrt();
    assert!(sqrt_neg_one.re.abs() < 1e-15);
    assert!((sqrt_neg_one.im - 1.0).abs() < EPSILON);

    // i^i = e^(-pi/2) ≈ 0.207879576
    let i_pow_i = Complex::I.powc(Complex::I);
    let expected_i_pow_i = (-FRAC_PI_2).exp();
    assert!((i_pow_i.re - expected_i_pow_i).abs() < EPSILON);
    assert_eq!(i_pow_i.im, 0.0, "i^i.im must be strictly 0.0 without NaN");
}

#[test]
fn test_complex_smith_division_extreme_scales() {
    let num = Complex::new(1.0, 1.0);

    // Division by tiny numbers (near underflow)
    let tiny_denom = Complex::new(1e-200, 1e-200);
    let div_tiny = num / tiny_denom;
    assert!((div_tiny.re - 1e200).abs() / 1e200 < 1e-14);
    assert!(div_tiny.im.abs() < 1e185);

    // Division by huge numbers (near overflow)
    let huge_denom = Complex::new(1e200, 1e200);
    let div_huge = num / huge_denom;
    assert!((div_huge.re - 1e-200).abs() / 1e-200 < 1e-14);
    assert!(div_huge.im.abs() < 1e-214);

    // Division by zero
    let div_zero = num / Complex::ZERO;
    assert!(div_zero.re.is_nan());
    assert!(div_zero.im.is_nan());
}

// =========================================================================
// 5. EMPIRICAL VERIFICATION: Window Invariants and Gain Stress Tests
// =========================================================================

#[test]
fn test_symmetric_windows_cola_behavior() {
    // Symmetric Hann window of length N=64 has ripples in the overlap sum at R = N/2
    let w_sym = generate_window_symmetric(WindowType::Hann, 64).unwrap();
    assert!(
        !verify_cola(&w_sym, 32, 1e-5),
        "Symmetric Hann window at R=N/2 must FAIL COLA due to non-constant sum"
    );

    // Periodic Hann window MUST pass COLA
    let w_per = generate_window(WindowType::Hann, 64).unwrap();
    assert!(
        verify_cola(&w_per, 32, 1e-5),
        "Periodic Hann window at R=N/2 must PASS COLA"
    );
}

#[test]
fn test_window_gain_functions_extremes_and_empty() {
    // Empty buffer handling
    assert_eq!(coherent_gain(&[]), 0.0);
    assert_eq!(energy_gain(&[]), 0.0);
    assert_eq!(equivalent_noise_bandwidth(&[]), 0.0);

    // Large finite values without panic or loss of proportionality
    let large_w = vec![1e150; 100];
    let gc = coherent_gain(&large_w);
    assert!((gc - 1e150).abs() / 1e150 < 1e-14);
    let ge = energy_gain(&large_w);
    assert!((ge - 1e300).abs() / 1e300 < 1e-14);
    let enbw = equivalent_noise_bandwidth(&large_w);
    assert!((enbw - 1.0).abs() < 1e-12);
}

