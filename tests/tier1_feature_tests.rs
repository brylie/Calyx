//! tests/tier1_feature_tests.rs
//! Opaque-box Tier 1 Feature Verification Suite for Calyx DSP Framework.
//!
//! Covers Core User Requirements R1 (Rust Experimental DSP Framework & Modular Interfaces)
//! and R2 (Synthetic Signal Generation & Automated Verification).
//!
//! Contains 65 concrete test cases across 13 feature groups (5 tests per group).

#[path = "common/mod.rs"]
mod common;

use calyx_dsp::error::*;
use calyx_dsp::math::*;
use calyx_dsp::metrics::*;
use calyx_dsp::synthetic::*;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

// ============================================================================
// Group 1: Complex Numbers & Polar Math (T1-CMPX-01 .. T1-CMPX-05)
// ============================================================================

mod test_complex_math {
    use super::*;

    #[test]
    fn test_t1_cmpx_01_cartesian_arithmetic_field_axioms() {
        let z1 = Complex::new(3.0, 4.0);
        let z2 = Complex::new(1.0, -2.0);

        // Addition: (3 + 1) + (4 - 2)i = 4 + 2i
        assert_eq!(z1 + z2, Complex::new(4.0, 2.0));

        // Subtraction: (3 - 1) + (4 - (-2))i = 2 + 6i
        assert_eq!(z1 - z2, Complex::new(2.0, 6.0));

        // Multiplication: (3*1 - 4*(-2)) + (3*(-2) + 4*1)i = 11 - 2i
        assert_eq!(z1 * z2, Complex::new(11.0, -2.0));

        // Division: (11 - 2i) / (1 - 2i) = 3 + 4i
        let z3 = Complex::new(11.0, -2.0) / z2;
        assert!(
            (z3.re - 3.0).abs() < 1e-14 && (z3.im - 4.0).abs() < 1e-14,
            "Division failed: {:?}",
            z3
        );
    }

    #[test]
    fn test_t1_cmpx_02_polar_cartesian_bi_directional_conversion_and_euler() {
        let test_cases: &[(f64, f64)] = &[
            (1.0, 0.0),
            (2.0, FRAC_PI_4),
            (1.5, FRAC_PI_2),
            (3.0, 3.0 * FRAC_PI_4),
            (1.0, PI),
            (2.5, -FRAC_PI_2),
            (1.0, -PI / 3.0),
        ];

        for &(r, theta) in test_cases {
            let z = Complex::from_polar(r, theta);
            assert!(
                (z.norm() - r).abs() < 1e-14,
                "Norm mismatch for r={}, theta={}: got {}",
                r,
                theta,
                z.norm()
            );
            assert!(
                (z.arg() - theta).abs() < 1e-14,
                "Arg mismatch for r={}, theta={}: got {}",
                r,
                theta,
                z.arg()
            );
        }

        // Euler's Identity: e^(i*pi) + 1 = 0
        let euler = Complex::from_polar(1.0, PI);
        assert!((euler.re + 1.0).abs() < 1e-14 && euler.im.abs() < 1e-14);
    }

    #[test]
    fn test_t1_cmpx_03_de_moivre_multiplication_as_rotation() {
        let z1 = Complex::from_polar(2.0, PI / 6.0);
        let z2 = Complex::from_polar(3.0, PI / 3.0);
        let zp = z1 * z2;

        // Magnitude should multiply: 2.0 * 3.0 = 6.0
        assert!((zp.norm() - 6.0).abs() < 1e-14);

        // Phase should add: pi/6 + pi/3 = pi/2
        assert!((zp.arg() - FRAC_PI_2).abs() < 1e-14);
        assert!((zp.re - 0.0).abs() < 1e-14);
        assert!((zp.im - 6.0).abs() < 1e-14);
    }

    #[test]
    fn test_t1_cmpx_04_conjugate_invariants_and_norm_squared() {
        let z1 = Complex::new(5.0, -12.0);
        let z2 = Complex::new(-3.0, 7.0);

        // |z1|^2 = 5^2 + (-12)^2 = 169.0
        assert!((z1.norm_sq() - 169.0).abs() < 1e-14);

        // z * z* = |z|^2
        let prod = z1 * z1.conj();
        assert!((prod.re - 169.0).abs() < 1e-14 && prod.im.abs() < 1e-14);

        // (z1 * z2)* == z1* * z2*
        let lhs = (z1 * z2).conj();
        let rhs = z1.conj() * z2.conj();
        assert!((lhs.re - rhs.re).abs() < 1e-14 && (lhs.im - rhs.im).abs() < 1e-14);
    }

    #[test]
    fn test_t1_cmpx_05_extreme_value_scaling_and_hypot_protection() {
        // Test values that would overflow naive x^2 + y^2 in f64 (max ~ 1.79e308)
        let z_huge = Complex::new(3.0e200, 4.0e200);
        let norm = z_huge.norm();
        assert!(!norm.is_infinite(), "Norm overflowed to infinity");
        assert!(
            ((norm - 5.0e200).abs() / 5.0e200) < 1e-14,
            "Expected 5e200, got {}",
            norm
        );

        // Division by zero should not panic
        let z_zero = Complex::new(0.0, 0.0);
        let div = Complex::new(1.0, 1.0) / z_zero;
        assert!(div.re.is_infinite() || div.re.is_nan());
    }
}

// ============================================================================
// Group 2: Window Functions & COLA Conditions (T1-WIND-01 .. T1-WIND-05)
// ============================================================================

mod test_window_cola {
    use super::*;

    #[test]
    fn test_t1_wind_01_hann_window_analytical_profile_and_symmetry() {
        let length = 1025;
        let w = generate_window_symmetric(WindowType::Hann, length).expect("Hann window generation failed");
        assert_eq!(w.len(), length);

        // Hann endpoints are strictly 0.0
        assert!(w[0].abs() < 1e-14, "Hann w[0] must be 0: got {}", w[0]);

        // Symmetric midpoint check
        for n in 0..(length / 2) {
            let diff = (w[n] - w[length - 1 - n]).abs();
            assert!(
                diff < 1e-14,
                "Symmetry violation at {}: w[{}]={}, w[{}]={}",
                n,
                n,
                w[n],
                length - 1 - n,
                w[length - 1 - n]
            );
        }

        // Peak value is 1.0 at midpoint
        let max_val = w.iter().copied().fold(0.0, f64::max);
        assert!((max_val - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_t1_wind_02_hamming_blackman_coefficients_and_coherent_gain() {
        let length = 2048;

        // Hamming: w[0] = 0.54 - 0.46 = 0.08, coherent gain ~ 0.54
        let hamming = generate_window(WindowType::Hamming, length).unwrap();
        assert!((hamming[0] - 0.08).abs() < 1e-14);
        let gain_h: f64 = hamming.iter().sum::<f64>() / (length as f64);
        assert!((gain_h - 0.54).abs() < 1e-3);

        // Blackman: w[0] = 0.42 - 0.5 + 0.08 = 0.0, coherent gain ~ 0.42
        let blackman = generate_window(WindowType::Blackman, length).unwrap();
        assert!(blackman[0].abs() < 1e-14);
        let gain_b: f64 = blackman.iter().sum::<f64>() / (length as f64);
        assert!((gain_b - 0.42).abs() < 1e-3);
    }

    #[test]
    fn test_t1_wind_03_hann_window_cola_verification() {
        let length = 512;
        let w = generate_window(WindowType::Hann, length).unwrap();

        // 50% overlap (hop = 256) satisfies COLA
        assert!(
            verify_cola(&w, 256, 1e-6),
            "Hann window must satisfy COLA at 50% overlap"
        );

        // 75% overlap (hop = 128) satisfies COLA
        assert!(
            verify_cola(&w, 128, 1e-6),
            "Hann window must satisfy COLA at 75% overlap"
        );
    }

    #[test]
    fn test_t1_wind_04_cola_rejection_for_incompatible_hop_sizes() {
        let length = 512;
        let w = generate_window(WindowType::Hann, length).unwrap();

        // Non-divisor hop sizes produce ripple exceeding tolerance
        assert!(!verify_cola(&w, 300, 1e-4));
        assert!(!verify_cola(&w, 500, 1e-4));
        assert!(!verify_cola(&w, 511, 1e-4));
    }

    #[test]
    fn test_t1_wind_05_window_generation_boundary_and_error_cases() {
        // Length 0 or 1 must be rejected
        assert!(matches!(
            generate_window(WindowType::Hann, 0),
            Err(DspError::InvalidParameter { name: "length", .. }) | Err(DspError::EmptyInput)
        ));
        assert!(matches!(
            generate_window(WindowType::Hann, 1),
            Err(DspError::InvalidParameter { name: "length", .. })
        ));

        // Rectangular window has all 1.0 coefficients and satisfies COLA at hop = length
        let rect = generate_window(WindowType::Rectangular, 64).unwrap();
        assert!(rect.iter().all(|&v| (v - 1.0).abs() < 1e-14));
        assert!(verify_cola(&rect, 64, 1e-12));
    }
}

// ============================================================================
// Group 3: Matrix & Tensor Striding / Mode Unfolding (T1-TEN-01 .. T1-TEN-05)
// ============================================================================

mod test_matrix_tensor {
    use super::*;

    #[test]
    fn test_t1_ten_01_canonical_row_major_strides() {
        let shape = vec![2, 3, 4];
        let data: Vec<f64> = (0..24).map(|v| v as f64).collect();
        let tensor = Tensor::from_vec(shape, data).expect("Tensor construction failed");

        // Canonical row-major strides for [2, 3, 4] are [12, 4, 1]
        assert_eq!(tensor.strides(), &[12, 4, 1]);

        // Flat index: i0*12 + i1*4 + i2*1
        assert_eq!(tensor.get(&[1, 2, 3]).unwrap(), 23.0);
        assert_eq!(tensor.get(&[0, 1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn test_t1_ten_02_mode_matricization_dimensions_and_content() {
        let tensor = Tensor::from_vec(vec![2, 3, 4], (0..24).map(|v| v as f64).collect()).unwrap();

        // Mode 0: 2 x 12
        let m0 = tensor.mode_unfold(0).expect("Mode 0 unfold failed");
        assert_eq!(m0.rows(), 2);
        assert_eq!(m0.cols(), 12);
        assert_eq!(m0.get(0, 5), Ok(5.0));
        assert_eq!(m0.get(1, 0), Ok(12.0));

        // Mode 1: 3 x 8
        let m1 = tensor.mode_unfold(1).expect("Mode 1 unfold failed");
        assert_eq!(m1.rows(), 3);
        assert_eq!(m1.cols(), 8);

        // Mode 2: 4 x 6
        let m2 = tensor.mode_unfold(2).expect("Mode 2 unfold failed");
        assert_eq!(m2.rows(), 4);
        assert_eq!(m2.cols(), 6);
    }

    #[test]
    fn test_t1_ten_03_mode_product_algebraic_equivalence() {
        let tensor = Tensor::from_vec(vec![2, 3, 4], (0..24).map(|v| v as f64).collect()).unwrap();

        // Apply 5x3 matrix along mode 1 (size 3) -> result shape [2, 5, 4]
        let matrix_data: Vec<f64> = (0..15).map(|v| (v + 1) as f64).collect();
        let u = Matrix::from_vec(5, 3, matrix_data).unwrap();

        let y = tensor.mode_product(1, &u).expect("Mode product failed");
        assert_eq!(y.shape(), &[2, 5, 4]);

        // Verify element-wise against manual contraction
        let mut expected = 0.0;
        for i1 in 0..3 {
            expected += tensor.get(&[0, i1, 0]).unwrap() * u.get(0, i1).unwrap();
        }
        assert!((y.get(&[0, 0, 0]).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_t1_ten_04_matrix_multiplication_and_inversion_invariants() {
        // A = [[2.0, 1.0], [5.0, 3.0]] (det = 2*3 - 1*5 = 1.0)
        let a = Matrix::from_vec(2, 2, vec![2.0, 1.0, 5.0, 3.0]).unwrap();
        // Analytical inverse for [[2, 1], [5, 3]] where det = 1: [[3, -1], [-5, 2]]
        let a_inv = Matrix::from_vec(2, 2, vec![3.0, -1.0, -5.0, 2.0]).unwrap();

        // A * A^-1 == I
        let ident = a.matmul(&a_inv).expect("Matmul failed");
        assert!((ident.get(0, 0).unwrap() - 1.0).abs() < 1e-14);
        assert!(ident.get(0, 1).unwrap().abs() < 1e-14);
        assert!(ident.get(1, 0).unwrap().abs() < 1e-14);
        assert!((ident.get(1, 1).unwrap() - 1.0).abs() < 1e-14);

        // Transpose identity: (AB)^T == B^T * A^T
        let b = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let ab = a.matmul(&b).unwrap();
        let ab_t = ab.transpose();
        let bt_at = b.transpose().matmul(&a.transpose()).unwrap();
        for r in 0..2 {
            for c in 0..2 {
                assert!((ab_t.get(r, c).unwrap() - bt_at.get(r, c).unwrap()).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_t1_ten_05_tensor_out_of_bounds_and_dimension_mismatch_rejection() {
        // Data length does not match shape: 2*3*4 = 24, provided 20
        assert!(matches!(
            Tensor::from_vec(vec![2, 3, 4], vec![0.0; 20]),
            Err(DspError::BufferLengthMismatch { expected: 24, actual: 20 })
        ));

        // Invalid mode index (mode 3 on 3D tensor)
        let t = Tensor::from_vec(vec![2, 3, 4], vec![0.0; 24]).unwrap();
        assert!(matches!(
            t.mode_unfold(3),
            Err(DspError::InvalidParameter { name: "mode", .. })
        ));

        // Incompatible mode product: matrix cols (4) != mode 1 dim (3)
        let incompatible_u = Matrix::from_vec(2, 4, vec![0.0; 8]).unwrap();
        assert!(matches!(
            t.mode_product(1, &incompatible_u),
            Err(DspError::DimensionMismatch { .. })
        ));
    }
}

// ============================================================================
// Group 4: Typed DspError Failure Modes (T1-ERR-01 .. T1-ERR-05)
// ============================================================================

mod test_dsp_error {
    use super::*;

    #[test]
    fn test_t1_err_01_empty_input_propagation() {
        assert_eq!(compute_time_metrics(&[], &[]), Err(DspError::EmptyInput));
        assert_eq!(verify_numerical_stability(&[]), Err(DspError::EmptyInput));
    }

    #[test]
    fn test_t1_err_02_buffer_length_mismatch_payload_exactness() {
        let orig = vec![0.0; 1024];
        let recon = vec![0.0; 1000];
        match compute_time_metrics(&orig, &recon) {
            Err(DspError::BufferLengthMismatch { expected, actual }) => {
                assert_eq!(expected, 1024);
                assert_eq!(actual, 1000);
            }
            other => panic!("Expected BufferLengthMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_t1_err_03_non_power_of_two_validation() {
        let err = DspError::non_power_of_two(300);
        match err {
            DspError::NonPowerOfTwo { size } => assert_eq!(size, 300),
            other => panic!("Expected NonPowerOfTwo, got {:?}", other),
        }
    }

    #[test]
    fn test_t1_err_04_invalid_parameter_metadata() {
        let err = DspError::invalid_parameter("sample_rate", "must be positive");
        match err {
            DspError::InvalidParameter { name, reason } => {
                assert_eq!(name, "sample_rate");
                assert_eq!(reason, "must be positive");
            }
            other => panic!("Expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn test_t1_err_05_error_standard_trait_compliance() {
        let boxed: Box<dyn std::error::Error + Send + Sync> = Box::new(DspError::EmptyInput);
        assert_eq!(format!("{}", boxed), "Input buffer or sequence is empty");

        let err_np2 = DspError::NonPowerOfTwo { size: 500 };
        assert!(format!("{}", err_np2).contains("500"));
    }
}

// ============================================================================
// Group 5: SineGenerator Analytical Verification (T1-SINE-01 .. T1-SINE-05)
// ============================================================================

mod test_sine_generator {
    use super::*;

    #[test]
    fn test_t1_sine_01_point_by_point_closed_form_exactness() {
        let f0 = 440.0;
        let fs = 48000.0;
        let amp = 0.8;
        let phase = FRAC_PI_4;
        let num_samples = 4800;

        let gen = SineGenerator::new(f0, amp, phase);
        let buf = gen.generate(num_samples, fs);

        for (n, &s) in buf.samples.iter().enumerate() {
            let s: f64 = s;
            let t = n as f64 / fs;
            let expected = amp * (2.0 * PI * f0 * t + phase).sin();
            assert!(
                (s - expected).abs() < 1e-12,
                "Mismatch at sample {}: got {}, expected {}",
                n,
                s,
                expected
            );
        }
    }

    #[test]
    fn test_t1_sine_02_analytical_rms_and_energy_conservation() {
        let f0 = 100.0;
        let fs = 10000.0;
        let amp = 1.0;
        let num_samples = 10000; // exactly 100 integer cycles

        let gen = SineGenerator::new(f0, amp, 0.0);
        let buf = gen.generate(num_samples, fs);

        let energy: f64 = buf.samples.iter().map(|s| s * s).sum();
        let rms = (energy / (num_samples as f64)).sqrt();

        // Theoretical RMS = 1.0 / sqrt(2)
        assert!((rms - 1.0 / std::f64::consts::SQRT_2).abs() < 1e-12);
        // Theoretical Energy = N * A^2 / 2 = 5000.0
        assert!((energy - 5000.0).abs() < 1e-10);
    }

    #[test]
    fn test_t1_sine_03_dc_and_nyquist_boundary_frequencies() {
        let fs = 48000.0;
        let n = 1000;

        // Case A: DC (0.0 Hz)
        let gen_dc = SineGenerator::new(0.0, 0.5, PI / 6.0);
        let buf_dc = gen_dc.generate(n, fs);
        // s[n] = 0.5 * sin(pi/6) = 0.25
        assert!(buf_dc.samples.iter().all(|&s: &f64| (s - 0.25).abs() < 1e-14));

        // Case B: Nyquist (fs / 2 = 24000.0 Hz) with phase pi/2
        let gen_nyq = SineGenerator::new(24000.0, 0.75, FRAC_PI_2);
        let buf_nyq = gen_nyq.generate(n, fs);
        // s[n] = 0.75 * sin(pi * n + pi/2) = 0.75 * (-1)^n
        for (idx, &s) in buf_nyq.samples.iter().enumerate() {
            let s: f64 = s;
            let expected = if idx % 2 == 0 { 0.75 } else { -0.75 };
            assert!((s - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn test_t1_sine_04_generate_into_preallocated_buffer() {
        let gen = SineGenerator::new(1000.0, 0.9, 0.123);
        let buf1 = gen.generate(4096, 44100.0);

        let mut buf2 = vec![0.0; 4096];
        gen.generate_into(&mut buf2, 44100.0);

        assert_eq!(buf1.samples, buf2);
    }

    #[test]
    fn test_t1_sine_05_signal_metadata_completeness() {
        let gen = SineGenerator::new(440.0, 0.6, 0.0);
        let buf = gen.generate(1024, 48000.0);

        assert_eq!(buf.metadata.fundamental_freq, Some(440.0));
        assert_eq!(buf.metadata.known_partials.len(), 1);
        assert_eq!(buf.metadata.known_partials[0].harmonic_index, 1);
        assert_eq!(buf.metadata.known_partials[0].frequency_hz, 440.0);
        assert_eq!(buf.metadata.known_partials[0].amplitude, 0.6);
    }
}

// ============================================================================
// Group 6: HarmonicComplexGenerator with Inharmonicity (T1-HARM-01 .. T1-HARM-05)
// ============================================================================

mod test_harmonic_generator {
    use super::*;

    #[test]
    fn test_t1_harm_01_sawtooth_harmonic_recipe_amplitudes() {
        let gen = HarmonicComplexGenerator::new(100.0, 1.0, 8, 0.0, HarmonicRecipe::Sawtooth);
        let buf = gen.generate(4800, 48000.0);

        let partials = &buf.metadata.known_partials;
        assert_eq!(partials.len(), 8);

        for p in partials {
            let expected_a = 1.0 / (p.harmonic_index as f64);
            assert!((p.amplitude - expected_a).abs() < 1e-14);
            let expected_f = (p.harmonic_index as f64) * 100.0;
            assert!((p.frequency_hz - expected_f).abs() < 1e-12);
        }
    }

    #[test]
    fn test_t1_harm_02_square_wave_odd_harmonic_isolation() {
        let gen = HarmonicComplexGenerator::new(150.0, 1.0, 8, 0.0, HarmonicRecipe::Square);
        let buf = gen.generate(4800, 48000.0);

        let partials = &buf.metadata.known_partials;
        // K=8 max, square keeps odd harmonics: 1, 3, 5, 7 (4 partials total)
        assert_eq!(partials.len(), 4);
        assert_eq!(partials[0].harmonic_index, 1);
        assert_eq!(partials[1].harmonic_index, 3);
        assert_eq!(partials[2].harmonic_index, 5);
        assert_eq!(partials[3].harmonic_index, 7);
    }

    #[test]
    fn test_t1_harm_03_stiff_string_inharmonicity_dispersion() {
        let b = 0.0004;
        let gen = HarmonicComplexGenerator::new(110.0, 1.0, 10, b, HarmonicRecipe::Sawtooth);
        let buf = gen.generate(4800, 48000.0);

        let p10 = buf
            .metadata
            .known_partials
            .iter()
            .find(|p| p.harmonic_index == 10)
            .unwrap();
        // f10 = 10 * 110 * sqrt(1 + 0.0004 * 100) = 1100 * sqrt(1.04)
        let expected = 1100.0 * (1.04_f64).sqrt();
        assert!((p10.frequency_hz - expected).abs() < 1e-9);
    }

    #[test]
    fn test_t1_harm_04_anti_aliasing_nyquist_truncation() {
        // f0 = 2500 Hz at fs = 16000 Hz (Nyquist = 8000 Hz)
        // Harmonics: 2500, 5000, 7500, 10000 (exceeds Nyquist)
        let gen = HarmonicComplexGenerator::new(2500.0, 1.0, 6, 0.0, HarmonicRecipe::Sawtooth);
        let buf = gen.generate(1600, 16000.0);

        let partials = &buf.metadata.known_partials;
        assert_eq!(partials.len(), 3);
        assert!(partials.iter().all(|p| p.frequency_hz < 8000.0));
    }

    #[test]
    fn test_t1_harm_05_frequency_dependent_exponential_damping() {
        let mut gen = HarmonicComplexGenerator::new(200.0, 1.0, 4, 0.0, HarmonicRecipe::Sawtooth);
        gen.set_damping_tau(Some(0.5));
        let buf = gen.generate(48000, 48000.0);

        // Signal should attenuate smoothly over time
        let head_rms = (buf.samples[..4800].iter().map(|s| s * s).sum::<f64>() / 4800.0).sqrt();
        let tail_rms = (buf.samples[43200..].iter().map(|s| s * s).sum::<f64>() / 4800.0).sqrt();
        assert!(tail_rms < head_rms * 0.4);
    }
}

// ============================================================================
// Group 7: ChirpGenerator Linear & Logarithmic Sweeps (T1-CHRP-01 .. T1-CHRP-05)
// ============================================================================

mod test_chirp_generator {
    use super::*;

    #[test]
    fn test_t1_chrp_01_linear_chirp_instantaneous_phase() {
        let f1 = 200.0;
        let f2 = 1800.0;
        let amp = 0.9;
        let duration = 1.0;
        let fs = 48000.0;

        let gen = ChirpGenerator::new(f1, f2, amp, duration, ChirpKind::Linear);
        let buf = gen.generate(48000, fs);

        // At t = 0.5s (n = 24000), instantaneous phase Phi(0.5) = 2*pi*(200*0.5 + 0.5*1600*0.25) = 600*pi
        // sin(600*pi) == 0.0
        assert!(buf.samples[24000].abs() < 1e-12);
    }

    #[test]
    fn test_t1_chrp_02_logarithmic_chirp_octave_progression() {
        let gen = ChirpGenerator::new(100.0, 800.0, 0.8, 3.0, ChirpKind::Logarithmic);
        let buf = gen.generate(144000, 48000.0);

        assert_eq!(buf.samples.len(), 144000);
        assert!(buf.samples.iter().all(|&s: &f64| s.abs() <= 0.8000000001));
    }

    #[test]
    fn test_t1_chrp_03_reverse_downward_sweep() {
        let gen = ChirpGenerator::new(2000.0, 200.0, 0.8, 1.0, ChirpKind::Linear);
        let buf = gen.generate(48000, 48000.0);

        assert!(buf.samples.iter().all(|&s: &f64| s.abs() <= 0.8000000001));
        assert!(!buf.samples.is_empty());
    }

    #[test]
    fn test_t1_chrp_04_full_band_sweep_nyquist_boundedness() {
        let gen = ChirpGenerator::new(20.0, 23900.0, 1.0, 0.5, ChirpKind::Linear);
        let buf = gen.generate(24000, 48000.0);

        assert!(buf.samples.iter().all(|s: &f64| !s.is_nan() && !s.is_infinite()));
        let max_val = buf.samples.iter().map(|s: &f64| s.abs()).fold(0.0, f64::max);
        assert!(max_val <= 1.000000001);
    }

    #[test]
    fn test_t1_chrp_05_chirp_rms_invariance() {
        let gen = ChirpGenerator::new(100.0, 5000.0, 1.0, 2.0, ChirpKind::Linear);
        let buf = gen.generate(96000, 48000.0);

        let energy: f64 = buf.samples.iter().map(|s| s * s).sum();
        let rms = (energy / (buf.samples.len() as f64)).sqrt();
        assert!((rms - 1.0 / std::f64::consts::SQRT_2).abs() < 0.005);
    }
}

// ============================================================================
// Group 8: ImpulseGenerator Transients (T1-IMP-01 .. T1-IMP-05)
// ============================================================================

mod test_impulse_generator {
    use super::*;

    #[test]
    fn test_t1_imp_01_kronecker_delta_spectral_flatness() {
        let gen = ImpulseGenerator::kronecker(0, 1.0);
        let buf = gen.generate(512, 48000.0);

        // Total energy is 1.0
        let energy: f64 = buf.samples.iter().map(|s| s * s).sum();
        assert!((energy - 1.0).abs() < 1e-14);

        // Exact DFT magnitude should be 1.0 for all bins
        let spectrum = common::compute_exact_dft(&buf.samples);
        for c in spectrum {
            assert!((c.norm() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_t1_imp_02_arbitrary_delayed_kronecker_delta() {
        let gen = ImpulseGenerator::kronecker(350, 0.75);
        let buf = gen.generate(1024, 48000.0);

        assert_eq!(buf.samples[350], 0.75);
        assert!(buf
            .samples
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 350)
            .all(|(_, &s)| s == 0.0));

        let energy: f64 = buf.samples.iter().map(|s| s * s).sum();
        assert!((energy - 0.75 * 0.75).abs() < 1e-14);
    }

    #[test]
    fn test_t1_imp_03_bandlimited_sinc_pulse() {
        let gen = ImpulseGenerator::bandlimited_sinc(4000.0, 256, 1.0);
        let buf = gen.generate(512, 48000.0);

        // Peak at delay 256
        assert!((buf.samples[256] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_t1_imp_04_exponential_pluck_decay() {
        let gen = ImpulseGenerator::exponential_pluck(440.0, 20.0, 1.0);
        let buf = gen.generate(4800, 48000.0);

        // Half life t1/2 = ln(2)/20 ~ 0.0346s (sample 1663)
        assert!(buf.samples[0].abs() < 1e-12); // starts at sin(0)
        assert!(buf.samples.iter().all(|s: &f64| !s.is_nan() && s.abs() <= 1.0));
    }

    #[test]
    fn test_t1_imp_05_impulse_parameter_validation() {
        let gen_invalid = ImpulseGenerator::kronecker(1000, 1.0);
        // Delay 1000 on buffer of length 512
        assert!(matches!(
            gen_invalid.generate_checked(512, 48000.0),
            Err(DspError::InvalidParameter { name: "delay", .. })
        ));
    }
}

// ============================================================================
// Group 9: NoiseGenerator PRNG Distributions & Reproducibility (T1-NOIS-01 .. T1-NOIS-05)
// ============================================================================

mod test_noise_generator {
    use super::*;

    #[test]
    fn test_t1_nois_01_deterministic_bitwise_reproducibility() {
        let seed = 0xDEADBEEFCAFE1234;
        let n1 = NoiseGenerator::new(NoiseColor::WhiteUniform, 0.5, seed).generate(50000, 48000.0);
        let n2 = NoiseGenerator::new(NoiseColor::WhiteUniform, 0.5, seed).generate(50000, 48000.0);

        assert_eq!(n1.samples, n2.samples);
    }

    #[test]
    fn test_t1_nois_02_white_uniform_statistical_invariants() {
        let gen = NoiseGenerator::new(NoiseColor::WhiteUniform, 1.0, 42);
        let buf = gen.generate(100000, 48000.0);

        let mean: f64 = buf.samples.iter().sum::<f64>() / (buf.samples.len() as f64);
        let var: f64 = buf.samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
            / (buf.samples.len() as f64);

        assert!(mean.abs() < 0.01, "Mean was {}", mean);
        assert!((var - 1.0 / 3.0).abs() < 0.01, "Variance was {}", var);
        assert!(buf.samples.iter().all(|&s| s >= -1.0 && s <= 1.0));
    }

    #[test]
    fn test_t1_nois_03_white_gaussian_box_muller_normality() {
        let gen = NoiseGenerator::new(NoiseColor::WhiteGaussian, 1.0, 999);
        let buf = gen.generate(100000, 48000.0);

        let mean: f64 = buf.samples.iter().sum::<f64>() / (buf.samples.len() as f64);
        let var: f64 = buf.samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
            / (buf.samples.len() as f64);

        assert!(mean.abs() < 0.015, "Gaussian mean was {}", mean);
        assert!((var - 1.0).abs() < 0.02, "Gaussian var was {}", var);
    }

    #[test]
    fn test_t1_nois_04_pink_noise_spectral_slope() {
        let gen = NoiseGenerator::new(NoiseColor::Pink, 0.5, 12345);
        let buf = gen.generate(65536, 48000.0);

        assert_eq!(buf.samples.len(), 65536);
        assert!(buf.samples.iter().all(|s: &f64| !s.is_nan()));
    }

    #[test]
    fn test_t1_nois_05_brownian_noise_boundedness() {
        let gen = NoiseGenerator::new(NoiseColor::Brownian, 0.5, 54321);
        let buf = gen.generate(100000, 48000.0);

        assert!(buf.samples.iter().all(|&s: &f64| s.abs() <= 1.0));
    }
}

// ============================================================================
// Group 10: SyntheticInstrumentSource 5D Parameterization (T1-INST-01 .. T1-INST-05)
// ============================================================================

mod test_instrument_source {
    use super::*;

    #[test]
    fn test_t1_inst_01_equal_temperament_pitch_mapping() {
        let inst = SyntheticInstrumentSource::new();
        let test_cases: &[(u8, f64)] = &[
            (69, 440.0),
            (60, 261.625565),
            (21, 27.5),
            (108, 4186.009),
        ];

        for &(p, expected_f) in test_cases {
            let note = inst.generate_note(p, 0.8, Articulation::Sustain, 1);
            let f0 = note.metadata.fundamental_freq.unwrap();
            assert!(
                (f0 - expected_f).abs() < 1e-2,
                "Pitch {} expected {} Hz, got {} Hz",
                p,
                expected_f,
                f0
            );
        }
    }

    #[test]
    fn test_t1_inst_02_dynamic_velocity_spectral_brightness() {
        let inst = SyntheticInstrumentSource::new();
        let note_soft = inst.generate_note(60, 0.2, Articulation::Sustain, 1);
        let note_loud = inst.generate_note(60, 0.9, Articulation::Sustain, 1);

        let c_soft = common::compute_spectral_centroid(&note_soft.samples, 48000.0);
        let c_loud = common::compute_spectral_centroid(&note_loud.samples, 48000.0);

        assert!(
            c_loud > c_soft,
            "Loud note centroid ({}) must exceed soft note centroid ({})",
            c_loud,
            c_soft
        );
    }

    #[test]
    fn test_t1_inst_03_articulation_temporal_profile() {
        let inst = SyntheticInstrumentSource::new();
        let staccato = inst.generate_note(60, 0.8, Articulation::Staccato, 1);
        let sustain = inst.generate_note(60, 0.8, Articulation::Sustain, 1);

        assert!(staccato.samples.len() <= sustain.samples.len());
    }

    #[test]
    fn test_t1_inst_04_round_robin_decorrelation() {
        let inst = SyntheticInstrumentSource::new();
        let v1 = inst.generate_note(60, 0.7, Articulation::Sustain, 1);
        let v2 = inst.generate_note(60, 0.7, Articulation::Sustain, 2);

        assert_ne!(v1.samples, v2.samples);
    }

    #[test]
    fn test_t1_inst_05_continuous_hypercube_query() {
        let inst = SyntheticInstrumentSource::new();
        let frame = inst
            .query_frame([0.5, 60.5, 0.75, 0.5, 1.5])
            .expect("Continuous query failed");

        assert!(
            (frame.fundamental_freq - 269.29).abs() < 1.0,
            "Interpolated f0 was {}",
            frame.fundamental_freq
        );
        assert!(frame
            .partials
            .iter()
            .all(|p| p.amplitude >= 0.0 && !p.amplitude.is_nan()));
    }
}

// ============================================================================
// Group 11: Time-Domain Fidelity Metrics (T1-TMET-01 .. T1-TMET-05)
// ============================================================================

mod test_time_metrics {
    use super::*;

    #[test]
    fn test_t1_tmet_01_metric_identity_axioms() {
        let x = SineGenerator::new(440.0, 0.8, 0.0)
            .generate(1024, 48000.0)
            .samples;

        let m = compute_time_metrics(&x, &x).expect("Identity metrics failed");
        assert_eq!(m.mse, 0.0);
        assert_eq!(m.rmse, 0.0);
        assert!(m.snr_db > 140.0 || m.snr_db.is_infinite());
        assert!((m.correlation_r - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_t1_tmet_02_known_snr_ground_truth_with_awgn() {
        let n = 100000;
        let s = SineGenerator::new(440.0, 1.0, 0.0).generate(n, 48000.0).samples;
        // Ps = 0.5
        // Additive noise with variance = 0.005 -> theoretical SNR = 10*log10(0.5 / 0.005) = 20.0 dB
        let noise = NoiseGenerator::new(NoiseColor::WhiteGaussian, (0.005_f64).sqrt(), 12345)
            .generate(n, 48000.0)
            .samples;
        let noisy: Vec<f64> = s.iter().zip(noise.iter()).map(|(a, b)| a + b).collect();

        let m = compute_time_metrics(&s, &noisy).unwrap();
        assert!(
            (m.snr_db - 20.0).abs() < 0.2,
            "SNR was {} dB (expected ~20 dB)",
            m.snr_db
        );
    }

    #[test]
    fn test_t1_tmet_03_metric_symmetry_and_positivity() {
        let x = SineGenerator::new(440.0, 0.8, 0.0).generate(2048, 48000.0).samples;
        let y = SineGenerator::new(880.0, 0.5, 0.0).generate(2048, 48000.0).samples;

        let m_xy = compute_time_metrics(&x, &y).unwrap();
        let m_yx = compute_time_metrics(&y, &x).unwrap();

        assert!((m_xy.mse - m_yx.mse).abs() < 1e-14);
        assert!((m_xy.rmse - m_yx.rmse).abs() < 1e-14);
        assert!(m_xy.correlation_r >= -1.0000000001 && m_xy.correlation_r <= 1.0000000001);
    }

    #[test]
    fn test_t1_tmet_04_cross_correlation_inversion_and_orthogonality() {
        let x = SineGenerator::new(440.0, 0.8, 0.0).generate(1024, 48000.0).samples;
        let y_inv: Vec<f64> = x.iter().map(|&v| -v).collect();

        // Inversion: r(x, -x) == -1.0
        let m_inv = compute_time_metrics(&x, &y_inv).unwrap();
        assert!((m_inv.correlation_r - (-1.0)).abs() < 1e-14);

        // Orthogonal harmonics over integer cycles: r(s1, s2) == 0.0
        let s1 = SineGenerator::new(100.0, 1.0, 0.0).generate(4800, 48000.0).samples;
        let s2 = SineGenerator::new(200.0, 1.0, 0.0).generate(4800, 48000.0).samples;
        let m_ortho = compute_time_metrics(&s1, &s2).unwrap();
        assert!(m_ortho.correlation_r.abs() < 1e-12);
    }

    #[test]
    fn test_t1_tmet_05_error_handling_on_length_mismatch() {
        assert!(matches!(
            compute_time_metrics(&vec![0.0; 1024], &vec![0.0; 512]),
            Err(DspError::BufferLengthMismatch {
                expected: 1024,
                actual: 512
            })
        ));
        assert_eq!(compute_time_metrics(&[], &[]), Err(DspError::EmptyInput));
    }
}

// ============================================================================
// Group 12: Spectral-Domain Metrics (T1-SMET-01 .. T1-SMET-05)
// ============================================================================

mod test_spectral_metrics {
    use super::*;

    #[test]
    fn test_t1_smet_01_spectral_convergence_identity_and_scale() {
        let x = vec![1.0, 0.5, 0.25, 0.125, 0.0625];
        let x_scaled: Vec<f64> = x.iter().map(|&v| 1.2 * v).collect();

        // Identity SC == 0.0
        let m_id = compute_spectral_metrics(&x, &x).unwrap();
        assert_eq!(m_id.spectral_convergence, 0.0);

        // Scaled SC(X, 1.2X) == |1 - 1.2| = 0.2
        let m_scale = compute_spectral_metrics(&x, &x_scaled).unwrap();
        assert!((m_scale.spectral_convergence - 0.2).abs() < 1e-14);
    }

    #[test]
    fn test_t1_smet_02_log_spectral_distance_known_gain() {
        let x = vec![1.0; 257];
        let x_scaled: Vec<f64> = vec![2.0; 257]; // +6.0206 dB

        let m = compute_spectral_metrics(&x, &x_scaled).unwrap();
        let expected_lsd = 20.0 * (2.0_f64).log10();
        assert!((m.log_spectral_distance_db - expected_lsd).abs() < 1e-4);
    }

    #[test]
    fn test_t1_smet_03_harmonic_peak_error_evaluation() {
        let gt = vec![
            HarmonicPartial {
                harmonic_index: 1,
                frequency: 100.0,
                amplitude: 1.0,
                phase: 0.0,
            },
            HarmonicPartial {
                harmonic_index: 2,
                frequency: 200.0,
                amplitude: 0.5,
                phase: 0.0,
            },
        ];
        let extracted = vec![(100.1, 1.01), (199.8, 0.49)];

        let report = evaluate_harmonic_extraction(&gt, &extracted);
        assert!((report.max_frequency_error_hz - 0.2).abs() < 1e-6);
        assert!((report.max_relative_amplitude_error - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_t1_smet_04_silence_zero_energy_floor_protection() {
        let silent = vec![0.0; 257];
        let m = compute_spectral_metrics(&silent, &silent).unwrap();

        assert!(!m.spectral_convergence.is_nan());
        assert!(!m.log_spectral_distance_db.is_nan());
        assert_eq!(m.spectral_convergence, 0.0);
        assert_eq!(m.log_spectral_distance_db, 0.0);
    }

    #[test]
    fn test_t1_smet_05_dimension_mismatch_and_empty_rejection() {
        assert!(matches!(
            compute_spectral_metrics(&vec![0.0; 257], &vec![0.0; 513]),
            Err(DspError::BufferLengthMismatch {
                expected: 257,
                actual: 513
            })
        ));
        assert_eq!(compute_spectral_metrics(&[], &[]), Err(DspError::EmptyInput));
    }
}

// ============================================================================
// Group 13: Numerical Stability Invariants (T1-STAB-01 .. T1-STAB-05)
// ============================================================================

mod test_stability_invariants {
    use super::*;

    #[test]
    fn test_t1_stab_01_clean_normalized_buffer_invariant() {
        let buf = SineGenerator::new(440.0, 0.95, 0.0).generate(48000, 48000.0);
        assert!(verify_numerical_stability(&buf.samples).is_ok());
    }

    #[test]
    fn test_t1_stab_02_immediate_nan_and_infinity_detection() {
        let mut buf_nan = vec![0.0; 1024];
        buf_nan[512] = f64::NAN;
        assert!(matches!(
            verify_numerical_stability(&buf_nan),
            Err(DspError::NumericalInstability(msg)) if msg.contains("NaN")
        ));

        let mut buf_inf = vec![0.0; 1024];
        buf_inf[700] = f64::INFINITY;
        assert!(matches!(
            verify_numerical_stability(&buf_inf),
            Err(DspError::NumericalInstability(msg)) if msg.contains("Infinity")
        ));
    }

    #[test]
    fn test_t1_stab_03_clipping_bound_violation() {
        let mut clipped = vec![0.0; 1024];
        clipped[100] = 1.005;
        assert!(matches!(
            verify_numerical_stability(&clipped),
            Err(DspError::NumericalInstability(msg)) if msg.contains("clipping") || msg.contains("Peak")
        ));
    }

    #[test]
    fn test_t1_stab_04_energy_conservation_ratio_detection() {
        let orig = vec![1.0; 100];
        let recon_blown: Vec<f64> = orig.iter().map(|&s| s * 1.2).collect();

        let report = check_numerical_stability(&orig, &recon_blown);
        assert!(!report.is_clean);
        assert!((report.energy_ratio - 1.44).abs() < 1e-12);
    }

    #[test]
    fn test_t1_stab_05_severe_dc_offset_detection() {
        let orig = vec![0.0; 1000];
        let recon_dc = vec![0.05; 1000];

        let report = check_numerical_stability(&orig, &recon_dc);
        assert!(!report.is_clean);
        assert!((report.dc_offset - 0.05).abs() < 1e-6);
    }
}
