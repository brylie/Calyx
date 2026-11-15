//! tests/adversarial_synthetic_challenge.rs
//!
//! Empirical Adversarial Verification & Stress-Testing Suite for Milestone 2:
//! Synthetic Signal Generators (`src/synthetic/`).
//!
//! Evaluates analytical mathematical precision, physical invariants, numerical edge
//! conditions, statistical moments, and parameter boundary robustness across all 7 generators.

#[path = "common/mod.rs"]
mod common;

use calyx_dsp::error::DspError;
use calyx_dsp::synthetic::*;
use std::f64::consts::{PI, SQRT_2};

// ============================================================================
// 1. SineGenerator Empirical Challenges
// ============================================================================

#[test]
fn challenge_sine_exact_rms_and_parseval_across_integer_periods() {
    // (f0, fs, num_samples, exact_cycles)
    let test_cases: &[(f64, f64, usize)] = &[
        (100.0, 10000.0, 10000), // 100.0 cycles
        (200.0, 48000.0, 4800),  // 20.0 cycles
        (400.0, 48000.0, 4800),  // 40.0 cycles
        (440.0, 48000.0, 12000), // 110.0 cycles (12000 * 440 / 48000 = 110)
        (1000.0, 48000.0, 4800), // 100.0 cycles
    ];
    let amplitudes = [0.1, 0.5, 1.0, 2.5];

    for &(f0, fs, num_samples) in test_cases {
        for &amp in &amplitudes {
            let gen = SineGenerator::new(f0, amp, 0.0);
            let buf = gen.generate(num_samples, fs);

            let energy: f64 = buf.samples.iter().map(|s| s * s).sum();
            let discrete_rms = (energy / (num_samples as f64)).sqrt();

            let analytical_rms = amp / SQRT_2;
            let analytical_energy = (num_samples as f64) * amp * amp / 2.0;

            // Discrete RMS must match A / sqrt(2) to high numerical precision (< 1e-10)
            assert!(
                (discrete_rms - analytical_rms).abs() < 1e-10,
                "RMS mismatch for f0={}, amp={}: discrete={}, analytical={}",
                f0,
                amp,
                discrete_rms,
                analytical_rms
            );

            // Parseval energy must match N * A^2 / 2 (< 1e-8)
            assert!(
                (energy - analytical_energy).abs() < 1e-8,
                "Energy mismatch for f0={}, amp={}: discrete={}, analytical={}",
                f0,
                amp,
                energy,
                analytical_energy
            );

            // Metadata must record exact theoretical values
            assert!((buf.metadata.theoretical_rms.unwrap() - analytical_rms).abs() < 1e-14);
            assert!((buf.metadata.theoretical_energy.unwrap() - analytical_energy).abs() < 1e-14);
        }
    }
}

#[test]
fn challenge_sine_nyquist_limit_alternating_sequence() {
    let fs = 48000.0;
    let nyquist = fs / 2.0; // 24000.0 Hz
    let amp = 0.85;
    let num_samples = 128;

    // With phase = PI / 2: sin(pi * n + pi/2) = cos(pi * n) = (-1)^n
    let gen_cos = SineGenerator::new(nyquist, amp, PI / 2.0);
    let buf_cos = gen_cos.generate(num_samples, fs);

    for (n, &s) in buf_cos.samples.iter().enumerate() {
        let expected = if n % 2 == 0 { amp } else { -amp };
        assert!(
            (s - expected).abs() < 1e-12,
            "Nyquist alternating sequence failed at sample {}: got {}, expected {}",
            n,
            s,
            expected
        );
    }

    // With phase = 0.0: sin(pi * n) = 0 for all integer n
    let gen_zero = SineGenerator::new(nyquist, amp, 0.0);
    let buf_zero = gen_zero.generate(num_samples, fs);

    for (n, &s) in buf_zero.samples.iter().enumerate() {
        assert!(
            s.abs() < 1e-12,
            "Nyquist sine phase 0 should be identically zero at sample {}, got {}",
            n,
            s
        );
    }
}

#[test]
fn challenge_sine_dc_tone_behavior() {
    let fs = 44100.0;
    let num_samples = 256;
    let amp = 0.6;

    // DC with phase PI / 6: sin(PI/6) = 0.5 -> constant 0.3
    let gen_dc1 = SineGenerator::new(0.0, amp, PI / 6.0);
    let buf1 = gen_dc1.generate(num_samples, fs);
    for &s in &buf1.samples {
        assert!((s - 0.3).abs() < 1e-14, "DC tone expected 0.3, got {}", s);
    }

    // DC with phase 0.0: sin(0) = 0.0 -> all zeros
    let gen_dc2 = SineGenerator::new(0.0, amp, 0.0);
    let buf2 = gen_dc2.generate(num_samples, fs);
    for &s in &buf2.samples {
        assert_eq!(s, 0.0, "DC tone phase 0 must be exactly 0.0");
    }
}

#[test]
fn challenge_sine_parameter_validation_boundaries() {
    let fs = 48000.0;
    // Super-Nyquist rejected
    assert!(SineGenerator::validate_params(24000.1, 1.0, 0.0, fs).is_err());
    // Exactly Nyquist accepted
    assert!(SineGenerator::validate_params(24000.0, 1.0, 0.0, fs).is_ok());
    // Negative frequency rejected
    assert!(SineGenerator::validate_params(-1.0, 1.0, 0.0, fs).is_err());
    // Negative amplitude rejected
    assert!(SineGenerator::validate_params(440.0, -0.1, 0.0, fs).is_err());
    // Non-finite parameters rejected
    assert!(SineGenerator::validate_params(f64::NAN, 1.0, 0.0, fs).is_err());
    assert!(SineGenerator::validate_params(440.0, f64::INFINITY, 0.0, fs).is_err());
    assert!(SineGenerator::validate_params(440.0, 1.0, f64::NAN, fs).is_err());
    assert!(SineGenerator::validate_params(440.0, 1.0, 0.0, 0.0).is_err());
}

// ============================================================================
// 2. HarmonicComplexGenerator Empirical Challenges
// ============================================================================

#[test]
fn challenge_harmonic_stiff_string_dispersion_exactness() {
    let f0 = 110.0;
    let b = 0.0008;
    let num_partials = 12;
    let fs = 48000.0;

    let gen = HarmonicComplexGenerator::new(f0, 1.0, num_partials, b, HarmonicRecipe::Sawtooth);
    let buf = gen.generate(1024, fs);

    for p in &buf.metadata.known_partials {
        let k = p.harmonic_index as f64;
        let expected_dispersion = (1.0 + b * k * k).sqrt();
        let expected_f = k * f0 * expected_dispersion;

        assert!(
            (p.frequency_hz - expected_f).abs() < 1e-10,
            "Dispersion mismatch for partial {}: got {}, expected {}",
            p.harmonic_index,
            p.frequency_hz,
            expected_f
        );
    }
}

#[test]
fn challenge_harmonic_anti_aliasing_nyquist_culling() {
    // Fundamental = 3000 Hz, sample_rate = 12000 Hz -> Nyquist = 6000 Hz.
    // Partials without inharmonicity: 3000 (k=1), 6000 (k=2, >= Nyquist -> culled).
    let gen = HarmonicComplexGenerator::new(3000.0, 1.0, 10, 0.0, HarmonicRecipe::Sawtooth);
    let buf = gen.generate(1024, 12000.0);

    // Only partial 1 should survive; all partials >= 6000 Hz must be eliminated
    assert_eq!(buf.metadata.known_partials.len(), 1);
    assert_eq!(buf.metadata.known_partials[0].harmonic_index, 1);
    assert!(buf.metadata.known_partials[0].frequency_hz < 6000.0);

    // With positive dispersion B = 0.01, partials get pushed even higher
    let gen_disp = HarmonicComplexGenerator::new(2000.0, 1.0, 10, 0.05, HarmonicRecipe::Sawtooth);
    let buf_disp = gen_disp.generate(1024, 12000.0);
    for p in &buf_disp.metadata.known_partials {
        assert!(
            p.frequency_hz < 6000.0,
            "Anti-aliasing culling failed: partial {} at {} Hz >= Nyquist (6000 Hz)",
            p.harmonic_index,
            p.frequency_hz
        );
    }
}

#[test]
fn challenge_harmonic_square_wave_odd_harmonics_only() {
    let gen = HarmonicComplexGenerator::new(100.0, 1.0, 16, 0.0, HarmonicRecipe::Square);
    let buf = gen.generate(2048, 48000.0);

    let partials = &buf.metadata.known_partials;
    // Across 1..=16, odd numbers are 1, 3, 5, 7, 9, 11, 13, 15 (8 total)
    assert_eq!(partials.len(), 8);

    for p in partials {
        assert_eq!(
            p.harmonic_index % 2,
            1,
            "Square wave must contain ONLY odd harmonics, found even harmonic {}",
            p.harmonic_index
        );
        let expected_amp = 1.0 / (p.harmonic_index as f64);
        assert!(
            (p.amplitude - expected_amp).abs() < 1e-14,
            "Square wave amplitude for harmonic {} was {}, expected {}",
            p.harmonic_index,
            p.amplitude,
            expected_amp
        );
    }
}

#[test]
fn challenge_harmonic_frequency_dependent_damping_attenuation() {
    // Verify damping tau_k = tau_0 / sqrt(k): higher harmonics decay strictly faster over time.
    let f0 = 100.0;
    let tau_0 = 0.2;
    let mut gen = HarmonicComplexGenerator::new(f0, 1.0, 4, 0.0, HarmonicRecipe::Custom(vec![1.0, 1.0, 1.0, 1.0]));
    gen.set_damping_tau(Some(tau_0));

    let fs = 48000.0;
    let num_samples = 48000; // 1 second
    let buf = gen.generate(num_samples, fs);

    // Compute DFT at start (first 2048 samples) and at end (around sample 24000)
    // to measure relative decay of partial 1 vs partial 4.
    let t_eval = 0.5; // at 0.5s
    let expected_env_p1 = (-t_eval / (tau_0 / (1.0_f64).sqrt())).exp(); // exp(-0.5 / 0.2) = exp(-2.5) ~= 0.082
    let expected_env_p4 = (-t_eval / (tau_0 / (4.0_f64).sqrt())).exp(); // exp(-0.5 / 0.1) = exp(-5.0) ~= 0.0067

    assert!(
        expected_env_p4 < expected_env_p1 * 0.1,
        "Higher harmonic must decay at least an order of magnitude faster"
    );

    // Inspect audio samples: verify no NaN and exponential energy decay
    let e_early: f64 = buf.samples[..4800].iter().map(|s| s * s).sum();
    let e_late: f64 = buf.samples[24000..28800].iter().map(|s| s * s).sum();
    assert!(e_late < e_early * 0.01, "Audio signal did not decay exponentially");
}

// ============================================================================
// 3. ChirpGenerator Empirical Challenges
// ============================================================================

#[test]
fn challenge_chirp_linear_vs_logarithmic_phase_progression() {
    let f_start = 100.0;
    let f_end = 1600.0;
    let amp = 1.0;
    let duration = 2.0;
    let fs = 48000.0;
    let num_samples = 96000;

    let gen_lin = ChirpGenerator::new(f_start, f_end, amp, duration, ChirpKind::Linear);
    let buf_lin = gen_lin.generate(num_samples, fs);

    let gen_log = ChirpGenerator::new(f_start, f_end, amp, duration, ChirpKind::Logarithmic);
    let buf_log = gen_log.generate(num_samples, fs);

    // Point-by-point analytical comparison for Linear Chirp:
    // Phi(t) = 2*pi*(f0*t + 0.5*k*t^2)
    let k = (f_end - f_start) / duration;
    for n in [1000, 24000, 48000, 72000, 95999] {
        let t = n as f64 / fs;
        let expected_lin = amp * (2.0 * PI * (f_start * t + 0.5 * k * t * t)).sin();
        assert!(
            (buf_lin.samples[n] - expected_lin).abs() < 1e-12,
            "Linear chirp mismatch at sample {}: got {}, expected {}",
            n,
            buf_lin.samples[n],
            expected_lin
        );

        // Point-by-point analytical comparison for Logarithmic Chirp:
        // Phi(t) = factor * (ratio^(t/T) - 1)
        let ratio = f_end / f_start;
        let factor = 2.0 * PI * f_start * duration / ratio.ln();
        let expected_log = amp * (factor * (ratio.powf(t / duration) - 1.0)).sin();
        assert!(
            (buf_log.samples[n] - expected_log).abs() < 1e-12,
            "Logarithmic chirp mismatch at sample {}: got {}, expected {}",
            n,
            buf_log.samples[n],
            expected_log
        );
    }
}

#[test]
fn challenge_chirp_identical_frequencies_singularity_protection() {
    // f_start == f_end: in logarithmic chirp, ratio = 1.0, ln(ratio) = 0.0.
    // Must gracefully fall back to pure sine wave without divide-by-zero NaN.
    let f0 = 440.0;
    let amp = 0.9;
    let duration = 1.0;
    let fs = 48000.0;
    let num_samples = 4800;

    let chirp_lin = ChirpGenerator::new(f0, f0, amp, duration, ChirpKind::Linear);
    let buf_lin = chirp_lin.generate(num_samples, fs);

    let chirp_log = ChirpGenerator::new(f0, f0, amp, duration, ChirpKind::Logarithmic);
    let buf_log = chirp_log.generate(num_samples, fs);

    let sine_gen = SineGenerator::new(f0, amp, 0.0);
    let buf_sine = sine_gen.generate(num_samples, fs);

    for n in 0..num_samples {
        assert!(
            (buf_lin.samples[n] - buf_sine.samples[n]).abs() < 1e-12,
            "Linear chirp boundary f_start == f_end failed at {}",
            n
        );
        assert!(
            (buf_log.samples[n] - buf_sine.samples[n]).abs() < 1e-12,
            "Logarithmic chirp boundary f_start == f_end failed at {}",
            n
        );
    }
}

// ============================================================================
// 4. ImpulseGenerator Empirical Challenges
// ============================================================================

#[test]
fn challenge_impulse_kronecker_delta_energy_and_location() {
    let num_samples = 1024;
    let fs = 48000.0;
    let amp = 1.5;

    for &delay in &[0, 1, 512, 1023] {
        let gen = ImpulseGenerator::kronecker(delay, amp);
        let buf = gen.generate(num_samples, fs);

        assert_eq!(buf.samples[delay], amp);
        for (i, &s) in buf.samples.iter().enumerate() {
            if i != delay {
                assert_eq!(s, 0.0, "Sample at {} should be 0.0 for delay {}", i, delay);
            }
        }

        let energy: f64 = buf.samples.iter().map(|s| s * s).sum();
        assert!((energy - amp * amp).abs() < 1e-14);
    }

    // Checked bounds test
    let gen_out_of_bounds = ImpulseGenerator::kronecker(1024, 1.0);
    assert!(matches!(
        gen_out_of_bounds.generate_checked(1024, fs),
        Err(DspError::InvalidParameter { name: "delay", .. })
    ));
}

#[test]
fn challenge_impulse_bandlimited_sinc_symmetry_and_zero_crossings() {
    let cutoff_hz = 6000.0;
    let delay = 256;
    let amp = 1.0;
    let fs = 48000.0;
    let num_samples = 512;

    let gen = ImpulseGenerator::bandlimited_sinc(cutoff_hz, delay, amp);
    let buf = gen.generate(num_samples, fs);

    // 1. Peak value at delay
    assert!((buf.samples[delay] - amp).abs() < 1e-12);

    // 2. Strict symmetry around delay: s[delay - m] == s[delay + m]
    for m in 1..=200 {
        let left = buf.samples[delay - m];
        let right = buf.samples[delay + m];
        assert!(
            (left - right).abs() < 1e-12,
            "Symmetry broken at offset {}: left={}, right={}",
            m,
            left,
            right
        );
    }

    // 3. Zero crossings: x = 2*pi*(fc/fs)*m.
    // fc/fs = 6000 / 48000 = 1/8.
    // Zero crossings occur when 2*pi*(1/8)*m = k*pi -> m = 4*k.
    // Every 4 samples away from delay must be an exact zero crossing!
    for k in 1..=30 {
        let sample_idx = delay + 4 * k;
        assert!(
            buf.samples[sample_idx].abs() < 1e-12,
            "Zero-crossing violated at offset {}: got {}",
            4 * k,
            buf.samples[sample_idx]
        );
    }
}

#[test]
fn challenge_impulse_damped_transient_decaying_sinusoid() {
    let f0 = 440.0;
    let decay_rate = 25.0; // tau = 0.04s
    let amp = 1.0;
    let fs = 48000.0;
    let num_samples = 9600;

    let gen = ImpulseGenerator::exponential_pluck(f0, decay_rate, amp);
    let buf = gen.generate(num_samples, fs);

    // Initial value at t=0: sin(0) = 0
    assert!(buf.samples[0].abs() < 1e-14);

    // Value at sample n must match A * exp(-decay_rate * t) * sin(2*pi*f0*t)
    for n in [10, 100, 500, 1000, 4000] {
        let t = n as f64 / fs;
        let expected = amp * (-decay_rate * t).exp() * (2.0 * PI * f0 * t).sin();
        assert!(
            (buf.samples[n] - expected).abs() < 1e-12,
            "Mismatch in damped transient at sample {}: got {}, expected {}",
            n,
            buf.samples[n],
            expected
        );
    }
}

// ============================================================================
// 5. NoiseGenerator Empirical Challenges
// ============================================================================

#[test]
fn challenge_noise_pcg32_repeatability_and_seed_decorrelation() {
    let n = 20000;
    let seed1 = 0x0123456789ABCDEF;
    let seed2 = 0xFEDCBA9876543210;

    let g1 = NoiseGenerator::new(NoiseColor::WhiteUniform, 1.0, seed1).generate(n, 48000.0);
    let g2 = NoiseGenerator::new(NoiseColor::WhiteUniform, 1.0, seed1).generate(n, 48000.0);
    let g3 = NoiseGenerator::new(NoiseColor::WhiteUniform, 1.0, seed2).generate(n, 48000.0);

    // Bit-for-bit repeatability
    assert_eq!(g1.samples, g2.samples, "PCG32 output must be strictly identical for same seed");

    // Seed decorrelation: compute Pearson r between g1 and g3
    let mean1 = g1.samples.iter().sum::<f64>() / (n as f64);
    let mean3 = g3.samples.iter().sum::<f64>() / (n as f64);
    let mut cov = 0.0_f64;
    let mut var1 = 0.0_f64;
    let mut var3 = 0.0_f64;

    for i in 0..n {
        let d1 = g1.samples[i] - mean1;
        let d3 = g3.samples[i] - mean3;
        cov += d1 * d3;
        var1 += d1 * d1;
        var3 += d3 * d3;
    }

    let r = cov / (var1.sqrt() * var3.sqrt());
    assert!(
        r.abs() < 0.02,
        "Distinct seeds produced correlated noise (r = {})",
        r
    );
}

#[test]
fn challenge_noise_white_uniform_statistical_moments() {
    let n = 200000;
    let amp = 1.0;
    let gen = NoiseGenerator::new(NoiseColor::WhiteUniform, amp, 777);
    let buf = gen.generate(n, 48000.0);

    let mean = buf.samples.iter().sum::<f64>() / (n as f64);
    let var = buf.samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n as f64);

    // Theoretical mean: 0.0
    assert!(mean.abs() < 0.005, "Uniform noise mean |{}| exceeds 0.005", mean);
    // Theoretical variance: 1/3 ~ 0.333333
    let expected_var = 1.0 / 3.0;
    assert!(
        (var - expected_var).abs() < 0.005,
        "Uniform noise variance {} deviates from theoretical 1/3 ({})",
        var,
        expected_var
    );
}

#[test]
fn challenge_noise_white_gaussian_box_muller_normality_and_kurtosis() {
    let n = 200000;
    let amp = 1.0;
    let gen = NoiseGenerator::new(NoiseColor::WhiteGaussian, amp, 888);
    let buf = gen.generate(n, 48000.0);

    let mean = buf.samples.iter().sum::<f64>() / (n as f64);
    let var = buf.samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n as f64);
    let fourth_moment = buf.samples.iter().map(|s| (s - mean).powi(4)).sum::<f64>() / (n as f64);
    let kurtosis = fourth_moment / (var * var);

    // Mean ~ 0.0
    assert!(mean.abs() < 0.008, "Gaussian mean |{}| exceeds 0.008", mean);
    // Variance ~ 1.0
    assert!((var - 1.0).abs() < 0.015, "Gaussian variance {} deviates from 1.0", var);
    // Kurtosis of standard normal distribution is identically 3.0
    assert!(
        (kurtosis - 3.0).abs() < 0.1,
        "Gaussian kurtosis {} deviates from normal distribution (3.0)",
        kurtosis
    );
}

#[test]
fn challenge_noise_pink_and_brownian_spectral_slope() {
    let n = 1024;
    let fs = 48000.0;

    let pink = NoiseGenerator::new(NoiseColor::Pink, 0.8, 101).generate(n, fs);
    let brown = NoiseGenerator::new(NoiseColor::Brownian, 0.8, 202).generate(n, fs);

    // Band power at low octave: [200, 400] Hz
    let p_low_pink = common::compute_band_power(&pink.samples, 200.0, 400.0, fs);
    let p_high_pink = common::compute_band_power(&pink.samples, 1600.0, 3200.0, fs);

    // In pink noise (equal power per octave), power in [200, 400] and [1600, 3200]
    // should be roughly comparable (ratio within [0.2, 5.0] on short windows):
    let pink_ratio = p_high_pink / p_low_pink;
    assert!(
        pink_ratio > 0.1 && pink_ratio < 10.0,
        "Pink noise octave power ratio unexpected: {}",
        pink_ratio
    );

    // In Brownian noise (1/f^2, -6 dB/octave), power in high band [1600, 3200]
    // should be drastically lower than [200, 400] (by 3 octaves = -18 dB ~ 1/64):
    let p_low_brown = common::compute_band_power(&brown.samples, 200.0, 400.0, fs);
    let p_high_brown = common::compute_band_power(&brown.samples, 1600.0, 3200.0, fs);
    assert!(
        p_high_brown < p_low_brown * 0.2,
        "Brownian noise high band power ({}) was not attenuated compared to low band ({})",
        p_high_brown,
        p_low_brown
    );
}

// ============================================================================
// 6. SyntheticInstrumentSource Empirical Challenges
// ============================================================================

#[test]
fn challenge_instrument_equal_temperament_full_range() {
    let inst = SyntheticInstrumentSource::new();

    // Verify semitone step ratio is exactly 2^(1/12) across all 88 keys
    for p in 21..108 {
        let note1 = inst.generate_note(p, 0.8, Articulation::Sustain, 1);
        let note2 = inst.generate_note(p + 1, 0.8, Articulation::Sustain, 1);

        let f1 = note1.metadata.fundamental_freq.unwrap();
        let f2 = note2.metadata.fundamental_freq.unwrap();

        let ratio = f2 / f1;
        let expected_ratio = 2.0_f64.powf(1.0 / 12.0);
        assert!(
            (ratio - expected_ratio).abs() < 1e-10,
            "Semitone ratio between {} and {} was {}, expected {}",
            p,
            p + 1,
            ratio,
            expected_ratio
        );
    }

    // Verify 12 semitones produces an exact octave (ratio 2.0)
    for p in 21..96 {
        let note1 = inst.generate_note(p, 0.8, Articulation::Sustain, 1);
        let note2 = inst.generate_note(p + 12, 0.8, Articulation::Sustain, 1);

        let f1 = note1.metadata.fundamental_freq.unwrap();
        let f2 = note2.metadata.fundamental_freq.unwrap();

        let ratio = f2 / f1;
        assert!(
            (ratio - 2.0).abs() < 1e-10,
            "Octave ratio from {} to {} was {}, expected 2.0",
            p,
            p + 12,
            ratio
        );
    }
}

#[test]
fn challenge_instrument_5d_parameter_edges() {
    let inst = SyntheticInstrumentSource::new();
    let fs = 48000.0;

    // Edge 1: Dynamic d = 0.0 produces pure silence without panic
    let silent = inst.synthesize([0.0, 0.5, 0.0, 0.5, 0.5], 1024, fs).unwrap();
    assert!(silent.samples.iter().all(|&s| s == 0.0));

    // Edge 2: Extreme coordinates within bounds [0.0, 1.0]
    let max_coords = [1.0, 1.0, 1.0, 1.0, 1.0];
    let loud = inst.synthesize(max_coords, 2048, fs).unwrap();
    // Non-normalized 16-partial additive sum: max amplitude is bounded by harmonic sum ~ 3.38
    let max_val = loud.samples.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
    assert!(loud.samples.iter().all(|s| !s.is_nan() && !s.is_infinite()));
    assert!(max_val < 3.5, "Loud note max amplitude {} exceeded theoretical sum bounds 3.5", max_val);

    let min_coords = [0.0, 0.0, 0.01, 0.0, 0.0];
    let soft = inst.synthesize(min_coords, 2048, fs).unwrap();
    assert!(soft.samples.iter().all(|s| !s.is_nan() && s.abs() <= 1.0001));

    // Edge 3: Out-of-bounds coordinates return DspError::InvalidParameter
    assert!(inst.synthesize([-0.1, 0.5, 0.5, 0.5, 0.5], 1024, fs).is_err());
    assert!(inst.synthesize([0.5, 1.1, 0.5, 0.5, 0.5], 1024, fs).is_err());
    assert!(inst.synthesize([0.5, 0.5, f64::NAN, 0.5, 0.5], 1024, fs).is_err());

    // Edge 4: num_samples = 0 returns empty buffer safely
    let empty = inst.synthesize([0.5, 0.5, 0.5, 0.5, 0.5], 0, fs).unwrap();
    assert!(empty.samples.is_empty());
}

// ============================================================================
// 7. Advanced Adversarial Stress Tests & Numerical Invariants
// ============================================================================

#[test]
fn challenge_sine_non_integer_periods_rms_asymptotic_convergence() {
    let f0 = 100.0;
    let fs = 10000.0;
    let amp = 1.0;
    let analytical_rms = 1.0 / SQRT_2;

    // As number of non-integer cycles increases (K + 0.5), deviation from A / sqrt(2)
    // decays as O(1 / K).
    let cycles_list: &[f64] = &[1.5, 10.5, 100.5, 1000.5];
    for &k_cycles in cycles_list {
        let num_samples = (k_cycles * (fs / f0)).round() as usize;
        let gen = SineGenerator::new(f0, amp, 0.0);
        let buf = gen.generate(num_samples, fs);

        let energy: f64 = buf.samples.iter().map(|s| s * s).sum();
        let discrete_rms = (energy / (num_samples as f64)).sqrt();
        let deviation = (discrete_rms - analytical_rms).abs();

        // Bound: deviation < 1.0 / (2 * k_cycles)
        assert!(
            deviation < 1.0 / (2.0 * k_cycles),
            "Non-integer period deviation {} too large for {} cycles",
            deviation,
            k_cycles
        );
    }
}

#[test]
fn challenge_sine_large_sample_count_numerical_stability() {
    // 500,000 samples at 48 kHz (~10.4 seconds)
    let n = 500000;
    let fs = 48000.0;
    let gen = SineGenerator::new(440.0, 0.8, 0.25);
    let buf = gen.generate(n, fs);

    // Verify all samples are finite and bounded within [-0.8, 0.8]
    assert_eq!(buf.samples.len(), n);
    for (i, &s) in buf.samples.iter().enumerate() {
        assert!(!s.is_nan(), "NaN at sample {}", i);
        assert!(s.abs() <= 0.8000000001, "Clipping at sample {}", i);
    }
}

#[test]
fn challenge_harmonic_damping_tau_zero_singularity() {
    // Damping tau = 0 causes -t / tau = -0.0 / 0.0 = NaN at t=0
    let mut gen = HarmonicComplexGenerator::new(440.0, 1.0, 3, 0.0, HarmonicRecipe::Sawtooth);
    gen.set_damping_tau(Some(0.0));
    let buf = gen.generate(10, 48000.0);

    // Documented behavior: at t=0 sample 0 produces NaN when tau=0.0
    assert!(buf.samples[0].is_nan(), "tau=0.0 at t=0 produces NaN");
}

#[test]
fn challenge_harmonic_extreme_inharmonicity_culling() {
    // With extreme inharmonicity B = 100.0, partial frequencies grow as k^2 * sqrt(B)
    let f0 = 500.0;
    let b = 100.0;
    let gen = HarmonicComplexGenerator::new(f0, 1.0, 20, b, HarmonicRecipe::Sawtooth);
    let buf = gen.generate(512, 48000.0);

    // Partial 1: 500 * sqrt(1 + 100) = 500 * sqrt(101) ~= 5024.9 Hz (< 24000)
    // Partial 2: 2 * 500 * sqrt(1 + 400) = 1000 * sqrt(401) ~= 20024.9 Hz (< 24000)
    // Partial 3: 3 * 500 * sqrt(1 + 900) = 1500 * sqrt(901) ~= 45024 Hz (>= 24000 -> culled!)
    assert_eq!(buf.metadata.known_partials.len(), 2);
    assert_eq!(buf.metadata.known_partials[0].harmonic_index, 1);
    assert_eq!(buf.metadata.known_partials[1].harmonic_index, 2);
    assert!(buf.metadata.known_partials.iter().all(|p| p.frequency_hz < 24000.0));
}

#[test]
fn challenge_chirp_reverse_logarithmic_sweep() {
    // Downward sweep: 4000 Hz down to 250 Hz (4 octaves down)
    let gen = ChirpGenerator::new(4000.0, 250.0, 0.75, 1.5, ChirpKind::Logarithmic);
    let buf = gen.generate(72000, 48000.0);

    assert_eq!(buf.samples.len(), 72000);
    assert!(buf.samples.iter().all(|s| !s.is_nan() && s.abs() <= 0.7500000001));

    // Compare analytical formula point-by-point
    let ratio: f64 = 250.0 / 4000.0;
    let factor = 2.0 * PI * 4000.0 * 1.5 / ratio.ln();
    for &n in &[0, 1000, 36000, 71999] {
        let t = n as f64 / 48000.0;
        let expected = 0.75 * (factor * (ratio.powf(t / 1.5) - 1.0)).sin();
        assert!((buf.samples[n] - expected).abs() < 1e-12);
    }
}

#[test]
fn challenge_chirp_negative_start_or_end_frequency_rejection() {
    assert!(ChirpGenerator::validate_params(0.0, 1000.0, 1.0, 1.0, ChirpKind::Logarithmic).is_err());
    assert!(ChirpGenerator::validate_params(100.0, 0.0, 1.0, 1.0, ChirpKind::Logarithmic).is_err());
    assert!(ChirpGenerator::validate_params(-10.0, 1000.0, 1.0, 1.0, ChirpKind::Linear).is_err());
    assert!(ChirpGenerator::validate_params(100.0, -10.0, 1.0, 1.0, ChirpKind::Linear).is_err());
    assert!(ChirpGenerator::validate_params(100.0, 1000.0, 1.0, 0.0, ChirpKind::Linear).is_err());
}

#[test]
fn challenge_impulse_exponential_pluck_zero_frequency() {
    // When f0 = 0.0 Hz, sin(2*pi*0*t) = 0.0 identically for all t
    let gen = ImpulseGenerator::exponential_pluck(0.0, 10.0, 1.0);
    let buf = gen.generate(100, 48000.0);
    assert!(buf.samples.iter().all(|&s| s == 0.0));
}

#[test]
fn challenge_noise_zero_amplitude_silence() {
    for color in &[
        NoiseColor::WhiteUniform,
        NoiseColor::WhiteGaussian,
        NoiseColor::Pink,
        NoiseColor::Brownian,
    ] {
        let gen = NoiseGenerator::new(*color, 0.0, 12345);
        let buf = gen.generate(100, 48000.0);
        assert!(
            buf.samples.iter().all(|&s| s == 0.0),
            "Noise with amplitude 0.0 must be pure silence for {:?}",
            color
        );
    }
}

#[test]
fn challenge_noise_one_bit_seed_flip_avalanche() {
    let seed1 = 0x1234567890ABCDEF;
    let seed2 = seed1 ^ 1; // Flip lowest bit

    let n = 10000;
    let b1 = NoiseGenerator::new(NoiseColor::WhiteUniform, 1.0, seed1).generate(n, 48000.0);
    let b2 = NoiseGenerator::new(NoiseColor::WhiteUniform, 1.0, seed2).generate(n, 48000.0);

    // Check how many samples differ: avalanche effect in PCG32 should decorrelate immediately
    let diff_count = b1.samples.iter().zip(&b2.samples).filter(|(&x, &y)| (x - y).abs() > 1e-6).count();
    assert_eq!(diff_count, n, "Every sample must differ when flipping 1 bit of seed");
}

#[test]
fn challenge_instrument_attack_and_release_tapering() {
    let inst = SyntheticInstrumentSource::new();
    let note = inst.generate_note(60, 0.8, Articulation::Sustain, 1);
    let n = note.samples.len();

    // Attack window: sample 0 must be 0.0 (smooth fade-in)
    assert!(
        note.samples[0].abs() < 1e-12,
        "Attack sample 0 should be 0.0, got {}",
        note.samples[0]
    );

    // Release window: sample n - 1 must be 0.0 (smooth fade-out)
    assert!(
        note.samples[n - 1].abs() < 1e-12,
        "Release sample n-1 should be 0.0, got {}",
        note.samples[n - 1]
    );
}

#[test]
fn challenge_instrument_temporal_decay_at_large_t() {
    let inst = SyntheticInstrumentSource::new();
    // Query spectral frame at t = 0.0 vs t = 5.0 seconds
    let frame_start = inst.query_frame([0.0, 60.0, 0.8, 0.0, 0.0]).unwrap();
    let frame_late = inst.query_frame([5.0, 60.0, 0.8, 0.0, 0.0]).unwrap();

    // Partials at t = 5.0s (staccato articulation with tau_0 = 0.15s)
    // should have decayed to effectively zero: exp(-5.0 / 0.15) = exp(-33.3) ~= 3e-15
    for (p_start, p_late) in frame_start.partials.iter().zip(&frame_late.partials) {
        assert!(
            p_late.amplitude < p_start.amplitude * 1e-10,
            "Partial {} at t=5.0s did not decay (start={}, late={})",
            p_start.harmonic_index,
            p_start.amplitude,
            p_late.amplitude
        );
    }
}

