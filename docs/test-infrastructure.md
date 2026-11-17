# Calyx DSP Framework: Test Infrastructure Specification (`TEST_INFRA.md`)

**Document Version:** 1.1.0  
**Status:** Active Specification (Gated on M1-M4 Implementation; M5 Validation)  
**Date:** 2026-11-14  
**Author:** Calyx E2E Testing Track (`sub_orch_e2e_testing`)  
**Scope:** Core User Requirements R1 (Rust Experimental DSP Framework & Modular Interfaces) and R2 (Synthetic Signal Generation and Automated Verification)

---

## 1. Executive Summary & Infrastructure Architecture

The Calyx DSP testing infrastructure provides an automated, requirement-driven, opaque-box verification framework for the Calyx digital signal processing architecture. Built under **Demo Integrity Mode**, the test infrastructure operates with **strictly zero external crate dependencies for core DSP functionality**—relying entirely on the safe Rust standard library (`std`).

All audio signals used for verification are synthesized deterministically in memory via analytical mathematical models (pure tones, additive harmonic series, inharmonic piano wire models, continuous frequency chirps, bandlimited sinc pulses, and seedable PCG32 pseudo-random noise). No external `.wav` or `.flac` test vectors or filesystem dependencies are required, guaranteeing 100% reproducible, deterministic in-memory execution. Full test suite execution under `cargo test` targets completion in under 15 seconds once all prerequisite implementation milestones (M1 through M4) are compiled.

### 1.1 Test Suite Directory Layout
```
/Users/brylie/code/Calyx/
├── TEST_INFRA.md                       # Test infrastructure, methodologies, and metric specifications
├── TEST_READY.md                       # Test suite readiness, execution schedule, and coverage checklist
└── tests/
    ├── common/                         # Shared test harness, analytical oracles, and assertion macros
    │   └── mod.rs
    ├── tier1_feature_tests.rs          # 65 opaque-box feature verification tests (>=5 per feature group)
    ├── tier2_boundary_tests.rs         # 150 boundary and corner tests across 10 core domains (verified via #[test] count)
    ├── tier3_interaction_tests.rs      # 16 canonical orthogonal covering array pairwise interaction tests
    └── tier4_application_tests.rs      # 4 real-world sampled acoustic instrument application scenarios
```

---

## 2. Test Philosophy

The Calyx DSP testing philosophy is governed by five foundational engineering principles:

### 2.1 Opaque-Box Verification Against Public Interface Contracts
All tests interact strictly with public structs, enums, functions, and traits exposed by `calyx_dsp` as defined in `PROJECT.md § Interface Contracts`:
- `calyx_dsp::error`: `DspError`, `DspResult<T>`
- `calyx_dsp::math`: `Complex<f64>`, `WindowType`, `generate_window`, `verify_cola`, `Matrix<f64>`, `Tensor<T>`
- `calyx_dsp::synthetic`: `SignalGenerator`, `SignalBuffer`, `SignalMetadata`, `SineGenerator`, `HarmonicComplexGenerator`, `ChirpGenerator`, `ImpulseGenerator`, `NoiseGenerator`, `SyntheticInstrumentSource`
- `calyx_dsp::metrics`: `compute_time_metrics`, `compute_spectral_metrics`, `verify_numerical_stability`
- `calyx_dsp::spectral`: `Dft`, `Fft`, `Stft`, `HarmonicPartial`, `SpectralFrame`, `SpectralTrajectory`, `SpectralAnalyzer`
- `calyx_dsp::hypercube`: `HypercubeDecomposer`, `CompressedHypercube`, `PCA`, `NMF`, `TuckerDecomposition`
- `calyx_dsp::synthesis`: `AdditiveOscillatorBank`, `OverlapAdd`, `SignalReconstructor`
- `calyx_dsp::pipeline`: `DspPipeline`, `DspStage`, `PipelineResult`

Tests never access crate-private functions or internal memory layout. Algorithms may be refactored, optimized, or swapped without breaking the test suite.

### 2.2 Mathematical Analytical Ground Truth
Every test assertion compares algorithmic results against closed-form mathematical equations rather than subjective or heuristic approximations:
- Pure sinusoidal tones: $s[n] = A \sin(2\pi f_0 n / f_s + \phi)$ with analytical $\text{RMS} = A / \sqrt{2}$.
- Inharmonic stiff strings: $f_k = k f_0 \sqrt{1 + B k^2}$ with frequency-dependent decay $\tau_k = \tau_0 / \sqrt{k}$.
- Parseval's Energy Conservation: $\sum_{n=0}^{N-1} |x[n]|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |X[k]|^2$.
- Constant Overlap-Add (COLA) Condition: $\sum_{m=-\infty}^{\infty} w[n - mR] = \text{constant}$.
- Bitwise PRNG Reproducibility: Deterministic PCG32 state transitions.

### 2.3 Dual-Domain Quantitative Evaluation
Evaluating audio reconstruction fidelity exclusively in the time domain is insufficient: linear phase shifts or dispersive phase all-pass filters preserve acoustic timbre while degrading time-domain Signal-to-Noise Ratio (SNR) toward 0 dB. Conversely, spectral-only metrics can mask transient smearing. The test harness mandates dual-domain evaluation:
1. **Time Domain**: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Signal-to-Noise Ratio (SNR in dB), Peak Signal-to-Noise Ratio (PSNR in dB), and Normalized Cross-Correlation ($r$).
2. **Spectral Domain**: Spectral Convergence (SC), Log-Spectral Distance (LSD in dB), and Sub-Bin Harmonic Peak Tracking Error ($\Delta f$ in Hz, $\Delta A$ relative %).

### 2.4 Strict Numerical Stability Invariants
Numerical stability is enforced through zero-tolerance invariant assertions on every output signal:
- `nan_count == 0`: No `NaN` values permitted under any circumstances.
- `inf_count == 0`: No positive or negative `Infinity` values permitted.
- `subnormal_count == 0`: No floating-point denormals that degrade CPU instruction throughput.
- Full-scale bounds: $\max |s[n]| \le 1.0001$ (allowing $0.0001$ margin for floating-point rounding near 0 dBFS).
- Energy conservation: $0.85 \le E_{\text{recon}} / E_{\text{orig}} \le 1.15$ (prohibiting resonant explosions or unnatural attenuation).
- DC offset constraint: $|\frac{1}{N} \sum s[n]| < 10^{-4}$ for zero-mean AC waveforms.

### 2.5 Progressive Tiered Coverage Hierarchy
1. **Tier 1 (Feature Unit Tests)**: Verifies functional correctness of individual features against mathematical specs ($\ge 5$ tests per feature group, 65 total).
2. **Tier 2 (Boundary & Corner Cases)**: Probes mathematical singularities, empty inputs, extreme frequencies, and parameter violations ($\ge 5$ tests per boundary domain, 150 total).
3. **Tier 3 (Pairwise Subsystem Interactions)**: Validates dataflow transitions between adjacent DSP stages across an orthogonal covering array of 16 canonical configurations.
4. **Tier 4 (Real-World Application Scenarios)**: Exercises authentic end-to-end multi-dimensional acoustic instrument synthesis, inharmonic piano string physics, expressive gestures, and quality tier trade-offs (4 scenarios).

---

## 3. Formal Test Methodologies

### 3.1 Category-Partition Method (Ostrand & Balcer)
Each DSP module's input domain is partitioned into disjoint equivalence classes to maximize coverage without redundant permutations:

| Parameter Category | Equivalence Partitions |
|---|---|
| **Sample Rate ($f_s$)** | $44,100\text{ Hz}$ (Standard Audio), $48,000\text{ Hz}$ (Studio / Video), $96,000\text{ Hz}$ (High-Resolution) |
| **Buffer Length ($N$)** | Power-of-two ($256, 512, 1024, 2048$), Arbitrary non-power-of-two ($300, 500, 777, 1000$) |
| **Window Type** | Hann, Hamming, Blackman, Bartlett, Rectangular |
| **Hop Ratio ($H/N$)** | $1/4$ ($75\%$ overlap - Hann COLA), $1/2$ ($50\%$ overlap - Bartlett/Hamming COLA), $1/1$ (Non-overlapping) |
| **Decomposition Mode** | PCA / SVD, Lee-Seung NMF, Tucker Tensor Decomposition (HOSVD), CP-ALS |
| **Compression Rank ($R$)** | Minimal ($R=1$), Low ($R=2$), Balanced ($R=4, 8$), Reference ($R=16$) |
| **Signal Amplitude ($A$)** | Silence ($A=0$), Pianissimo ($A=0.01$), Nominal ($A=0.8$), Full Scale ($A=1.0$), Overdriven ($A=1.05$) |
| **Signal Frequency ($f$)** | DC ($0\text{ Hz}$), Sub-bass ($20\text{ Hz}$), Mid-range ($440\text{ Hz}$), High-frequency ($10\text{ kHz}$), Nyquist ($f_s/2$), Super-Nyquist ($>f_s/2$) |

### 3.2 Boundary Value Analysis (BVA)
Boundary value testing probes behavior at the extreme limits of valid and invalid domains:
- **Frequency Boundaries**: $f = 0.0\text{ Hz}$ (DC limit), $f = 0.01\text{ Hz}$ (phase monotonicity), $f = f_s/2 - \epsilon$ (sub-Nyquist), $f = f_s/2$ (exact Nyquist), $f > f_s/2$ (aliasing rejection).
- **Time/Sample Boundaries**: $N = 0$ (`DspError::EmptyInput`), $N = 1$ (degenerate scalar), $N = 65,536$ (large buffer).
- **Hop Size Boundaries**: $H = 0$ (`DspError::InvalidParameter`), $H = 1$ (maximum sliding density), $H = N$ (critical non-overlap boundary), $H > N$ (gaps in coverage).
- **Matrix/Tensor Boundaries**: $0 \times 0$ (empty error), $1 \times 1$ (scalar matrix), rank-1 outer products, collinear rows (ill-conditioned Gram matrices).
- **Dynamic Headroom**: $s[n] = 1.0$ (0 dBFS boundary), $s[n] = 1.0002$ (clipping invariant trigger), $s[n] = 10^{-14}$ (denormal floor).

### 3.3 Pairwise Combinatorial Testing (All-Pairs Orthogonal Covering Arrays)
Testing all combinations of 6 factors with 2 to 4 levels would require over $1,000$ individual tests. The E2E test suite adopts an orthogonal covering array of 16 canonical configurations that guarantees every possible 2-way parameter interaction between components is exercised at least once.

### 3.4 Real-World Acoustic Workload and Scenario Testing
Tier 4 exercises complete musical production workloads:
- 5-dimensional acoustic note spaces $(t, p, d, a, v)$ spanning 5 pitch registers, 4 dynamic layers, sustain/staccato articulations, and 2 round-robin variations.
- Continuous multi-second gestures featuring simultaneous glissando pitch sweeps, tremolo amplitude modulation, and dynamic filter cutoff tracking.
- Real-time synthesis performance: memory allocations are bounded and frame processing throughput is verified.

---

## 4. Feature Inventory & Test Tier Mapping

All 30 features from `PROJECT.md` are mapped to their primary test tiers and milestone delivery gates:

| Feature # | Feature Name | Description | Milestone Gate | Primary Tier | Target Test File |
|---|---|---|---|---|---|
| **F01** | `Complex<T>` Primitives | Cartesian/Polar representation, arithmetic, norm, arg | M1 | Tier 1 & Tier 2 | `tier1_feature_tests.rs`, `tier2_boundary_tests.rs` |
| **F02** | Window Functions & COLA | Hann, Hamming, Blackman, Bartlett, COLA checks | M1 | Tier 1 & Tier 2 | `tier1_feature_tests.rs`, `tier2_boundary_tests.rs` |
| **F03** | Matrix & Strided Tensor | Tensor striding, mode unfolding, n-mode product | M1 | Tier 1 & Tier 2 | `tier1_feature_tests.rs`, `tier2_boundary_tests.rs` |
| **F04** | Typed `DspError` System | Comprehensive domain error enum and formatting | M1 | Tier 1 & Tier 2 | `tier1_feature_tests.rs`, `tier2_boundary_tests.rs` |
| **F05** | `SineGenerator` | Phase accumulator pure tone with analytical RMS | M2 | Tier 1 & Tier 2 | `tier1_feature_tests.rs`, `tier2_boundary_tests.rs` |
| **F06** | `HarmonicComplexGenerator`| Multi-tone series, inharmonicity ($B$), recipes | M2 | Tier 1 & Tier 3 | `tier1_feature_tests.rs`, `tier3_interaction_tests.rs` |
| **F07** | `ChirpGenerator` | Linear and logarithmic continuous frequency sweeps | M2 | Tier 1 & Tier 3 | `tier1_feature_tests.rs`, `tier3_interaction_tests.rs` |
| **F08** | `ImpulseGenerator` | Kronecker delta, bandlimited sinc, transients | M2 | Tier 1 & Tier 2 | `tier1_feature_tests.rs`, `tier2_boundary_tests.rs` |
| **F09** | `NoiseGenerator` | Deterministic PCG32 PRNG (uniform, Gaussian, pink) | M2 | Tier 1 & Tier 2 | `tier1_feature_tests.rs`, `tier2_boundary_tests.rs` |
| **F10** | `SyntheticInstrumentSource`| 5D parameterized acoustic note synthesizer | M2 | Tier 1 & Tier 4 | `tier1_feature_tests.rs`, `tier4_application_tests.rs` |
| **F11** | Time-Domain Metrics | MSE, RMSE, SNR (dB), PSNR (dB), cross-correlation | M2 | Tier 1 & Tier 3 | `tier1_feature_tests.rs`, `tier3_interaction_tests.rs` |
| **F12** | Spectral-Domain Metrics | Spectral Convergence, LSD (dB), Peak Error | M2 | Tier 1 & Tier 3 | `tier1_feature_tests.rs`, `tier3_interaction_tests.rs` |
| **F13** | Numerical Stability Checks| Invariants: NaN, Inf, clipping, energy conservation | M2 | Tier 1 & Tier 2 | `tier1_feature_tests.rs`, `tier2_boundary_tests.rs` |
| **F14** | First-Principles DFT | Exact $O(N^2)$ direct transform for ground truth | M3 | Tier 2 & Tier 3 | `tier2_boundary_tests.rs`, `tier3_interaction_tests.rs` |
| **F15** | Radix-2 Cooley-Tukey FFT| $O(N \log N)$ FFT with bit-reversal & twiddle tables | M3 | Tier 2 & Tier 3 | `tier2_boundary_tests.rs`, `tier3_interaction_tests.rs` |
| **F16** | STFT Engine | Sliding-block STFT with configurable window and hop | M3 | Tier 2 & Tier 3 | `tier2_boundary_tests.rs`, `tier3_interaction_tests.rs` |
| **F17** | Spectral Peak Detection | Parabolic sub-bin interpolation on log magnitudes | M3 | Tier 2 & Tier 3 | `tier2_boundary_tests.rs`, `tier3_interaction_tests.rs` |
| **F18** | Harmonic Partial Tracking | Partials tracking based on $f_0$ and inharmonicity $\beta$ | M3 | Tier 2 & Tier 4 | `tier2_boundary_tests.rs`, `tier4_application_tests.rs` |
| **F19** | Stochastic Residual Extraction| Subtraction of deterministic partials + ERB energies | M3 | Tier 2 & Tier 4 | `tier2_boundary_tests.rs`, `tier4_application_tests.rs` |
| **F20** | Hypercube Coordinates | 5-axis coordinate mapping and continuous interpolation | M4 | Tier 2 & Tier 4 | `tier2_boundary_tests.rs`, `tier4_application_tests.rs` |
| **F21** | First-Principles PCA / SVD| Covariance PCA via power iteration / SVD | M4 | Tier 2 & Tier 3 | `tier2_boundary_tests.rs`, `tier3_interaction_tests.rs` |
| **F22** | First-Principles NMF | Lee-Seung multiplicative updates with zero defense | M4 | Tier 2 & Tier 3 | `tier2_boundary_tests.rs`, `tier3_interaction_tests.rs` |
| **F23** | Tucker Decomposition | HOSVD tensor factorization with point evaluation | M4 | Tier 2 & Tier 4 | `tier2_boundary_tests.rs`, `tier4_application_tests.rs` |
| **F24** | CP-ALS Tensor Decomposition| Canonical Polyadic via regularized ALS | M4 | Tier 2 & Tier 3 | `tier2_boundary_tests.rs`, `tier3_interaction_tests.rs` |
| **F25** | Additive Oscillator Bank | Sinusoidal synthesis with continuous phase tracking | M4 | Tier 2 & Tier 3 | `tier2_boundary_tests.rs`, `tier3_interaction_tests.rs` |
| **F26** | Phase Handling Models | Harmonic locking ($\phi_k = k\phi_0$), continuous, dispersion | M4 | Tier 2 & Tier 4 | `tier2_boundary_tests.rs`, `tier4_application_tests.rs` |
| **F27** | Stochastic Noise Resynthesis | Shaped noise synthesis matching residual envelopes | M4 | Tier 2 & Tier 4 | `tier2_boundary_tests.rs`, `tier4_application_tests.rs` |
| **F28** | Overlap-Add (OLA) Resynthesis| Windowed block synthesis with COLA normalization | M4 | Tier 2 & Tier 3 | `tier2_boundary_tests.rs`, `tier3_interaction_tests.rs` |
| **F29** | Composable `DspPipeline` | Modular trait-based pipeline chaining stages | M4 | Tier 2 & Tier 4 | `tier2_boundary_tests.rs`, `tier4_application_tests.rs` |
| **F30** | Quality Tier Configuration| Presets (`Reference`, `Production`, `Preview`) | M4 | Tier 2 & Tier 4 | `tier2_boundary_tests.rs`, `tier4_application_tests.rs` |

---

## 5. Quantitative Reconstruction Fidelity Metrics Suite

### 5.1 Time-Domain Metrics
Let $x[n]$ denote the original reference signal and $\hat{x}[n]$ denote the reconstructed signal for $n \in \{0, \dots, N-1\}$.

1. **Mean Squared Error (MSE)**:
   $$\text{MSE} = \frac{1}{N} \sum_{n=0}^{N-1} (x[n] - \hat{x}[n])^2$$

2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\text{MSE}}$$

3. **Signal-to-Noise Ratio (SNR in dB)**:
   $$\text{SNR} = 10 \log_{10} \left( \frac{\sum_{n=0}^{N-1} x[n]^2 + \epsilon}{\sum_{n=0}^{N-1} (x[n] - \hat{x}[n])^2 + \epsilon} \right)$$
   where $\epsilon = 10^{-12}$ prevents division by zero when comparing silent signals. For identical non-zero signals, SNR is clamped to $+150.0\text{ dB}$.

4. **Peak Signal-to-Noise Ratio (PSNR in dB)**:
   $$\text{PSNR} = 20 \log_{10} \left( \frac{\max_{n} |x[n]| + \epsilon}{\text{RMSE} + \epsilon} \right)$$

5. **Normalized Cross-Correlation ($r$)**:
   $$r = \frac{\sum_{n=0}^{N-1} x[n] \hat{x}[n]}{\sqrt{\sum_{n=0}^{N-1} x[n]^2} \sqrt{\sum_{n=0}^{N-1} \hat{x}[n]^2} + \epsilon} \in [-1.0, 1.0]$$

### 5.2 Spectral-Domain Metrics
Let $|X[k]|$ and $|\hat{X}[k]|$ denote the magnitude spectra across $K$ discrete Fourier bins.

1. **Spectral Convergence (SC)**:
   $$\text{SC} = \frac{\sqrt{\sum_{k=0}^{K-1} (|X[k]| - |\hat{X}[k]|)^2}}{\sqrt{\sum_{k=0}^{K-1} |X[k]|^2 + \epsilon}}$$
   Evaluates to $0.0$ for identical spectra; invariant to global scalar scaling: $\text{SC}(X, cX) = |1 - c|$.

2. **Log-Spectral Distance (LSD in dB)**:
   $$\text{LSD} = \sqrt{ \frac{1}{K} \sum_{k=0}^{K-1} \left( 20 \log_{10}(|X[k]| + \delta) - 20 \log_{10}(|\hat{X}[k]| + \delta) \right)^2 }$$
   where $\delta = 10^{-6}$ corresponds to a $-120\text{ dBFS}$ floor threshold, eliminating $\log(0)$ singularities.

3. **Harmonic Peak Tracking Error**:
   $$\Delta f_{\max} = \max_{k} |f_{\text{extracted}, k} - f_{\text{theoretical}, k}|$$
   $$\Delta A_{\text{rel}, \max} = \max_{k} \frac{|A_{\text{extracted}, k} - A_{\text{theoretical}, k}|}{A_{\text{theoretical}, k}}$$

### 5.3 Target Threshold Matrix Across Quality Tiers

| Operational Tier | Max MSE | Min SNR (dB) | Min PSNR (dB) | Max Spectral Convergence | Max LSD (dB) | Max Peak Freq Error |
|---|---|---|---|---|---|---|
| **Tier 1 (Analytical Ground Truth)** | $< 10^{-12}$ | $> 120.0\text{ dB}$ | $> 120.0\text{ dB}$ | $< 10^{-12}$ | $< 0.001\text{ dB}$ | $< 10^{-6}\text{ Hz}$ |
| **Reference Tier (High Fidelity)** | $< 10^{-4}$ | $> 35.0\text{ dB}$ | $> 45.0\text{ dB}$ | $< 0.03$ | $< 0.8\text{ dB}$ | $< 0.5\text{ Hz}$ |
| **Production Tier (Balanced)** | $< 10^{-3}$ | $> 28.0\text{ dB}$ | $> 36.0\text{ dB}$ | $< 0.08$ | $< 1.5\text{ dB}$ | $< 1.5\text{ Hz}$ |
| **Preview Tier (Low-Complexity)** | $< 10^{-2}$ | $> 18.0\text{ dB}$ | $> 24.0\text{ dB}$ | $< 0.20$ | $< 2.8\text{ dB}$ | $< 5.0\text{ Hz}$ |

---

## 6. Test Runner Commands & Milestone Verification Gates

### 6.1 Current Foundation Verification (Milestone M1)
The foundation crate modules (`calyx_dsp::error` and `calyx_dsp::math`) are fully implemented and verified via the library unit test suite:
```bash
cargo test --lib
```
*Current Empirical Status*: 43 tests pass cleanly in 0.00s with zero warnings.

### 6.2 Milestone Progression Verification Schedule
Integration test files in `tests/` exercise cross-module contracts. They compile and execute as the corresponding milestone modules are delivered:

1. **M1 Contract-Aligned Tests**:
   Tests covering F01–F04 in `tests/tier1_feature_tests.rs` and `tests/tier2_boundary_tests.rs` execute once method signatures (`from_vec` instead of `new`, `DspResult` unpacking) are aligned.
2. **Milestone M2 Verification Gate**:
   Delivery of `calyx_dsp::synthetic` and `calyx_dsp::metrics` enables:
   ```bash
   cargo test --test tier1_feature_tests
   ```
   *Pass Criteria*: All 65 feature tests pass; analytical ground-truth SNR $> 120\text{ dB}$.
3. **Milestone M3 Verification Gate**:
   Delivery of `calyx_dsp::spectral` enables spectral boundary tests in Tier 2 and spectral interactions in Tier 3.
4. **Milestone M4 Verification Gate**:
   Delivery of `calyx_dsp::hypercube`, `calyx_dsp::synthesis`, and `calyx_dsp::pipeline` enables the complete 16-interaction matrix (Tier 3) and 4 real-world application scenarios (Tier 4).

### 6.3 Milestone M5 Full Suite Acceptance Run
The formal acceptance verification for Milestone M5 executes the entire E2E test suite:
```bash
cargo test --test tier1_feature_tests --test tier2_boundary_tests --test tier3_interaction_tests --test tier4_application_tests
```

*Milestone M5 Invariants & Verification Guarantees*:
- 100% pass across all 235 tests (zero failures, zero panics).
- Zero warnings under `cargo test` and `cargo check`.
- Strictly safe Rust (`#![deny(unsafe_code)]`).
- Deterministic in-memory execution: all PRNG generators seeded with known constants.
- Continuous Integration execution time $< 15$ seconds total for all 235 tests.

### 6.4 Empirical Inventory Verification Command
To independently verify the exact test function inventory without running the compiler:
```bash
grep -c "#\[test\]" tests/tier1_feature_tests.rs tests/tier2_boundary_tests.rs tests/tier3_interaction_tests.rs tests/tier4_application_tests.rs
```
*Expected Verifiable Output*:
- `tests/tier1_feature_tests.rs`: 65
- `tests/tier2_boundary_tests.rs`: 150
- `tests/tier3_interaction_tests.rs`: 16
- `tests/tier4_application_tests.rs`: 4
- **Total Test Functions**: 235
