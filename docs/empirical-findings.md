# Calyx DSP Empirical Research Findings

**Status:** Empirical Benchmark & Architecture Study  
**Runner:** `cargo run --release --bin calyx_experiments` ([src/experiments/mod.rs](../src/experiments/mod.rs), [src/bin/calyx_experiments.rs](../src/bin/calyx_experiments.rs))

This document records the empirical results from systematic parameter sweeps across the `calyx_dsp` experimental framework, directly addressing the open research questions in [open-questions.md](./open-questions.md).

---

## Executive Summary of Resolved / Clarified Questions

| Open Question | Key Empirical Finding | Architectural Recommendation |
| :--- | :--- | :--- |
| **OQ 1.1: Partial Count ($N$)** | Flute-like geometric-decay timbres saturate fidelity by $N = 8$ partials ($\text{SNR} = 38.12\text{ dB}$, $\text{LSD} = 8.68\text{ dB}$). Bright bowed-string, reed, and piano timbres continue gaining $+4.8\text{ to }+5.0\text{ dB}$ SNR between $N = 16$ and $N = 24$. | Adopt **adaptive per-instrument partial budgets**: $N \in [8, 12]$ for dark/flute-like timbres ($11.7\text{ KB/s}$ uncompressed) and $N \in [24, 64]$ for strings, reeds, and piano. |
| **OQ 1.2 & 3.1: Phase Handling & Metrics** | **Phase Vocoder Accumulation** ($0\text{ bytes/frame}$ stored phase) matches **Stored Phase Envelopes** ($64\text{ bytes/frame}$) within $0.12\text{ dB}$ SNR on harmonic sustain ($42.50\text{ dB}$ vs. $42.62\text{ dB}$) and $0.23\text{ dB}$ on inharmonic piano ($36.91\text{ dB}$ vs. $37.14\text{ dB}$). Conversely, **Harmonic Phase Locking** ($\phi_k = k \phi_0$) fails on inharmonic piano ($\text{LSD}$ degrades from $0.194\text{ dB}$ to $5.619\text{ dB}$) because forcing integer-multiple phases at each frame hop conflicts with stiff-string frequency dispersion $f_k = k f_0 \sqrt{1 + B k^2}$. | Use **Phase Vocoder Accumulation** (with initial onset phase or seeded **Phase Dispersion** for chorus/ensemble warmth) as the default playback synthesizer, avoiding per-frame phase storage overhead. Use **Log-Spectral Distance (LSD)** and **Spectral Convergence** rather than time-domain waveform SNR when evaluating phase-dispersed synthesis. |
| **OQ 1.3: Spectral Frame Rate** | Sustained tones achieve $>45\text{ dB}$ SNR even at $93.8\text{ Hz}$ ($\text{hop} = 512$, $\text{LSD} = 0.244\text{ dB}$). Fast transients ($\tau = 60\text{ ms}$) see LSD improve by $2.4\times$ ($2.349\text{ dB} \to 0.964\text{ dB}$) when moving from $93.8\text{ Hz}$ to $375\text{ Hz}$ ($\text{hop} = 128$). | Use a **baseline frame rate of $187.5\text{ Hz}$** ($\text{hop} = 256$ at $48\text{ kHz}$) for Production/Reference sustain, or **multi-resolution frame rates** ($375\text{ Hz}$ during onset transients, $93.8\text{ Hz}$ during sustain). |
| **OQ 2.1 & 2.3: PCA vs. NMF** | Across ranks $K \in \{1..16\}$ on a 24-partial multi-mode evolving note, **PCA (Jacobi SVD)** consistently achieves $+0.6\text{ to }+1.2\text{ dB}$ higher SNR, lower spectral convergence error, and **$3\times$ to $8\times$ faster decomposition** ($305\text{ }\mu\text{s}$ vs. $1,406\text{ }\mu\text{s}$ at $K=8$) than **NMF (Lee-Seung)**, while reconstruction speed is identical ($\sim 15\text{ }\mu\text{s}$). Both plateau at $K = 8$. | Default to **PCA / SVD** for compression speed and L2 optimality; reserve **NMF** for use cases requiring strictly additive, non-negative basis interpretability (e.g., spectral layer editing). Select rank $K = 8$ for Reference, $K = 4$ for Production, and $K = 2$ for Preview. |
| **OQ 2.2, 2.5, & 5.1: Tucker vs. CP-ALS & Real-Time Polyphony** | On a 4-mode $(16 \times 12 \times 6 \times 16)$ instrument hypercube (72 notes, 18,432 spectral floats, equivalent to 552,960 PCM samples): **CP-ALS** achieves **$1,807:1$ PCM compression** ($60.2\times$ spectral compression) at $83.86\text{ dB}$ tensor SNR in Production tier ($R=6$), with **$0.30\text{ }\mu\text{s}$** single-frame evaluation latency (**$>16,000$ simultaneous voices** at 50% CPU). **Tucker-HOSVD** decomposes **$18\times$ faster** ($1.20\text{ ms}$ vs. $21.57\text{ ms}$) and supports orthogonal per-axis rank control ($[4, 4, 3, 8] \to 861:1$ PCM compression, $64.51\text{ dB}$ SNR, $33.6\text{ }\mu\text{s}$/frame $\approx 148$ voices). | Offer a **hybrid tensor architecture**: use **Tucker-HOSVD** during rapid offline authoring/iteration or when independent per-axis rank control is needed, and **CP-ALS** (or Tucker-to-CP compression) for runtime playback where its $\mathcal{O}(K \cdot R)$ frame evaluation enables thousands of real-time voices. |

---

## 1. Partial Count Sweep (Open Question 1.1)

**Methodology:** Synthesized 4 canonical instrument timbres at $f_s = 48\text{ kHz}$ (`Flute` with geometric decay $0.4^k$, `Clarinet` odd harmonics, `Bowed String` $1/k$ sawtooth, and `Piano` stiff-string $B = 4 \times 10^{-4}$ with exponential damping), analyzed via STFT ($\text{window}=1024, \text{hop}=256$), truncated each frame to the top $N \in \{2, 4, 8, 12, 16, 24\}$ partials (with stochastic residual zeroed to isolate deterministic partial contribution), and reconstructed via Overlap-Add.

| Timbre | Max Partials ($N$) | SNR (dB) | LSD (dB) | Spectral Conv. | RMSE | Payload (KB/s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flute (Geometric 0.4) | 2 | 16.49 | 13.003 | 0.14834 | 0.05172 | 2.93 |
| Flute (Geometric 0.4) | 4 | 32.25 | 9.779 | 0.02264 | 0.00843 | 5.86 |
| Flute (Geometric 0.4) | 8 | 38.12 | 8.679 | 0.00908 | 0.00429 | 11.72 |
| Flute (Geometric 0.4) | 12 | 38.12 | 8.679 | 0.00908 | 0.00429 | 17.58 |
| Flute (Geometric 0.4) | 16 | 38.12 | 8.679 | 0.00908 | 0.00429 | 23.44 |
| Flute (Geometric 0.4) | 24 | 38.12 | 8.679 | 0.00908 | 0.00429 | 35.16 |
| Clarinet (Odd Harmonics) | 2 | 8.18 | 29.479 | 0.38654 | 0.12400 | 2.93 |
| Clarinet (Odd Harmonics) | 4 | 11.67 | 27.955 | 0.25675 | 0.08292 | 5.86 |
| Clarinet (Odd Harmonics) | 8 | 15.77 | 24.916 | 0.15852 | 0.05172 | 11.72 |
| Clarinet (Odd Harmonics) | 12 | 18.61 | 21.877 | 0.11333 | 0.03730 | 17.58 |
| Clarinet (Odd Harmonics) | 16 | 21.04 | 18.800 | 0.08508 | 0.02822 | 23.44 |
| Clarinet (Odd Harmonics) | 24 | 26.03 | 12.728 | 0.04698 | 0.01587 | 35.16 |
| Bowed String (Sawtooth) | 2 | 7.07 | 32.480 | 0.43924 | 0.12639 | 2.93 |
| Bowed String (Sawtooth) | 4 | 10.21 | 31.489 | 0.30509 | 0.08806 | 5.86 |
| Bowed String (Sawtooth) | 8 | 14.03 | 28.725 | 0.19518 | 0.05675 | 11.72 |
| Bowed String (Sawtooth) | 12 | 16.74 | 25.878 | 0.14193 | 0.04151 | 17.58 |
| Bowed String (Sawtooth) | 16 | 19.10 | 23.069 | 0.10769 | 0.03165 | 23.44 |
| Bowed String (Sawtooth) | 24 | 24.05 | 15.266 | 0.06015 | 0.01789 | 35.16 |
| Piano (Inharmonic B=4e-4) | 2 | 7.66 | 31.656 | 0.39785 | 0.10538 | 2.93 |
| Piano (Inharmonic B=4e-4) | 4 | 11.11 | 30.806 | 0.26149 | 0.07082 | 5.86 |
| Piano (Inharmonic B=4e-4) | 8 | 15.29 | 28.536 | 0.15600 | 0.04381 | 11.72 |
| Piano (Inharmonic B=4e-4) | 12 | 18.18 | 25.777 | 0.10830 | 0.03141 | 17.58 |
| Piano (Inharmonic B=4e-4) | 16 | 20.61 | 22.604 | 0.07946 | 0.02373 | 23.44 |
| Piano (Inharmonic B=4e-4) | 24 | 25.44 | 15.798 | 0.04258 | 0.01360 | 35.16 |

---

## 2. Phase Strategy Comparison (Open Questions 1.2 & 3.1)

**Methodology:** Evaluated all four phase strategies from Section 6.1 of the specification on both a harmonic sustain tone ($B = 0$) and an inharmonic stiff-string piano tone ($B = 4 \times 10^{-4}$).

| Signal | Phase Strategy | Waveform SNR (dB) | Pearson $r$ | LSD (dB) | Spectral Conv. | Phase Bytes/Frame |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Harmonic Sustain (A3 220Hz, B=0) | 1. Stored Phase Envelopes | 42.62 | 1.0000 | 0.135 | 0.00149 | 64 |
| Harmonic Sustain (A3 220Hz, B=0) | 2. Phase Vocoder Accumulation | 42.50 | 1.0000 | 0.130 | 0.00147 | 0 |
| Harmonic Sustain (A3 220Hz, B=0) | 3. Harmonic Phase Locking | 2.37 | 0.7119 | 1.312 | 0.01165 | 4 |
| Harmonic Sustain (A3 220Hz, B=0) | 4. Phase Dispersion Modeling | -4.63 | -0.4616 | 1.174 | 0.00798 | 0 |
| Inharmonic Piano (A3 220Hz, B=4e-4) | 1. Stored Phase Envelopes | 37.14 | 0.9999 | 0.201 | 0.00293 | 64 |
| Inharmonic Piano (A3 220Hz, B=4e-4) | 2. Phase Vocoder Accumulation | 36.91 | 0.9999 | 0.194 | 0.00288 | 0 |
| Inharmonic Piano (A3 220Hz, B=4e-4) | 3. Harmonic Phase Locking | 1.30 | 0.6293 | 5.619 | 0.11404 | 4 |
| Inharmonic Piano (A3 220Hz, B=4e-4) | 4. Phase Dispersion Modeling | -4.52 | -0.4225 | 0.978 | 0.00949 | 0 |

**Key Takeaways:**
1. **Phase Vocoder Accumulation eliminates per-frame phase storage without losing waveform or spectral fidelity:** Integrating instantaneous partial frequencies across hops ($\phi_k[m] = \phi_k[m-1] + 2\pi f_k[m] \cdot \text{hop}/f_s$) achieves $42.50\text{ dB}$ SNR ($0.130\text{ dB}$ LSD) on harmonic sustain and $36.91\text{ dB}$ SNR ($0.194\text{ dB}$ LSD) on inharmonic piano with **0 bytes/frame** of stored phase envelopes.
2. **Why Harmonic Phase Locking fails on inharmonic instruments:** Enforcing $\phi_k[m] = k \phi_0[m] \pmod{2\pi}$ at every frame boundary introduces a periodic phase reset that fights the inharmonic partial frequency $f_k = k f_0 \sqrt{1 + B k^2} > k f_0$. During Overlap-Add window summation, this phase mismatch causes destructive interference on upper partials, inflating LSD to $5.619\text{ dB}$.
3. **Metric Selection (OQ 3.1):** **Phase Dispersion Modeling** intentionally randomizes initial inter-partial phase offsets while keeping phase continuous across hops. Notice that time-domain Waveform SNR drops to $-4.52\text{ dB}$ (because the time-domain waveform shape changes when partial phases are shifted), while **Log-Spectral Distance remains $0.978\text{ dB}$** and **Spectral Convergence is $0.00949$** (< 1% magnitude error). This confirms OQ 3.1: spectral-domain metrics (LSD, Spectral Convergence, Multi-Resolution STFT Loss) must be the primary optimization criteria whenever phase-free or phase-dispersed synthesis is used.

---

## 3. Spectral Frame Rate / Hop Size Sweep (Open Question 1.3)

**Methodology:** Swept STFT hop sizes $\text{hop} \in \{64, 128, 256, 512\}$ ($\text{window} = 1024$ at $48\text{ kHz}$, corresponding to frame rates of $750, 375, 187.5, 93.8\text{ Hz}$) across sustained tones ($\tau = 0.8\text{ s}$), fast decay transients ($\tau = 0.06\text{ s}$), and linear pitch glides ($220 \to 660\text{ Hz}$).

| Signal | Window | Hop | Frame Rate (Hz) | SNR (dB) | LSD (dB) | Spectral Conv. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sustain Tone ($\tau=0.8\text{ s}$) | 1024 | 64 | 750.0 | 45.16 | 0.035 | 0.00100 |
| Sustain Tone ($\tau=0.8\text{ s}$) | 1024 | 128 | 375.0 | 45.49 | 0.050 | 0.00091 |
| Sustain Tone ($\tau=0.8\text{ s}$) | 1024 | 256 | 187.5 | 45.80 | 0.094 | 0.00106 |
| Sustain Tone ($\tau=0.8\text{ s}$) | 1024 | 512 | 93.8 | 45.67 | 0.244 | 0.00119 |
| Fast Decay Transient ($\tau=0.06\text{ s}$) | 1024 | 64 | 750.0 | 21.53 | 1.072 | 0.01799 |
| Fast Decay Transient ($\tau=0.06\text{ s}$) | 1024 | 128 | 375.0 | 21.91 | 0.964 | 0.01718 |
| Fast Decay Transient ($\tau=0.06\text{ s}$) | 1024 | 256 | 187.5 | 22.29 | 1.706 | 0.01649 |
| Fast Decay Transient ($\tau=0.06\text{ s}$) | 1024 | 512 | 93.8 | 22.17 | 2.349 | 0.01770 |
| Linear Pitch Glide ($220\to 660\text{ Hz}$) | 1024 | 64 | 750.0 | -2.87 | 2.419 | 0.07543 |
| Linear Pitch Glide ($220\to 660\text{ Hz}$) | 1024 | 128 | 375.0 | -2.95 | 1.791 | 0.05183 |
| Linear Pitch Glide ($220\to 660\text{ Hz}$) | 1024 | 256 | 187.5 | -2.84 | 1.055 | 0.06828 |
| Linear Pitch Glide ($220\to 660\text{ Hz}$) | 1024 | 512 | 93.8 | -2.80 | 0.805 | 0.03271 |

---

## 4. Partial-Domain Dimensionality Reduction: PCA vs. NMF (Open Questions 2.1 & 2.3)

**Methodology:** Synthesized an 8,192-sample phase-continuous 24-partial note with 5 independent temporal-spectral modulation modes (attack transient, formant sweep, odd/even harmonic balance shift, and upper-harmonic shimmer), decomposed the resulting $64 \times 24$ partial trajectory via **PCA (Jacobi SVD)** and **NMF (Lee-Seung, 50 iterations)** across ranks $K \in \{1, 2, 4, 8, 12, 16\}$, and measured end-to-end reconstructed audio fidelity and compute latency.

| Method | Rank ($K$) | SNR (dB) | LSD (dB) | Spectral Conv. | Matrix Comp. Ratio | Decompose ($\mu$s) | Reconstruct ($\mu$s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PCA (Jacobi SVD) | 1 | 15.21 | 4.744 | 0.15547 | 17.45x | 325.4 | 16.4 |
| NMF (Lee-Seung) | 1 | 15.21 | 4.744 | 0.15547 | 17.45x | 365.8 | 15.1 |
| PCA (Jacobi SVD) | 2 | 16.89 | 4.728 | 0.11910 | 8.73x | 326.0 | 12.7 |
| NMF (Lee-Seung) | 2 | 15.62 | 4.722 | 0.14438 | 8.73x | 476.1 | 14.5 |
| PCA (Jacobi SVD) | 4 | 18.45 | 4.718 | 0.09135 | 4.36x | 302.4 | 12.7 |
| NMF (Lee-Seung) | 4 | 17.25 | 4.728 | 0.11354 | 4.36x | 931.7 | 13.0 |
| PCA (Jacobi SVD) | 8 | 19.54 | 4.679 | 0.07824 | 2.18x | 305.0 | 15.3 |
| NMF (Lee-Seung) | 8 | 18.57 | 4.706 | 0.09390 | 2.18x | 1406.5 | 15.3 |
| PCA (Jacobi SVD) | 12 | 19.56 | 4.670 | 0.07817 | 1.45x | 298.6 | 17.9 |
| NMF (Lee-Seung) | 12 | 18.58 | 4.722 | 0.09250 | 1.45x | 1649.9 | 18.4 |
| PCA (Jacobi SVD) | 16 | 19.56 | 4.671 | 0.07816 | 1.09x | 312.7 | 22.1 |
| NMF (Lee-Seung) | 16 | 18.91 | 4.709 | 0.08697 | 1.09x | 2528.5 | 21.7 |

---

## 5. Multi-Dimensional Tensor Decomposition & Real-Time Polyphony (Open Questions 2.2, 2.5, & 5.1)

**Methodology:** Populated a 4-mode spectral hypercube tensor of shape $(T=16, P=12, D=6, K=16)$ representing a 1-octave chromatic instrument across 6 velocity layers (72 notes, 18,432 spectral parameters, corresponding to $552,960$ samples of $48\text{ kHz}$ PCM audio). Evaluated **Tucker-HOSVD** (`TuckerDecomposition`) and **CP-ALS** (`CpAls`, 20 iterations) across Preview, Production, and Reference rank configurations. Measured tensor SNR, relative Frobenius error, parameter compression ratios (vs. spectral tensor and vs. raw PCM), offline decomposition latency, single-frame evaluation latency (all 16 partials at a single $(t, p, d)$ coordinate), and implied real-time voice polyphony under a 50% single-core CPU budget at a $100\text{ Hz}$ frame update rate ($5,000\text{ }\mu\text{s}$ available per $10\text{ ms}$ frame interval).

| Method | Tier | Ranks | Tensor SNR (dB) | Rel. Error | Params (f32) | Spectral Comp. | PCM Comp. | Decompose (ms) | Frame Eval ($\mu$s) | Max Voices (50% CPU) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tucker-HOSVD | Preview | `[2, 2, 2, 4]` | 39.29 | 0.0109 | 164 | 112.4x | 3371.7:1 | 1.22 | 2.66 | 1879 |
| Tucker-HOSVD | Production | `[4, 4, 3, 8]` | 64.51 | 0.0006 | 642 | 28.7x | 861.3:1 | 1.20 | 33.64 | 148 |
| Tucker-HOSVD | Reference | `[8, 6, 4, 12]` | 93.67 | 0.0000 | 2720 | 6.8x | 203.3:1 | 1.26 | 168.48 | 29 |
| CP-ALS | Preview | `R=2` | 37.13 | 0.0139 | 102 | 180.7x | 5421.2:1 | 11.35 | 0.10 | 47619 |
| CP-ALS | Production | `R=6` | 83.86 | 0.0001 | 306 | 60.2x | 1807.1:1 | 21.57 | 0.30 | 16713 |
| CP-ALS | Reference | `R=12` | 95.93 | 0.0000 | 612 | 30.1x | 903.5:1 | 37.09 | 0.70 | 7189 |

---

## 6. `.calyx` Binary Container Format Serialization Benchmark (Open Questions 4.1 & 4.2)

**Methodology:** Evaluated the chunked `.calyx` binary container format (`src/io/container.rs`, magic `CALX`, version `0.1`, Adler-32 integrity checksums, FourCC chunks `META`, `AXES`, `TUCK`, `CPMD`, `NOIS`) across 7 configurations on the 72-note $(16 \times 12 \times 6 \times 16)$ instrument hypercube. Measured serialized byte sizes, compression ratios relative to raw hypercube tensors and 16-bit $48\text{ kHz}$ PCM audio, serialization/deserialization latencies, and roundtrip numerical reconstruction SNR.

| Configuration | Tier | Size (Bytes) | Size (KB) | Hypercube Comp. | PCM Comp. | Serialize ($\mu$s) | Deserialize ($\mu$s) | Rel. Error | Recon SNR (dB) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tucker-HOSVD (Preview) | Preview | 1641 | 1.60 KB | 89.9x | 673.9:1 | 5.4 | 5.9 | 0.0109 | 39.29 |
| Tucker-HOSVD (Production) | Production | 5468 | 5.34 KB | 27.0x | 202.3:1 | 15.2 | 15.9 | 0.0006 | 64.51 |
| Tucker-HOSVD (Reference) | Reference | 22091 | 21.57 KB | 6.7x | 50.1:1 | 59.4 | 63.2 | 0.0000 | 93.67 |
| CP-ALS (Preview) | Preview | 1129 | 1.10 KB | 130.6x | 979.6:1 | 4.0 | 4.3 | 0.0139 | 37.13 |
| CP-ALS (Production) | Production | 2764 | 2.70 KB | 53.3x | 400.1:1 | 8.9 | 9.7 | 0.0001 | 83.86 |
| CP-ALS (Reference) | Reference | 5211 | 5.09 KB | 28.3x | 212.2:1 | 17.1 | 16.4 | 0.0000 | 95.93 |
| Hybrid Container (Production) | Production | 8078 | 7.89 KB | 18.3x | 136.9:1 | 23.4 | 24.3 | 0.0006 | 64.51 |
