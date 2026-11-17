# Calyx Open Questions

**Status:** Living document
**License:** This document is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) License](https://creativecommons.org/licenses/by/4.0/).

This document tracks unresolved research and design questions that need investigation before or during implementation. Each question includes context on why it matters, what approaches might work, and what kind of expertise would help answer it.

Questions are grouped by topic area. Within each group, questions are roughly ordered by how early they need to be resolved (questions that block PoC work come first).

If you have expertise relevant to any of these questions, contributions are welcome via GitHub Issues or Discussions.

---

- [Calyx Open Questions](#calyx-open-questions)
  - [1. Spectral analysis and representation](#1-spectral-analysis-and-representation)
    - [1.1 How many partials are needed per frame?](#11-how-many-partials-are-needed-per-frame)
    - [1.2 How should phase information be handled?](#12-how-should-phase-information-be-handled)
    - [1.3 What spectral frame rate is sufficient?](#13-what-spectral-frame-rate-is-sufficient)
    - [1.4 How should inharmonicity be represented?](#14-how-should-inharmonicity-be-represented)
    - [1.5 How should the stochastic component be modeled?](#15-how-should-the-stochastic-component-be-modeled)
  - [2. Compression](#2-compression)
    - [2.1 PCA vs. NMF vs. other methods for partial amplitude reduction](#21-pca-vs-nmf-vs-other-methods-for-partial-amplitude-reduction)
    - [2.2 Which tensor decomposition method is best suited?](#22-which-tensor-decomposition-method-is-best-suited)
    - [2.3 How should decomposition rank be selected?](#23-how-should-decomposition-rank-be-selected)
    - [2.4 How effective is shared component extraction?](#24-how-effective-is-shared-component-extraction)
    - [2.5 What compression ratios are achievable in practice?](#25-what-compression-ratios-are-achievable-in-practice)
  - [3. Perceptual quality](#3-perceptual-quality)
    - [3.1 What objective metrics best predict subjective quality for spectral compression?](#31-what-objective-metrics-best-predict-subjective-quality-for-spectral-compression)
    - [3.2 How does quality degrade in a musical mix vs. in isolation?](#32-how-does-quality-degrade-in-a-musical-mix-vs-in-isolation)
    - [3.3 Which instrument types are most and least tolerant of spectral compression?](#33-which-instrument-types-are-most-and-least-tolerant-of-spectral-compression)
  - [4. Container format and compatibility](#4-container-format-and-compatibility)
    - [4.1 What metadata must the container format preserve?](#41-what-metadata-must-the-container-format-preserve)
    - [4.2 How should the container format support versioning and extensibility?](#42-how-should-the-container-format-support-versioning-and-extensibility)
    - [4.3 Should the format support partial loading?](#43-should-the-format-support-partial-loading)
  - [5. Real-time synthesis](#5-real-time-synthesis)
    - [5.1 What is the computational cost of hypercube reconstruction?](#51-what-is-the-computational-cost-of-hypercube-reconstruction)
    - [5.2 How should the additive oscillator bank be implemented for efficiency?](#52-how-should-the-additive-oscillator-bank-be-implemented-for-efficiency)
    - [5.3 How should transitions between notes be handled?](#53-how-should-transitions-between-notes-be-handled)
  - [6. Intellectual property](#6-intellectual-property)
    - [6.1 Are there existing patents that constrain the design?](#61-are-there-existing-patents-that-constrain-the-design)
    - [6.2 Should Calyx pursue a defensive patent strategy?](#62-should-calyx-pursue-a-defensive-patent-strategy)
  - [7. Ecosystem and adoption](#7-ecosystem-and-adoption)
    - [7.1 What is the minimum viable integration with existing tools?](#71-what-is-the-minimum-viable-integration-with-existing-tools)
    - [7.2 What governance model should the project adopt as it grows?](#72-what-governance-model-should-the-project-adopt-as-it-grows)


## 1. Spectral analysis and representation

### 1.1 How many partials are needed per frame?

**Status:** Resolved (Empirical Benchmark — see [docs/empirical-findings.md §1](./empirical-findings.md#1-partial-count-sweep-open-question-11))

**Resolution:** Adopt **adaptive per-instrument partial budgets**.
- Dark or geometric-decay timbres (such as flute with $0.4^k$ decay) reach a complete fidelity plateau at $N = 8$ partials ($\text{SNR} = 38.12\text{ dB}$, $\text{LSD} = 8.68\text{ dB}$, $11.72\text{ KB/s}$ uncompressed payload); allocating up to 24 partials triples bandwidth to $35.16\text{ KB/s}$ with $0.00\text{ dB}$ fidelity gain.
- Bright timbres (bowed-string sawtooth $1/k$, odd-harmonic clarinet, inharmonic stiff-string piano $B = 4 \times 10^{-4}$) continue gaining $+4.8\text{ to }+5.0\text{ dB}$ SNR between $N = 16$ and $N = 24$.
- Recommended baseline budgets: $N \in [8, 12]$ for dark/flute timbres, and $N \in [24, 64]$ for strings, reeds, and piano depending on quality tier.

The number of tracked partials (N) directly affects both the fidelity of the representation and the size of the uncompressed hypercube. Too few partials and the reconstruction sounds thin or synthetic. Too many and the data volume grows without perceptual benefit.

### 1.2 How should phase information be handled?

**Status:** Resolved (Empirical Benchmark — see [docs/empirical-findings.md §2](./empirical-findings.md#2-phase-strategy-comparison-open-questions-12--31))

**Resolution:** Codify **Phase Vocoder Accumulation** as the normative default playback strategy with 0 bytes/frame stored phase overhead.
- Phase Vocoder Accumulation integrates partial instantaneous frequencies across frame hops ($\phi_k[m] = \phi_k[m-1] + 2\pi f_k[m] \cdot \text{hop}/f_s$). It matches uncompressed stored phase envelopes within $0.12\text{ dB}$ SNR on harmonic sustain ($42.50\text{ dB}$ vs. $42.62\text{ dB}$) and within $0.23\text{ dB}$ on inharmonic piano ($36.91\text{ dB}$ vs. $37.14\text{ dB}$), eliminating 64 bytes/frame of phase data (at 16 partials).
- Optional seeded **Phase Dispersion Modeling** (`Pcg32`) provides natural ensemble warmth ($\text{LSD} < 1.0\text{ dB}$, $\text{Spectral Convergence} < 0.01$).
- **Harmonic Phase Locking** ($\phi_k = k \phi_0$) is deprecated and prohibited for inharmonic timbres due to boundary phase cancellation clashing with physical dispersion ($5.62\text{ dB}$ LSD).
- Evaluation must use **Log-Spectral Distance (LSD)** and **Spectral Convergence** rather than waveform SNR.

### 1.3 What spectral frame rate is sufficient?

**Status:** Resolved (Empirical Benchmark — see [docs/empirical-findings.md §3](./empirical-findings.md#3-spectral-frame-rate--hop-size-sweep-open-question-13))

**Resolution:** Adopt a baseline spectral frame rate of **$187.5\text{ Hz}$** ($\text{hop} = 256$ at $48\text{ kHz}$ sampling rate, $\text{window} = 1024$) for Production and Reference tiers, or multi-resolution framing.
- Sustained tones achieve $>45.8\text{ dB}$ SNR and $0.094\text{ dB}$ LSD at $187.5\text{ Hz}$ (and $>45.6\text{ dB}$ even at $93.8\text{ Hz}$).
- Fast decay transients ($\tau = 60\text{ ms}$) see LSD improve $2.4\times$ from $2.349\text{ dB}$ at $93.8\text{ Hz}$ down to $0.964\text{ dB}$ at $375\text{ Hz}$ ($\text{hop} = 128$).
- Recommended strategy: $187.5\text{ Hz}$ uniform baseline for general synthesis, with optional multi-resolution framing ($375\text{ Hz}$ during attack transients, $93.8\text{ to } 187.5\text{ Hz}$ during sustain).

### 1.4 How should inharmonicity be represented?

Instruments like piano, marimba, and bells produce partials that deviate from exact integer multiples of the fundamental. These deviations are perceptually important (they contribute to the characteristic "warmth" of piano tone, for instance).

Options include: storing per-partial frequency ratios in every spectral frame (expensive), storing a per-pitch inharmonicity profile that applies across all frames at that pitch (compact but less accurate), or modeling inharmonicity as a function of pitch and partial number (very compact but requires a good model).

**Needed:** Analysis of how inharmonicity varies across pitch, dynamic, and time for relevant instruments. Determine whether a simple parametric model is sufficient or whether per-frame data is necessary.

**Relevant expertise:** Acoustics, piano and percussion modeling, physical modeling synthesis.

### 1.5 How should the stochastic component be modeled?

The specification describes the stochastic (noise/residual) component as a spectral envelope in perceptual frequency bands plus an energy level. This is a simplified model. Real instrument noise (bow scraping, breath turbulence, key mechanisms, string buzz) has temporal structure, spatial characteristics, and statistical properties that a static spectral envelope may not capture well.

Richer stochastic models could include: temporal modulation envelopes for the noise component, sub-band correlation structures, or short segments of recorded noise that are triggered and modulated rather than synthesized from parameters.

**Needed:** Evaluation of how much stochastic detail is perceptually necessary for different instrument types. Compare the simple spectral envelope model against richer alternatives in listening tests.

**Relevant expertise:** Noise modeling, stochastic signal processing, SMS/spectral modeling.

---

## 2. Compression

### 2.1 PCA vs. NMF vs. other methods for partial amplitude reduction

**Status:** Resolved (Empirical Benchmark — see [docs/empirical-findings.md §4](./empirical-findings.md#4-partial-domain-dimensionality-reduction-pca-vs-nmf-open-questions-21--23))

**Resolution:** Default to **PCA (Jacobi SVD)** for compression speed and L2 numerical optimality.
- Across ranks $K \in \{1..16\}$ on multi-mode partial trajectories, PCA consistently delivers $+0.6\text{ to }+1.2\text{ dB}$ higher SNR, lower spectral convergence error, and **$3\times$ to $8\times$ faster decomposition** ($305\text{ }\mu\text{s}$ vs. $1,406\text{ }\mu\text{s}$ at $K=8$) compared to NMF (Lee-Seung multiplicative updates).
- Both PCA and NMF error curves plateau around $K=8$.
- Reconstruction compute latency is identical ($\sim 15\text{ }\mu\text{s}$).
- NMF is retained as an optional mode specifically when non-negative, additively interpretable basis components are required for interactive timbral sound design or spectral layer editing.

### 2.2 Which tensor decomposition method is best suited?

**Status:** Resolved (Empirical Architecture Study — see [docs/empirical-findings.md §5](./empirical-findings.md#5-multi-dimensional-tensor-decomposition--real-time-polyphony-open-questions-22-25--51))

**Resolution:** Adopt a **Hybrid Tucker + CP Workflow**.
- **Tucker-HOSVD** excels in offline authoring, running $18\times$ faster ($1.20\text{ ms}$ vs. $21.57\text{ ms}$) and enabling independent, non-uniform per-axis rank control ($[T, P, D, K]$) for modal exploration.
- **CP-ALS** is the optimal runtime format, evaluating a single frame in **$0.30\text{ }\mu\text{s}$** at rank $R=6$ (enabling **$>16,000$ simultaneous voices** at 50% CPU budget) and achieving **$1,807:1$ compression against raw PCM** ($60.2\times$ spectral compression) at $83.86\text{ dB}$ tensor SNR.
- The recommended pipeline authors hypercubes with Tucker-HOSVD and converts or factors to CP-ALS for distribution and real-time playback.

### 2.3 How should decomposition rank be selected?

**Status:** Resolved (Empirical Benchmark — see [docs/empirical-findings.md §4 & §5](./empirical-findings.md#5-multi-dimensional-tensor-decomposition--real-time-polyphony-open-questions-22-25--51))

**Resolution:** Decomposition ranks are codified into standardized quality tiers based on empirical reconstruction plateau analysis:
- **Preview Tier:** CP $R=2$, Tucker $[2, 2, 2, 4]$, PCA $K=2$. Reaches $\ge 37.1\text{ dB}$ SNR with $5,421:1$ PCM compression ($180.7\times$ spectral).
- **Production Tier:** CP $R=6$, Tucker $[4, 4, 3, 8]$, PCA $K=4 \text{ to } 8$. Reaches $\ge 64.5\text{ to } 83.9\text{ dB}$ SNR with $1,807:1$ PCM compression ($60.2\times$ spectral).
- **Reference Tier:** CP $R=12$, Tucker $[8, 6, 4, 12]$, PCA $K=8 \text{ to } 16$. Reaches $\ge 93.7\text{ to } 95.9\text{ dB}$ SNR with $903:1$ PCM compression ($30.1\times$ spectral).
Automated rank selection uses an explained variance / SNR threshold plateau criterion (e.g. $K=8$ for PCA where error plateaus).

### 2.4 How effective is shared component extraction?

The specification (Section 4.4) proposes extracting shared spectral components across articulations to reduce redundancy. For example, the sustain phase of several bowing techniques on the same pitch and dynamic may be nearly identical and could be stored once.

The effectiveness of this approach depends on how much overlap actually exists in practice. It may work well for some instrument families (bowed strings with many similar sustain techniques) and poorly for others (a piano has fewer articulation types, and the differences between them are more fundamental).

**Needed:** Quantitative analysis of spectral similarity across articulations for several instrument types. Measure what fraction of the hypercube content can be factored into shared components vs. articulation-specific residuals.

**Relevant expertise:** Audio analysis, music production (understanding of what articulations exist and how they relate).

### 2.5 What compression ratios are achievable in practice?

**Status:** Resolved (Empirical Benchmark — see [docs/empirical-findings.md §5](./empirical-findings.md#5-multi-dimensional-tensor-decomposition--real-time-polyphony-open-questions-22-25--51))

**Resolution:** Measured empirical compression ratios on a 4-mode instrument tensor ($16 \times 12 \times 6 \times 16$, 72 notes, 552,960 PCM samples):
- **Preview ($R=2$):** **$5,421:1$ against raw 16-bit PCM** ($180.7\times$ against raw spectral frames) at $37.13\text{ dB}$ tensor SNR.
- **Production ($R=6$):** **$1,807:1$ against raw PCM** ($60.2\times$ spectral compression) at $83.86\text{ dB}$ tensor SNR.
- **Reference ($R=12$):** **$903:1$ against raw PCM** ($30.1\times$ spectral compression) at $95.93\text{ dB}$ tensor SNR.
- **Tucker Production ($[4, 4, 3, 8]$):** **$861:1$ against raw PCM** ($28.7\times$ spectral compression) at $64.51\text{ dB}$ tensor SNR.
These empirical results confirm and exceed the initial speculative estimates of 100:1 to 1000:1.

---

## 3. Perceptual quality

### 3.1 What objective metrics best predict subjective quality for spectral compression?

**Status:** Resolved (Empirical Benchmark — see [docs/empirical-findings.md §2](./empirical-findings.md#2-phase-strategy-comparison-open-questions-12--31))

**Resolution:** Codify **Log-Spectral Distance (LSD)** and **Spectral Convergence** (or Multi-Resolution STFT Loss) as the normative objective metrics. Time-domain waveform SNR and Pearson correlation $r$ fail as valid quality predictors when phase is unconstrained or deliberately dispersed: under Phase Dispersion Modeling, waveform SNR drops to $-4.52\text{ dB}$ and Pearson $r$ to $-0.4225$, despite perfect timbral preservation with LSD $< 0.98\text{ dB}$ and Spectral Convergence $< 0.0095$ (< 1% spectral magnitude error).

### 3.2 How does quality degrade in a musical mix vs. in isolation?

A compressed instrument sample may have audible artifacts when listened to solo but be perfectly acceptable in the context of a full musical arrangement where masking from other instruments hides the artifacts. Understanding this relationship is important for setting realistic quality targets.

If Production-tier compression is transparent in a mix context but slightly audible in isolation, that may be acceptable for the majority of use cases (most instrument samples are heard in context, not solo).

**Needed:** Listening tests comparing isolated and in-context presentations of Calyx-compressed audio at various quality tiers.

**Relevant expertise:** Music production, psychoacoustics, mixing engineering.

### 3.3 Which instrument types are most and least tolerant of spectral compression?

Different instruments have different spectral characteristics that affect compressibility. Instruments with simple, stable harmonic spectra (flute, clarinet in the chalumeau register) may compress well. Instruments with complex transients (piano), rapidly evolving spectra (brass with fast articulations), or significant noise content (nylon guitar, breathy flute) may be more challenging.

**Needed:** Systematic evaluation across instrument families. Identify which types are good early candidates for Calyx (likely to compress well and sound good) and which require additional modeling work.

**Relevant expertise:** Acoustics, orchestration, audio engineering.

---

## 4. Container format and compatibility

### 4.1 What metadata must the container format preserve?

For Calyx to serve as a distribution format, the .calyx file must carry enough metadata to reconstruct a functional instrument, not just the audio content. This includes key mapping, velocity curves, round robin configuration, release trigger behavior, loop points (if applicable to the compressed representation), and articulation definitions.

Different source formats (SFZ, SF2, Kontakt NKI) encode this metadata differently and support different feature sets.

**Needed:** Survey the metadata fields in SFZ 2.0, SF2, and (to the extent possible) Kontakt NKI. Define a Calyx metadata schema that can represent the common subset and accommodate format-specific extensions. Determine which metadata fields are essential for basic playback vs. which are needed for full behavioral fidelity.

**Relevant expertise:** Sampler format specifications, instrument design, plugin development.

### 4.2 How should the container format support versioning and extensibility?

The format will evolve. New compression methods, additional hypercube axes, and unforeseen use cases will emerge. The container format must allow older readers to handle newer files gracefully (skipping sections they don't understand) and must not break forward compatibility unnecessarily.

**Needed:** Design a section-based container structure with version tagging, inspired by existing extensible formats (PNG chunks, RIFF/WAV chunks, Matroska/EBML). Define rules for required vs. optional sections and for backward/forward compatibility.

**Relevant expertise:** File format design, binary format engineering.

### 4.3 Should the format support partial loading?

For large instruments (a full piano with all dynamic layers and pedal variants), it might be useful to load only a subset of the hypercube (e.g., a single dynamic layer, or a limited pitch range) without parsing the entire file. This requires the container format to support random access or indexed sections.

**Needed:** Evaluate whether partial loading is important enough to justify the added complexity in the container format. Consider how often users would want a subset of an instrument vs. the full thing.

**Relevant expertise:** File format design, sampler engine development, music production workflow.

---

## 5. Real-time synthesis

### 5.1 What is the computational cost of hypercube reconstruction?

**Status:** Resolved (Empirical Benchmark — see [docs/empirical-findings.md §5](./empirical-findings.md#5-multi-dimensional-tensor-decomposition--real-time-polyphony-open-questions-22-25--51))

**Resolution:** Empirical single-frame evaluation latency (reconstructing all 16 partial amplitudes at a single coordinate $(t, p, d)$) at $100\text{ Hz}$ frame rate:
- **CP-ALS:** Requires only $\mathcal{O}(K \cdot R)$ operations.
  - Preview ($R=2$): **$0.10\text{ }\mu\text{s}$** per frame $\to$ **$>45,000$ voices** under 50% single-core CPU budget.
  - Production ($R=6$): **$0.30\text{ }\mu\text{s}$** per frame $\to$ **$>16,000$ voices** under 50% single-core CPU budget.
  - Reference ($R=12$): **$0.70\text{ }\mu\text{s}$** per frame $\to$ **$>7,000$ voices** under 50% single-core CPU budget.
- **Tucker-HOSVD:** Requires multi-mode tensor contraction ($\mathcal{O}(\prod R_m)$).
  - Preview ($[2, 2, 2, 4]$): $2.66\text{ }\mu\text{s}$ per frame $\to$ $1,879$ voices.
  - Production ($[4, 4, 3, 8]$): $33.64\text{ }\mu\text{s}$ per frame $\to$ $148$ voices.
  - Reference ($[8, 6, 4, 12]$): $168.48\text{ }\mu\text{s}$ per frame $\to$ $28$ voices.
Because CP-ALS frame evaluation takes only $0.30\text{ }\mu\text{s}$ in Production tier, real-time voice evaluation is not CPU-bound by hypercube reconstruction, unlocking massive orchestral voice counts. Tucker decomposition is reserved for offline authoring or lower-polyphony applications.

For real-time playback (Use Case 3), the synthesis engine must reconstruct spectral frames from the compressed representation at the spectral frame rate (50-200 Hz) for each active voice. The cost depends on the decomposition method and rank.

### 5.2 How should the additive oscillator bank be implemented for efficiency?

Real-time additive synthesis requires generating and summing N sinusoidal partials per voice at the audio sample rate. For 64 partials and 16 voices, that is 1,024 oscillators updating at 44,100 to 96,000 Hz. This is computationally feasible on modern hardware but not trivial, especially if many voices are active simultaneously.

Optimization strategies include: SIMD vectorization of oscillator banks, wavetable-based oscillators instead of per-sample trigonometric functions, group-based synthesis where correlated partials are generated together, and inverse FFT resynthesis (computing a spectral frame and transforming to time domain via IFFT rather than summing individual sinusoids).

**Needed:** Benchmark different oscillator bank strategies in Rust. Determine which approach gives the best performance/quality tradeoff for the expected partial counts and voice counts.

**Relevant expertise:** Real-time audio programming, DSP optimization, SIMD programming.

### 5.3 How should transitions between notes be handled?

When a new note begins (note-on event), the synthesis engine must locate the appropriate region of the hypercube and begin reconstruction. For legato transitions, the engine may need to interpolate between the release phase of the previous note and the attack phase of the new note along the pitch axis.

Conventional samplers handle this with dedicated legato samples or scripted crossfades. Calyx's continuous representation offers the possibility of smoother transitions, but the interpolation behavior during transitions needs design attention.

**Needed:** Define and prototype transition behavior for monophonic legato, polyphonic note overlaps, and articulation changes. Evaluate perceptual quality of interpolated transitions vs. conventional crossfade approaches.

**Relevant expertise:** Instrument design, sampler engine development, real-time audio programming.

---

## 6. Intellectual property

### 6.1 Are there existing patents that constrain the design?

Spectral modeling synthesis, parametric audio coding, and audio compression are areas with significant patent activity, particularly around MPEG standards. Tensor decomposition methods themselves are generally unpatented (they are mathematical techniques), but specific applications of tensor methods to signal processing might be.

**Needed:** A patent landscape survey covering:

- Spectral modeling synthesis patents (particularly those arising from MPEG-4 standardization)
- Parametric audio coding patents (HILN and related)
- Additive resynthesis patents (most older patents will have expired, but newer methods may be active)
- Audio compression using dimensionality reduction or learned representations
- Any patents specifically covering compressed sample library formats

This survey does not need to be exhaustive at this stage, but should identify any obvious blocking patents that would require design modifications.

**Relevant expertise:** Patent search, intellectual property law, audio technology industry knowledge.

### 6.2 Should Calyx pursue a defensive patent strategy?

If Calyx's specific combination of spectral analysis, tensor decomposition, and hypercube-structured compression is novel (as the literature review suggests it may be), it could potentially be patented. Filing a defensive patent (or publishing a detailed technical disclosure to establish prior art) could protect the open source community from future patent claims by others.

The Apache 2.0 license includes a patent grant from contributors, but this does not prevent third parties from filing patents on similar techniques.

**Needed:** Evaluate whether a defensive patent filing, a formal prior art publication, or simply detailed public documentation is the best strategy for keeping this design space open.

**Relevant expertise:** Open source IP strategy, patent law, prior art documentation.

---

## 7. Ecosystem and adoption

### 7.1 What is the minimum viable integration with existing tools?

For Use Cases 1 and 2 (distribution and storage), Calyx must integrate with existing workflows. The minimum path is: vendor compresses to .calyx, consumer decompresses to SFZ + WAV, loads in any SFZ-compatible sampler.

But there are many possible integration points beyond this minimum: a Kontakt import/export script, a DAW plugin that loads .calyx files directly, integration with Decent Sampler's format, a command-line tool for batch conversion, a web-based preview player.

**Needed:** Survey potential early adopters (indie sample developers, open source sampler projects) to understand which integration points would provide the most value for the least implementation effort.

**Relevant expertise:** Music production workflow, plugin development, sample library development.

### 7.2 What governance model should the project adopt as it grows?

The specification, literature review, and CONTRIBUTING guide will establish initial norms. If the project attracts contributors, decisions about specification changes, release processes, and design direction will need a more formal structure.

Models to consider: benevolent dictator (current default), small steering committee, or foundation-style governance (Apache Foundation, Surge Synth Team model).

**Needed:** This question is premature until the project has active contributors. For now, document the decision-making process transparently and revisit when the community grows beyond a handful of people.

**Relevant expertise:** Open source community management, project governance.

---

*New questions should be added as they arise during research and development. Resolved questions should be moved to a "Resolved" section at the bottom of this document (or to the relevant specification section) with a summary of the resolution and any supporting evidence.*
