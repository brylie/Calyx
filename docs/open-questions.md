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

The number of tracked partials (N) directly affects both the fidelity of the representation and the size of the uncompressed hypercube. Too few partials and the reconstruction sounds thin or synthetic. Too many and the data volume grows without perceptual benefit.

The answer likely varies by instrument type. A flute might be well-represented by 16 to 24 partials. A piano or bowed string with rich upper harmonics might need 64 to 128. Percussive or inharmonic instruments may not decompose cleanly into partials at all.

**Needed:** Empirical analysis of partial counts vs. reconstruction quality across several instrument types. A sweep from 16 to 128 partials, measuring spectral convergence and conducting informal listening tests, would establish practical guidelines.

**Relevant expertise:** Audio DSP, spectral analysis, psychoacoustics.

### 1.2 How should phase information be handled?

The specification (Section 6.1) outlines four possible approaches to phase: harmonic phase locking, stored phase envelopes, phase dispersion modeling, and analysis-resynthesis phase recovery. Each has different implications for data size, computational cost, and perceptual quality.

Harmonic phase locking is the simplest and cheapest but may sound sterile. Stored phase envelopes preserve the original character but roughly double the data per spectral frame. Phase dispersion modeling and phase recovery are intermediate approaches that need experimentation to evaluate.

**Needed:** Comparative listening tests using the same source material resynthesized with each phase strategy. Particular attention should be paid to solo instrument passages where phase artifacts would be most exposed.

**Relevant expertise:** Additive synthesis, phase vocoder techniques, psychoacoustics.

### 1.3 What spectral frame rate is sufficient?

The analysis frame rate (frames per second for spectral envelope updates) determines how finely the time axis of the hypercube is sampled. Higher rates capture faster timbral changes but increase data volume linearly along the time axis.

For sustained tones, spectral content changes slowly and 50 frames/sec may be adequate. For fast attacks and transients, 200 frames/sec or higher might be needed. A variable frame rate (dense during attacks, sparse during sustains) could optimize the tradeoff but adds complexity to the data structure.

**Needed:** Analysis of spectral rate of change for different instrument types and playing techniques. Identify the minimum frame rate that avoids audible temporal aliasing for each category.

**Relevant expertise:** Audio DSP, signal processing.

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

PCA is the most straightforward approach for reducing the dimensionality of partial amplitude vectors, but it allows negative components, which are physically meaningless for amplitudes. NMF constrains both basis and coefficients to be non-negative, producing more interpretable decompositions, but NMF is computationally more expensive and its solutions are not unique.

Other candidates include sparse coding (overcomplete dictionaries with sparse activations) and autoencoders (learned nonlinear dimensionality reduction).

**Needed:** Head-to-head comparison of PCA, NMF, and at least one sparse method on spectral data extracted from real instrument samples. Compare reconstruction error, compression ratio, computational cost, and subjective quality.

**Relevant expertise:** Linear algebra, machine learning, audio DSP.

### 2.2 Which tensor decomposition method is best suited?

The specification mentions Tucker, CP, and tensor train decompositions. Each has different properties regarding compression efficiency, computational cost for random access (reconstructing a single spectral frame), uniqueness of the decomposition, and scalability to higher dimensions.

For the initial five-axis hypercube (time, pitch, dynamic, articulation, variation), Tucker or CP may be adequate. If additional axes are added later (microphone position, room, player), tensor train may scale better.

**Needed:** Empirical comparison of Tucker, CP, and tensor train decompositions on a populated spectral hypercube from the PoC. Key metrics: compression ratio at equivalent reconstruction error, time to reconstruct a single spectral frame, time to compute the decomposition.

**Relevant expertise:** Tensor methods, numerical linear algebra, scientific computing.

### 2.3 How should decomposition rank be selected?

The rank (number of components) in any tensor decomposition controls the tradeoff between compression and fidelity. Too low a rank discards perceptually important detail. Too high a rank wastes storage on imperceptible variation.

Rank selection could be guided by: explained variance thresholds (retain components explaining 99% of variance), perceptual metrics (increase rank until a quality metric plateaus), or cross-validation (hold out samples and measure reconstruction error on unseen data).

**Needed:** Develop a principled rank selection procedure, ideally one that can be automated for different instruments and quality tiers. Validate that the selected ranks produce perceptually acceptable results.

**Relevant expertise:** Statistical modeling, dimensionality reduction, perceptual audio evaluation.

### 2.4 How effective is shared component extraction?

The specification (Section 4.4) proposes extracting shared spectral components across articulations to reduce redundancy. For example, the sustain phase of several bowing techniques on the same pitch and dynamic may be nearly identical and could be stored once.

The effectiveness of this approach depends on how much overlap actually exists in practice. It may work well for some instrument families (bowed strings with many similar sustain techniques) and poorly for others (a piano has fewer articulation types, and the differences between them are more fundamental).

**Needed:** Quantitative analysis of spectral similarity across articulations for several instrument types. Measure what fraction of the hypercube content can be factored into shared components vs. articulation-specific residuals.

**Relevant expertise:** Audio analysis, music production (understanding of what articulations exist and how they relate).

### 2.5 What compression ratios are achievable in practice?

The use-cases document estimates 100:1 to 1000:1 compression ratios compared with PCM, but these are speculative. Actual ratios will depend on instrument type, number of partials retained, decomposition rank, and quality tier.

**Needed:** This is the central empirical question of the PoC. Measure compression ratios across at least three instrument types (e.g., piano, solo string, wind instrument) at each quality tier, paired with perceptual quality assessment.

**Relevant expertise:** Signal processing, audio engineering.

---

## 3. Perceptual quality

### 3.1 What objective metrics best predict subjective quality for spectral compression?

Standard audio quality metrics (SNR, PESQ, POLQA) were designed for speech or general audio and may not correlate well with perceived quality of spectrally compressed instrument samples. Spectral-domain metrics (spectral convergence, log-spectral distance) may be more appropriate, but their perceptual relevance for musical instrument tones is not well-established.

**Needed:** Correlate several objective metrics with subjective listening test results on Calyx-compressed instrument audio. Identify which metrics are most predictive and can be used for automated quality assessment during compression.

**Relevant expertise:** Psychoacoustics, audio quality evaluation, listening test methodology.

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

For real-time playback (Use Case 3), the synthesis engine must reconstruct spectral frames from the compressed representation at the spectral frame rate (50-200 Hz) for each active voice. The cost depends on the decomposition method and rank.

For Tucker decomposition, reconstruction involves multiplying the core tensor by factor vectors for each axis. For CP decomposition, it involves evaluating a sum of products. Both are dominated by the tensor rank and the number of axes.

**Needed:** Benchmark the reconstruction cost for each decomposition method at various ranks on representative hardware (a typical music production CPU, e.g., Intel i7/i9 or Apple M-series). Determine the maximum polyphony achievable within a given CPU budget.

**Relevant expertise:** Numerical computing, real-time audio programming, performance optimization.

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
