# Calyx Literature Review

**Status:** In progress
**License:** CC-BY-4.0

This document surveys prior work relevant to the Calyx project. It covers spectral analysis and modeling, audio compression, tensor and matrix decomposition methods, existing sampler technology, and open source audio projects. Each section summarizes the relevant ideas, notes what Calyx can borrow or build on, and identifies gaps or divergences.

Entries marked with **[TODO]** need further research, sourcing, or verification.

---

## 1. Spectral modeling synthesis

### 1.1 Sinusoidal plus residual models

The foundational framework for Calyx's spectral decomposition is the sinusoidal plus residual (S+R) model introduced by Serra and Smith.

**Serra, X. and Smith, J.O. (1990).** "Spectral Modeling Synthesis: A Sound Analysis/Synthesis System Based on a Deterministic plus Stochastic Decomposition." *Computer Music Journal* 14(4), 12-24.

This paper establishes the core idea of decomposing audio into a deterministic component (a sum of time-varying sinusoidal partials) and a stochastic component (the residual after subtracting the sinusoidal reconstruction). The deterministic component captures pitched, harmonic content; the stochastic component captures noise, breath, bow friction, and other aperiodic elements.

**Relevance to Calyx:** This decomposition maps directly onto the spectral frame structure defined in the Calyx specification (Section 2.2). The deterministic/stochastic split is essential because the two components have fundamentally different statistical properties and require different compression strategies. Partial amplitudes are smooth, low-bandwidth signals with strong cross-partial correlations (good candidates for PCA/NMF). The stochastic residual is better represented by a spectral envelope model in perceptual frequency bands.

**Serra, X. (1997).** "Musical Sound Modeling with Sinusoids plus Noise." In *Musical Signal Processing*, Swets & Zeitlinger.

Extends the SMS framework with more detailed treatment of the noise component and its spectral envelope representation. Introduces the idea of modeling noise as filtered white noise shaped by a time-varying spectral envelope.

**[TODO]** Locate and review the original SMS software distribution and its analysis/resynthesis pipeline. Understanding the practical implementation choices (frame rate, partial tracking algorithm, residual extraction method) will inform the Calyx analysis pipeline design.

### 1.2 Partial tracking algorithms

Accurate partial tracking is a prerequisite for populating the spectral hypercube. Several algorithms are relevant.

**McAulay, R.J. and Quatieri, T.F. (1986).** "Speech Analysis/Synthesis Based on a Sinusoidal Representation." *IEEE Transactions on Acoustics, Speech, and Signal Processing* 34(4), 744-754.

Introduces the peak-matching approach to sinusoidal tracking: detect spectral peaks in each frame, then link peaks across frames based on frequency proximity. This is the basis for most subsequent partial tracking methods.

**[TODO]** Review more recent partial tracking methods, particularly:

- Frequency-domain reassignment methods for improved frequency resolution
- Probabilistic and Bayesian approaches to partial tracking
- Partial tracking implementations in existing open source tools (Loris, SPEAR, libsms)

### 1.3 Phase vocoder techniques

Phase vocoders provide tools for time-frequency analysis and resynthesis that are relevant both to the analysis pipeline and to phase reconstruction during playback.

**Dolson, M. (1986).** "The Phase Vocoder: A Tutorial." *Computer Music Journal* 10(4), 14-27.

Classic introduction to phase vocoder analysis and resynthesis. Relevant for understanding phase propagation and the relationship between instantaneous frequency and phase advancement.

**Laroche, J. and Dolson, M. (1999).** "Improved Phase Vocoder Time-Scale Modification of Audio." *IEEE Transactions on Speech and Audio Processing* 7(3), 323-332.

Addresses phase coherence problems in phase vocoder resynthesis, particularly the "phasiness" artifacts that arise when phases are not properly propagated. Relevant to Calyx's phase handling strategies (specification Section 6.1).

**[TODO]** Survey phase reconstruction algorithms that could be used when Calyx stores amplitude-only spectral data and needs to synthesize plausible phase trajectories at playback time. Griffin-Lim and its variants are one starting point, but there may be more efficient approaches for the specific case of harmonic partial resynthesis.

### 1.4 IRCAM tools and research

IRCAM (Institut de Recherche et Coordination Acoustique/Musique) has produced extensive research and software tools for spectral analysis and sound modeling.

**[TODO]** Review the following IRCAM tools and associated publications:

- **AudioSculpt:** graphical interface for spectral analysis and transformation
- **SuperVP:** phase vocoder engine underlying AudioSculpt
- **PM2:** additive analysis/synthesis engine
- **Orchidea:** orchestration tool that uses spectral analysis for timbre matching

IRCAM's practical experience with spectral analysis of orchestral instruments is directly relevant. Their published analyses of instrument timbres across pitch, dynamic, and technique could inform expectations for Calyx's compression performance.

---

## 2. Additive synthesis

### 2.1 History and implementations

Additive synthesis has a long history in both hardware and software. Understanding prior implementations helps identify practical constraints and design patterns.

**[TODO]** Survey the following systems and their approaches:

- **Kawai K5 / K5000 series:** hardware additive synthesizers using harmonic amplitude envelopes. Notable for achieving real-time additive synthesis with 1990s-era hardware.
- **Cameleon 5000 (Symbolic Sound):** Kyma-based additive synthesis with sophisticated spectral modeling capabilities.
- **Native Instruments Razor:** real-time additive synthesizer with extensive modulation. Demonstrates the expressive potential of direct partial control.
- **AIR Music Technology Loom / Loom II:** additive synthesizer with morphing capabilities.

### 2.2 Resynthesis from analysis data

Several tools and systems focus specifically on resynthesizing audio from spectral analysis data, which is close to Calyx's playback use case.

**Fitz, K. and Haken, L.** "Loris: An Open Source Sound Modeling and Morphing Environment."

Loris is an open source C++ library for analysis, manipulation, and resynthesis of audio using reassigned bandwidth-enhanced sinusoidal models. It extends the basic sinusoidal model with bandwidth enhancement to better capture noisy components.

**Relevance to Calyx:** Loris's bandwidth-enhanced model is an alternative to the strict deterministic/stochastic split. Rather than separating noise into a separate component, it represents each partial as a sinusoid with a bandwidth parameter (effectively a narrow noise band around each partial). This could simplify the spectral frame representation at the cost of some modeling accuracy for broadband noise.

**Klingbeil, M. (2005).** "Software for Spectral Analysis, Editing, and Synthesis." *Proceedings of the International Computer Music Conference.*

SPEAR (Sinusoidal Partial Editing Analysis and Resynthesis) is a widely used GUI tool for partial tracking and editing. Its interface and workflow offer useful design reference for how spectral analysis results are presented and manipulated.

**[TODO]** Evaluate Loris and SPEAR as potential tools in the Calyx analysis pipeline, or as reference implementations for partial tracking algorithms.

---

## 3. Dimensionality reduction and matrix factorization

### 3.1 Principal Component Analysis (PCA)

**Jolliffe, I.T. (2002).** *Principal Component Analysis.* 2nd edition. Springer.

Standard reference for PCA. Relevant chapters: the mathematical formulation of PCA as eigendecomposition of the covariance matrix, choosing the number of components (scree plots, explained variance thresholds), and computational methods for large datasets.

**Relevance to Calyx:** PCA is the most straightforward approach for reducing the dimensionality of partial amplitude vectors. If 64 partials can be represented by 8 to 12 principal components with minimal perceptual loss, the compression within each spectral frame is approximately 5:1 to 8:1 before any cross-frame or cross-axis compression is applied.

**Limitation:** PCA components can be negative, which is physically meaningless for partial amplitudes (amplitudes are non-negative). This may not matter for reconstruction accuracy, but it complicates interpretation and may affect compression efficiency compared with non-negative methods.

### 3.2 Non-negative Matrix Factorization (NMF)

**Lee, D.D. and Seung, H.S. (1999).** "Learning the Parts of Objects by Non-negative Matrix Factorization." *Nature* 401, 788-791.

Foundational paper on NMF. Demonstrates that constraining both the basis and coefficient matrices to be non-negative produces "parts-based" representations where basis vectors correspond to recognizable components of the data.

**Lee, D.D. and Seung, H.S. (2001).** "Algorithms for Non-negative Matrix Factorization." *Advances in Neural Information Processing Systems* 13.

Introduces the multiplicative update rules for NMF that are still widely used.

**Smaragdis, P. and Brown, J.C. (2003).** "Non-negative Matrix Factorization for Polyphonic Music Transcription." *IEEE Workshop on Applications of Signal Processing to Audio and Acoustics.*

Demonstrates NMF applied to spectral data from musical audio. The learned basis vectors correspond to individual note spectra, and the activation matrix indicates when each note is active.

**Relevance to Calyx:** NMF is arguably a better fit than PCA for spectral data because partial amplitudes are inherently non-negative. The "parts-based" interpretation means NMF basis vectors can correspond to meaningful spectral shapes (e.g., "bright attack spectrum," "mellow sustain spectrum") that combine additively. This aligns naturally with how instrument timbres are constructed from harmonic partials.

**[TODO]** Investigate:

- Sparse NMF variants that encourage compact representations
- Online NMF algorithms suitable for processing large datasets incrementally
- Comparison of PCA vs. NMF reconstruction quality on spectral data from musical instruments

### 3.3 Other decomposition methods

**[TODO]** Review the following as potential alternatives or supplements:

- **Independent Component Analysis (ICA):** finds statistically independent components. May reveal different structure than PCA/NMF.
- **Sparse coding / dictionary learning:** learns an overcomplete dictionary of spectral atoms with sparse activations. Could provide high compression if instrument spectra are well-represented by a small number of active atoms from a larger dictionary.
- **Autoencoders:** neural network-based dimensionality reduction. More flexible than linear methods but harder to interpret and potentially slower to evaluate at runtime.

---

## 4. Tensor decomposition

### 4.1 Foundational methods

**Kolda, T.G. and Bader, B.W. (2009).** "Tensor Decompositions and Applications." *SIAM Review* 51(3), 455-500.

Comprehensive survey of tensor decomposition methods, including CP decomposition, Tucker decomposition, and their properties. Essential reading for understanding the mathematical framework and computational considerations.

**Relevance to Calyx:** The spectral hypercube is a multi-dimensional tensor, and tensor decomposition is the primary mechanism for compressing it. This survey provides the theoretical foundation for choosing between decomposition methods and understanding their tradeoffs.

### 4.2 CP (Canonical Polyadic) decomposition

Also known as CANDECOMP/PARAFAC. Represents a tensor as a sum of rank-one terms, each being an outer product of vectors along each mode.

**[TODO]** Review:

- Algorithms for computing CP decomposition (alternating least squares, gradient methods)
- Rank selection strategies (how to choose the number of components)
- Uniqueness properties (CP decomposition is essentially unique under mild conditions, unlike matrix factorization)
- Computational cost for reconstruction at a single point (relevant for real-time playback)

### 4.3 Tucker decomposition

Represents a tensor as a core tensor multiplied by factor matrices along each mode. More flexible than CP but introduces the core tensor, which can itself be large if not constrained.

**[TODO]** Review:

- Higher-Order Singular Value Decomposition (HOSVD) as a method for computing Tucker decomposition
- Truncated Tucker decomposition and its relationship to multi-linear PCA
- Compression ratios achievable for tensors with the expected structure of spectral hypercubes (smooth variation along each axis, strong correlations)

### 4.4 Tensor train and other formats

**Oseledets, I.V. (2011).** "Tensor-Train Decomposition." *SIAM Journal on Scientific Computing* 33(5), 2295-2317.

Tensor train (TT) decomposition represents a high-dimensional tensor as a chain of three-dimensional cores. It scales well with the number of dimensions and avoids the exponential growth ("curse of dimensionality") that affects Tucker decomposition in high dimensions.

**Relevance to Calyx:** The spectral hypercube has five or more axes. Tucker decomposition may become unwieldy as additional axes (microphone position, room, etc.) are added. Tensor train format could provide a more scalable alternative for high-dimensional hypercubes.

**[TODO]** Investigate:

- Practical implementations of tensor train decomposition (TT-Toolbox, tntorch)
- Random access efficiency: how quickly can a single element (spectral frame) be reconstructed from the TT representation?
- Comparison of Tucker vs. TT for tensors with 5 to 8 dimensions at the sizes expected for Calyx

### 4.5 Tensor decomposition in audio and signal processing

**[TODO]** Search for existing applications of tensor decomposition specifically to audio or music data. Areas to investigate:

- Tensor methods for multi-channel audio processing
- Tensor approaches to music information retrieval (MIR)
- Any prior work on tensor-based audio compression (this may be a gap in the literature, which would strengthen Calyx's novelty claim)

---

## 5. Audio compression and codecs

### 5.1 Perceptual audio coding

**Brandenburg, K. and Stoll, G. (1994).** "ISO/MPEG-1 Audio: A Generic Standard for Coding of High-Quality Digital Audio." *Journal of the Audio Engineering Society* 42(10), 780-792.

Overview of the MPEG-1 Layer III (MP3) standard. Relevant for its approach to perceptual masking: discarding spectral information that falls below the threshold of audibility.

**Relevance to Calyx:** Perceptual masking principles could inform quality tier design. At the Preview compression tier, Calyx could aggressively discard spectral detail that is perceptually masked, similar to how low-bitrate MP3 discards masked frequency components. The Production and Reference tiers would retain more detail.

### 5.2 Parametric audio coding

**MPEG-4 Part 3: Audio, Subpart 6.** Harmonic and Individual Lines plus Noise (HILN) parametric audio coder.

HILN represents audio as a combination of harmonic tones, individual sinusoidal lines, and noise components. This is conceptually close to the Calyx spectral frame model.

**[TODO]** Review the HILN specification and related publications in detail. Specific questions:

- How does HILN handle the sinusoidal/noise decomposition?
- What compression ratios does it achieve for tonal vs. noisy content?
- What are its perceptual quality limitations?
- Are there patents that constrain the design space? (MPEG-4 is heavily patented.)

### 5.3 Neural audio codecs

**Défossez, A. et al. (2022).** "High Fidelity Neural Audio Compression." arXiv:2210.13438. (EnCodec, Meta)

**Zeghidour, N. et al. (2021).** "SoundStream: An End-to-End Neural Audio Codec." arXiv:2107.03312. (Google)

Neural audio codecs use learned encoder-decoder networks with quantized latent representations to compress audio at very low bitrates while maintaining high perceptual quality.

**Relevance to Calyx:** These systems demonstrate that extremely high compression of audio is possible with learned representations. They operate on general audio rather than exploiting the specific structure of sampled instruments, which suggests Calyx's domain-specific approach could achieve comparable or better compression ratios for instrument content specifically. However, neural codecs have the advantage of handling arbitrary audio (transients, noise, speech, music) without requiring a harmonic decomposition.

**Divergence from Calyx:** Neural codecs produce opaque latent representations that are not interpretable or editable in musically meaningful ways. Calyx's explicit spectral representation is designed to be transparent: you can inspect and modify partial amplitudes, adjust articulation blending, and understand what the compression is doing. This transparency is a design goal, not just a side effect.

**[TODO]** Benchmark EnCodec and SoundStream on instrument sample content to establish a quality/compression baseline that Calyx should aim to match or exceed for its target domain.

### 5.4 Lossless and near-lossless audio compression

**[TODO]** Brief review of FLAC, ALAC, and WavPack as reference points for the archival end of the compression spectrum. Calyx is explicitly lossy, but understanding what lossless compression achieves on instrument samples (typically 50-70% of PCM size) helps contextualize the value of Calyx's much higher compression ratios.

---

## 6. Differentiable Digital Signal Processing (DDSP)

**Engel, J. et al. (2020).** "DDSP: Differentiable Digital Signal Processing." *ICLR 2020.*

DDSP uses neural networks to predict the parameters of a differentiable synthesizer (additive oscillators plus filtered noise) from audio input. The key insight is that by building DSP knowledge (oscillator banks, filters) into the architecture, the network can learn to control synthesis parameters rather than generating audio sample by sample. This produces high-quality resynthesis with very compact models.

**Relevance to Calyx:** DDSP demonstrates that additive-synthesis-based audio representation can achieve high fidelity with compact parameterization. The DDSP encoder effectively learns a compressed spectral representation, which is conceptually similar to what Calyx constructs through explicit analysis and tensor decomposition. DDSP's published results on instrument resynthesis (violin, flute, trumpet) provide useful quality benchmarks.

**Key differences:**

- DDSP learns its representation end-to-end via gradient descent. Calyx constructs its representation through explicit spectral analysis and algebraic decomposition.
- DDSP's latent space is not structured around musically meaningful axes (pitch, dynamic, articulation). Calyx's hypercube is explicitly organized along these axes.
- DDSP is optimized for single-instrument, single-condition resynthesis. Calyx is designed to represent the full variation space of a multi-sampled instrument.

**Engel, J. et al. (2020).** "Self-Supervised Pitch Detection by Inverse Audio Synthesis." *ICML 2020 Workshop on Self-Supervision in Audio and Speech.*

Uses DDSP-style resynthesis loss for self-supervised pitch estimation. Potentially relevant to the Calyx analysis pipeline's pitch detection stage.

**[TODO]** Review subsequent DDSP-related work:

- DDSP-based timbre transfer and instrument modeling
- Multi-instrument DDSP extensions
- DDSP efficiency and real-time performance characteristics

---

## 7. Sample library technology

### 7.1 Current sampler formats and engines

Understanding the current state of sampler technology is necessary to design Calyx for practical compatibility and to quantify the baseline it aims to improve upon.

**SFZ format.**
Open, text-based instrument definition format. Defines sample mappings, key ranges, velocity layers, round robins, and playback behavior. No compression of audio content; references external audio files (WAV, FLAC, OGG).

**[TODO]** Review the SFZ 2.0 specification and the ARIA engine extensions. Identify the metadata fields that would need equivalents in the Calyx container format.

**SF2 / SF3 (SoundFont).**
Binary format combining instrument definitions and embedded audio. SF2 uses uncompressed PCM; SF3 uses OGG Vorbis compression. The FluidR3 GM SoundFont (public domain) is a widely used open source implementation.

**Relevance to Calyx:** SF2/SF3 files with their embedded audio and metadata are good candidates for the initial PoC input pipeline, since they are self-contained and well-documented.

**Native Instruments Kontakt.**
The de facto industry standard for commercial sample libraries. Proprietary format (NKI/NKR/NCW). NCW is a lossless audio compression format that typically achieves 50% reduction from WAV. Kontakt supports complex scripting for articulation switching, round robin logic, and performance behavior.

**[TODO]** Research:

- NCW compression characteristics and how they compare with FLAC
- Kontakt's memory management: how it handles streaming vs. RAM residency
- The structure of NKI instrument definitions as reference for Calyx metadata design

**Other engines:**

- **Decent Sampler:** free, open-format sampler gaining traction for indie sample developers
- **sforzando / sfizz:** SFZ-compatible players
- **EXS24 / Sampler (Apple Logic):** proprietary but widely used

### 7.2 Commercial sample library landscape

**[TODO]** Document the current scale of commercial sample libraries to support the problem statement's quantitative claims:

- Survey library sizes from major vendors (Spitfire Audio, Orchestral Tools, Vienna Symphonic Library, Native Instruments, EastWest)
- Typical file sizes for orchestral, piano, and other instrument categories
- Distribution methods and download infrastructure used by vendors
- Customer pain points around storage and download times (forum discussions, reviews)

### 7.3 Multi-sample organization

**[TODO]** Document how existing sample libraries organize their content across the dimensions that Calyx models as hypercube axes:

- Velocity layers and dynamic crossfading
- Articulation switching (keyswitches, UACC, expression maps)
- Round robin and random variation
- Microphone positions and mix options
- Release triggers and legato transitions

Understanding these organizational patterns will inform both the analysis pipeline (how to ingest existing content) and the container format (what metadata to preserve).

---

## 8. Open source audio projects and ecosystem

### 8.1 Surge Synth Team

The Surge Synth Team maintains several open source audio plugins and provides a governance model relevant to Calyx.

**Surge XT:** open source hybrid synthesizer (C++, GPLv3). Notable for its active community, structured contribution process, and quality standards.

**Shortcircuit XT:** open source creative sampler, rebuilt from the original Shortcircuit 2 codebase using modern C++ and JUCE. Currently in alpha/beta. Relevant as a modern open source sampler implementation.

**Stochas:** open source probabilistic sequencer.

**Relevance to Calyx:** The Surge Synth Team's governance model (community Discord, contribution guidelines, release process) is a reference for how Calyx might structure its own community as it grows. Their experience managing open source audio projects with both developer and musician contributors is directly applicable.

**[TODO]** Review the Surge Synth Team's governance documentation and contribution guidelines in detail.

### 8.2 Rust audio ecosystem

**[TODO]** Survey the current state of Rust crates relevant to Calyx's implementation:

- **cpal:** cross-platform audio I/O
- **dasp:** digital audio signal processing primitives (sample types, conversions, ring buffers)
- **fundsp:** audio DSP library with a functional programming style; includes oscillators, filters, and signal routing
- **hound:** WAV file reading/writing
- **rodio:** audio playback library built on cpal
- **rubato:** sample rate conversion

Evaluate maturity, maintenance status, and suitability for Calyx's needs. Identify gaps where Calyx may need to implement its own components (e.g., partial tracking, SF2 parsing, additive synthesis oscillator bank).

### 8.3 Rust numerical computing ecosystem

**[TODO]** Survey Rust crates for the linear algebra and tensor operations Calyx needs:

- **ndarray:** N-dimensional array library; Rust's rough equivalent to NumPy
- **nalgebra:** linear algebra library (matrices, decompositions)
- **ndarray-linalg:** linear algebra operations on ndarray arrays (wraps LAPACK)
- **tch-rs:** Rust bindings for PyTorch (includes tensor operations, but heavy dependency)

Evaluate whether existing crates support the specific operations needed (SVD, NMF, Tucker decomposition, tensor train) or whether Calyx will need to implement these. Tensor decomposition support in Rust is likely sparse compared with Python (tensorly, tensortools).

### 8.4 Other relevant open source projects

**Loris.** Open source C++ library for bandwidth-enhanced sinusoidal analysis and resynthesis. Potential reference implementation or direct dependency for the analysis pipeline.

**libsms.** C library implementing Serra's SMS framework. Another candidate for reference or integration.

**SPEAR.** Sinusoidal Partial Editing Analysis and Resynthesis. GUI tool for spectral analysis and editing. Useful as a visual reference for understanding partial tracking output.

**[TODO]** Evaluate whether any of these C/C++ libraries could be called from Rust via FFI, or whether reimplementation in Rust is preferable for the PoC.

---

## 9. Perceptual quality evaluation

### 9.1 Objective metrics

**[TODO]** Survey objective audio quality metrics relevant to evaluating Calyx's compression quality:

- **Signal-to-Noise Ratio (SNR):** simple but poorly correlated with perceptual quality for spectral modifications
- **Spectral convergence:** measures the error in the spectral domain; more relevant than time-domain SNR for spectral compression
- **PESQ (Perceptual Evaluation of Speech Quality):** ITU standard for speech quality; applicability to music is limited
- **POLQA:** successor to PESQ; still speech-focused
- **ViSQOL:** Virtual Speech Quality Objective Listener; has a music mode (ViSQOLAudio) that may be more appropriate
- **MUSHRA:** Multiple Stimuli with Hidden Reference and Anchor; a subjective listening test methodology rather than an objective metric, but the standard approach for evaluating audio codec quality

### 9.2 Perceptual quality in musical context

**[TODO]** Investigate how perceptual quality should be evaluated specifically for sampled instrument libraries:

- Quality in isolation vs. quality in a musical mix (instruments are rarely heard solo in production)
- Sensitivity to compression artifacts in different frequency ranges and instrument types
- The role of musical context in masking compression artifacts
- Existing literature on perceptual thresholds for spectral modification of musical tones

---

## 10. Patent landscape

### 10.1 Areas of concern

**[TODO]** Survey the patent landscape for potential conflicts with Calyx's design. Key areas:

- Spectral modeling synthesis patents (particularly any originating from MPEG standardization)
- Parametric audio coding patents (MPEG-4 HILN and related)
- Additive synthesis patents (likely expired for older methods, but newer variations may be active)
- Tensor decomposition applied to signal processing (less likely to be patented, but worth checking)
- Sample library compression patents (any vendor-specific methods)

### 10.2 Patent mitigation

Calyx's Apache 2.0 licensing includes a patent grant from contributors. However, third-party patents could still constrain the design. Strategies for mitigation:

- Design around known patents where possible
- Document prior art that could invalidate overly broad patent claims
- Consider defensive patent pledges if any Calyx innovations are independently patentable

**[TODO]** Consult resources on open source patent strategy, including the Open Invention Network and Apache Foundation patent policy documentation.

---

## 11. Gaps and opportunities

Based on the literature surveyed so far, the following appear to be areas where Calyx offers something not well-covered by existing work:

1. **Structured multi-dimensional spectral compression for instrument libraries.** Tensor decomposition has been applied to many signal processing problems, but its application to compressing multi-sampled instrument libraries specifically appears to be underexplored.

2. **Explicit, interpretable spectral representation as a distribution format.** Neural codecs achieve high compression but produce opaque latent representations. Calyx's spectral hypercube is designed to be transparent and editable.

3. **Domain-specific compression exploiting instrument physics.** General audio codecs don't exploit the strong structural redundancies present in multi-sampled instrument content (cross-pitch, cross-dynamic, cross-articulation correlations).

4. **Open format for instrument distribution.** The dominant sampler formats (Kontakt NKI, EXS24) are proprietary. SFZ is open but doesn't include compression. An open, compressed format would benefit the entire ecosystem.

**[TODO]** Validate these gap claims through more thorough literature search. If prior work exists in these areas, Calyx's contribution should be reframed accordingly.

---

*This document is a living resource. Contributions of references, corrections, and additional relevant work are welcome via GitHub Issues and Pull Requests.*
