# Calyx Specification

**Version:** 0.1.0-draft
**Status:** Work in progress
**License:** This document is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) License](../LICENSE-content.md).

## Abstract

Calyx defines a compressed spectral representation format for multi-sampled instrument libraries. A sampled instrument's audio content is decomposed into harmonic partials, organized into a multi-dimensional tensor (spectral hypercube) indexed by musically meaningful parameters, and compressed using dimensionality reduction and tensor decomposition techniques. The resulting format is significantly smaller than equivalent PCM audio while preserving the perceptual character of the original instrument. The compressed representation can be decompressed back to conventional audio or played back directly by a compatible spectral synthesis engine.

## 1. Background and motivation

### 1.1 The cost structure of sampled instruments

Conventional sample libraries achieve expressive realism through brute-force coverage: recording many individual audio samples across multiple dimensions of variation. A single well-sampled orchestral instrument might include recordings at every semitone (or smaller intervals) across its full pitch range, at four to eight dynamic levels, with six to twelve articulation types, and three to six round robin variations per combination. A complete orchestral library multiplies this across dozens of instruments.

The resulting data volumes are substantial. Flagship orchestral libraries from major vendors routinely range from 100 to 500 gigabytes of PCM audio. A professional composer's collection of libraries may total several terabytes.

These volumes create costs at each stage:

**Production and hosting.** Vendors must store master copies and distribution-ready versions of all content on infrastructure that supports high-throughput delivery. Warm storage (cloud object storage with fast access) costs roughly $20 to $25 per terabyte per month at typical cloud pricing. CDN egress for delivering content to customers adds further cost per gigabyte transferred. For a vendor with a large catalog and active customer base, these are meaningful operational expenses.

**Distribution.** Customers download the full PCM content of each library. A 300 GB library on a 100 Mbps connection takes approximately seven hours to download. Many customers have slower connections or data caps that make large libraries impractical to acquire.

**Local storage.** Each library must reside on fast storage (SSD) for acceptable playback performance. Professional users routinely dedicate one or more high-capacity SSDs exclusively to sample libraries.

**Runtime memory.** During playback, sampler engines load PCM data into RAM. A full orchestral template with multiple articulations loaded per instrument can consume 32 to 64 GB of RAM or more, often representing the primary constraint on template complexity.

**Musical expression.** Conventional samplers switch between discrete sample layers using velocity splits, keyswitch-triggered articulation changes, and crossfade-based dynamic transitions. These transitions are inherently discontinuous. Crossfading between two recordings that were not performed as a continuous gesture produces audible artifacts, particularly in exposed solo passages. This limits the fluidity and realism of sampled instrument performance.

### 1.2 The redundancy in sampled content

The large data volumes in conventional libraries contain enormous redundancy that is not exploited by general-purpose compression (zip, FLAC, etc.). This redundancy is structural, rooted in the physics of how acoustic instruments produce sound:

**Cross-pitch redundancy.** A violin playing A4 and Bb4 produce nearly identical harmonic structures, shifted in frequency. The spectral envelope changes gradually across the instrument's range.

**Cross-dynamic redundancy.** A piano note played mezzo-forte and forte shares most of its harmonic content; the primary differences are in the relative amplitudes of upper partials and the character of the initial transient.

**Cross-articulation redundancy.** Many articulations share common components. The sustain portion of a legato violin note and a detaché note on the same pitch at the same dynamic are nearly identical; the differences are concentrated in the onset and release.

**Temporal redundancy.** Within a single note, the spectral content during the sustain phase changes slowly. Envelope shapes for individual partials are smooth and low-bandwidth signals.

A representation that operates in the spectral domain and is structured to align with these dimensions of variation can exploit all of these redundancies simultaneously.

### 1.3 Design goals

Calyx aims to:

1. Define a compressed representation that reduces instrument library sizes by one to three orders of magnitude compared with equivalent PCM content.
2. Preserve perceptual fidelity at configurable quality levels, with a production-quality tier that is indistinguishable from the original in typical musical contexts.
3. Support decompression back to PCM for compatibility with existing sampler engines.
4. Support direct playback from the compressed representation, enabling continuous interpolation across all parameter axes.
5. Remain open, unencumbered by patents or restrictive licensing, and implementable by anyone.

## 2. Conceptual model

### 2.1 The spectral hypercube

The core data structure in Calyx is a multi-dimensional tensor, referred to as a spectral hypercube, that encodes the spectral content of a sampled instrument across its full space of variation.

Each axis of the hypercube corresponds to a dimension of musical variation:

| Axis             | Description                                   | Typical range                    |
| ---------------- | --------------------------------------------- | -------------------------------- |
| Time (t)         | Spectral evolution within a single note event | ~100 frames/sec                  |
| Pitch (p)        | Fundamental frequency / MIDI note number      | 24 to 108 (instrument dependent) |
| Dynamic (d)      | Playing intensity / velocity                  | 4 to 16 levels                   |
| Articulation (a) | Playing technique                             | Variable per instrument          |
| Variation (v)    | Round robin or stochastic variation           | 1 to 8 variations                |

Each cell in the hypercube contains a spectral frame: a vector of values describing the sound's frequency content at that point in the parameter space.

### 2.2 Spectral frame contents

A spectral frame at position (t, p, d, a, v) contains:

**Deterministic component:**
- Partial amplitudes: a vector of N amplitude values, one per tracked harmonic partial (e.g., N = 64 or 128). These represent the instantaneous amplitude of each partial at this point in the parameter space.
- Partial frequency ratios (optional): deviation of each partial from exact harmonic spacing. Relevant for instruments with inharmonicity (piano, bells). For perfectly harmonic instruments, this can be omitted or stored as a per-pitch inharmonicity profile rather than per-frame.
- Phase relationships (optional): inter-partial phase offsets where perceptually relevant. See Section 6.1 for discussion of phase handling strategies.

**Stochastic component:**
- Spectral envelope of the noise/residual component, represented as a low-order filter shape (e.g., bark-band or ERB-band energy values). This captures aperiodic content such as bow noise, breath noise, hammer impact, and key mechanisms.
- Stochastic energy: overall level of the noise component relative to the deterministic component.

This decomposition into deterministic and stochastic components follows the sinusoidal plus residual model established in spectral modeling synthesis (Serra and Smith, 1990).

### 2.3 Relationship to existing paradigms

**Compared with wavetable synthesis:** a wavetable is a one-dimensional array of waveform snapshots, typically indexed by a single parameter (often a modulation knob or LFO). The spectral hypercube generalises this to N dimensions and operates on spectral frames rather than time-domain waveforms.

**Compared with conventional multi-sampling:** a sample library stores discrete PCM recordings indexed by pitch, dynamic, and articulation. The spectral hypercube stores a continuous, compressed spectral representation across these same dimensions, enabling interpolation rather than crossfading between discrete points.

**Compared with physical modeling:** physical modeling synthesises sound from mathematical models of instrument mechanics. Calyx is data-driven rather than physics-driven; it derives its representation from recorded audio rather than from physical equations. However, the compressed spectral representation could complement physical models by providing efficient timbral targets or excitation data.

## 3. Analysis pipeline

### 3.1 Input requirements

Calyx takes as input a conventional multi-sampled instrument: a collection of audio files with associated metadata describing each sample's pitch, dynamic level, articulation type, and variation index. Supported input formats include:

- SFZ instrument definitions with referenced audio files
- SF2 / SF3 soundfont files
- Kontakt instruments (via NKI/NKR parsing, where legally and technically feasible)
- Loose audio files with structured directory naming or accompanying metadata files

The metadata is essential because it defines the mapping from individual audio files to positions in the spectral hypercube.

### 3.2 Spectral analysis

Each input audio sample is analysed to extract its spectral content over time:

1. **Pitch detection:** estimate the fundamental frequency contour of the sample. For monophonic instrument samples with known pitch, this is straightforward. The fundamental provides the reference for harmonic partial tracking.

2. **Partial tracking:** identify and track individual harmonic partials across the duration of the sample. For each frame (at the analysis frame rate, typically 50 to 200 frames per second), estimate the amplitude and frequency of each partial.

3. **Residual extraction:** subtract the reconstructed deterministic (sinusoidal) component from the original signal to obtain the stochastic residual. Analyse the residual's spectral envelope in perceptual frequency bands.

4. **Frame assembly:** combine partial amplitudes, frequency deviations, and residual envelope into a spectral frame for each time step.

### 3.3 Hypercube population

The spectral frames from each analysed sample are placed into the hypercube at the position corresponding to that sample's metadata (pitch, dynamic, articulation, variation). Time forms one axis of the hypercube; the remaining axes are populated according to the sample mapping.

Where the input sampling is sparse (e.g., samples recorded only at every third semitone), intermediate positions can be left empty for later interpolation, or pre-interpolated during the analysis phase.

### 3.4 Alignment and normalization

Before compression, the hypercube contents must be aligned across samples:

- **Temporal alignment:** note onsets should be aligned so that corresponding phases (attack, sustain, release) occupy the same time indices across the pitch, dynamic, and articulation axes. This is necessary for compression to exploit cross-sample redundancy.
- **Amplitude normalization:** partial amplitudes may need normalization to a consistent reference level to separate loudness from spectral shape.
- **Pitch normalization:** partial frequencies can be represented as ratios relative to the fundamental, removing the pitch axis from the frequency data and leaving only inharmonicity information.

## 4. Compression

### 4.1 Overview

The populated spectral hypercube is compressed to reduce storage and transmission size. Compression exploits the redundancies described in Section 1.2: correlations across pitch, dynamic, articulation, time, and among the partials themselves.

Calyx supports multiple compression strategies that can be applied independently or in combination. The choice of strategy and its parameters determine the tradeoff between compression ratio and reconstruction fidelity.

### 4.2 Partial-domain dimensionality reduction

Within each spectral frame, the N partial amplitudes are correlated: when an instrument plays louder, many partials increase in amplitude together; the spectral envelope shifts but retains its overall shape. This correlation can be exploited by reducing the dimensionality of the partial amplitude vector.

**Principal Component Analysis (PCA):** compute the principal components of partial amplitude vectors across the full hypercube. Project each frame onto the top K components (where K << N). Reconstruction multiplies the K-dimensional coefficient vector by the basis matrix to recover approximate partial amplitudes.

**Non-negative Matrix Factorization (NMF):** since partial amplitudes are non-negative, NMF may provide a more physically meaningful decomposition than PCA. NMF finds a basis of non-negative spectral shapes and non-negative activation coefficients, which can be interpreted as "spectral building blocks" that combine additively.

Typical reduction: N = 64 partials reduced to K = 8 to 16 components, depending on quality target.

### 4.3 Tensor decomposition

The full hypercube, after partial-domain reduction, is a multi-dimensional tensor of coefficient values. Tensor decomposition methods compress this structure by factoring it into smaller component matrices or tensors.

**Tucker decomposition:** decomposes the tensor into a small core tensor multiplied by factor matrices along each axis. Each factor matrix captures the principal modes of variation along its axis (e.g., how spectral content varies across dynamic levels). The core tensor captures the interactions between axes.

**CP (Canonical Polyadic) decomposition:** represents the tensor as a sum of rank-one components, each of which is an outer product of vectors along each axis. More constrained than Tucker but can be more compact for tensors with low effective rank.

**Tensor train decomposition:** represents the tensor as a chain of three-dimensional cores. Scales well to high-dimensional tensors and allows efficient element access.

The rank or number of components in each decomposition controls the compression ratio and reconstruction quality. These parameters are configurable per quality tier (see Section 4.5).

### 4.4 Shared component extraction

Many instruments contain articulations or playing techniques that share significant common material. For example:

- The sustain phase of legato, detaché, and marcato bowing may be nearly identical at the same pitch and dynamic.
- Fortissimo onsets across different articulations share similar attack transient shapes.
- Round robin variations differ only slightly from one another.

Calyx can exploit this by extracting shared spectral components and storing them once, with per-articulation or per-variation residuals stored as lightweight deltas. This is analogous to dictionary learning or codebook-based compression.

The shared component extraction can operate as a preprocessing step before tensor decomposition, further reducing the effective size of the hypercube.

### 4.5 Quality tiers

Calyx defines configurable quality tiers that control the aggressiveness of compression:

| Tier       | Intended use             | Compression target  | Quality target                                       |
| ---------- | ------------------------ | ------------------- | ---------------------------------------------------- |
| Preview    | Auditioning, browsing    | Maximum compression | Recognisable character; artifacts acceptable         |
| Production | Music production, mixing | Balanced            | Perceptually transparent in typical musical contexts |
| Reference  | Solo exposure, mastering | Conservative        | Near-indistinguishable from original                 |

Each tier maps to specific parameter choices in the compression pipeline: number of retained PCA/NMF components, tensor decomposition rank, and stochastic model fidelity.

## 5. Container format

### 5.1 File structure

A Calyx file (.calyx) contains:

1. **Header:** format version, instrument metadata (name, type, pitch range, etc.), axis definitions (which dimensions are present and their ranges), quality tier, and compression parameters used.
2. **Basis data:** PCA/NMF basis matrices, tensor decomposition factor matrices and core tensors. This is the shared structural information needed for reconstruction.
3. **Coefficient data:** the compressed spectral content, stored as tensor decomposition coefficients or compressed coefficient streams.
4. **Stochastic model:** parameters for the residual/noise component, stored separately from the deterministic partial data.
5. **Metadata tables:** mapping information that relates hypercube axes to musical parameters (e.g., articulation index 3 = "pizzicato"), sample alignment data, and any instrument-specific configuration.

### 5.2 Extensibility

The container format should support:

- Additional axes beyond the core set (e.g., microphone position, player identity, room acoustics).
- Future compression methods that may supersede the initial PCA/tensor approach.
- Auxiliary data such as performance scripts, key mappings, or GUI layout hints for a playback engine.
- Versioned sections, so that older readers can skip sections they don't understand.

### 5.3 Relationship to existing formats

Calyx is not intended to replace SFZ, SF2, or other instrument definition formats at the metadata level. A future version of the specification may define mappings between Calyx container metadata and existing format conventions, allowing a Calyx file to carry equivalent information to an SFZ instrument definition alongside its compressed spectral data.

## 6. Reconstruction and playback

### 6.1 Phase handling

The spectral hypercube as described in Section 2.2 primarily stores amplitude information for deterministic partials. Reconstructing audio from amplitude-only data requires a phase assignment strategy.

Several approaches are possible:

**Harmonic phase locking:** assign phases based on the fundamental frequency, with each partial's phase derived as an integer multiple of the fundamental's phase. This produces clean, coherent tones but may sound overly synthetic for instruments with natural phase variation.

**Stored phase envelopes:** include phase information in the spectral frames, compressed alongside amplitudes. This preserves the original phase relationships but increases data size and compression complexity.

**Phase dispersion modeling:** store a per-instrument or per-articulation phase dispersion profile that describes the statistical distribution of inter-partial phase relationships. At synthesis time, phases are assigned according to this distribution, introducing natural variation without storing per-frame phase data.

**Analysis-resynthesis phase recovery:** use phase vocoder techniques to reconstruct plausible phase trajectories from amplitude envelopes and frequency tracks. This adds computational cost at playback but requires no additional stored data.

The optimal approach may vary by instrument type and quality tier. This is an open research question (see docs/open-questions.md).

### 6.2 Real-time synthesis

For direct playback from the compressed representation, the synthesis engine must:

1. Determine the current position in the hypercube based on incoming performance data (MIDI note, velocity, articulation keyswitches, expression controllers).
2. Reconstruct the spectral frame at that position by evaluating the tensor decomposition and basis expansion.
3. Synthesise audio from the spectral frame using an additive oscillator bank (deterministic component) and filtered noise generator (stochastic component).
4. Interpolate smoothly as performance parameters change, producing continuous timbral transitions.

The computational cost of step 2 is the primary constraint on real-time feasibility. For a Tucker decomposition with moderate rank, reconstructing a single spectral frame involves a small number of matrix multiplications, which is well within the budget of modern CPUs at audio frame rates (50 to 200 Hz for spectral frame updates, independent of the audio sample rate).

### 6.3 Decompression to PCM

For compatibility with existing sampler engines, Calyx files can be decompressed to conventional audio. The decompression process:

1. Selects discrete points in the hypercube corresponding to the desired sample set (e.g., every semitone, at specific velocity layers, for specific articulations).
2. Reconstructs spectral frames at each point.
3. Synthesises time-domain audio from the spectral frames using overlap-add resynthesis.
4. Writes the resulting audio files and generates a compatible instrument definition (SFZ or similar).

This produces a conventional sample library that can be loaded into any standard sampler. The decompressed library will not be bit-identical to the original source material (the compression is lossy), but should be perceptually equivalent at the Production or Reference quality tier.

## 7. Limitations and known tradeoffs

### 7.1 Transient fidelity

The sinusoidal partial model is weakest at representing sharp transients: hammer strikes, plucked string attacks, percussive onsets, and similar events. These contain broadband energy that is poorly captured by a small number of tracked partials. The stochastic component (Section 2.2) partially addresses this, but the overall fidelity of attacks and transients is likely to be the primary perceptual limitation of the format, particularly at higher compression ratios.

Specialized transient modeling, potentially using a separate short-time representation for onset regions, may be needed. This is an open research area.

### 7.2 Non-harmonic and percussive sources

Instruments with strongly inharmonic or noise-dominated spectra (cymbals, snare drums, shakers) may not compress well under a model designed primarily for harmonic content. The spectral hypercube may still offer modest compression benefits for such sources, but the achievable ratios will be lower and the perceptual quality more variable. The format should handle these gracefully, even if the compression is less dramatic than for harmonic instruments.

### 7.3 Lossy compression

Calyx compression is lossy. Information is discarded during dimensionality reduction and tensor decomposition. The discarded information cannot be recovered from the compressed representation. Vendors and users should treat original PCM content as the archival master and the Calyx representation as a derived distribution format.

### 7.4 Disk streaming considerations

It is technically possible to stream audio directly from a Calyx file on disk, reconstructing spectral frames on demand rather than loading the full compressed representation into RAM. This would reduce RAM consumption further but would increase disk I/O and introduce latency for random access (e.g., when a new note triggers and requires reading a new region of the hypercube). The tradeoffs between RAM residency and disk streaming depend on the specific use case and hardware context. The initial specification focuses on RAM-resident playback; disk streaming is a potential future optimisation.

## 8. Future directions

The following areas are out of scope for the initial specification but are tracked for future work:

- **Multi-instrument containers:** encoding an entire ensemble or orchestra as a single Calyx file, with cross-instrument compression exploiting shared spectral components (e.g., similar harmonic structures across string instruments).
- **Streaming and incremental loading:** protocols for progressive loading of Calyx data, allowing playback to begin before the full file is loaded.
- **Neural compression:** using learned codecs (autoencoders, neural audio codecs) as an alternative or complement to PCA/tensor methods.
- **Spatial audio:** encoding microphone position and spatial characteristics as additional hypercube axes.
- **Performance scripting:** embedding instrument behavior logic (legato transitions, repetition detection, expression mapping) within the container format.

## References

*This section will be populated as the literature review develops. Key references include:*

- Serra, X. and Smith, J.O. (1990). "Spectral Modeling Synthesis: A Sound Analysis/Synthesis System Based on a Deterministic plus Stochastic Decomposition." Computer Music Journal 14(4).
- Engel, J. et al. (2020). "DDSP: Differentiable Digital Signal Processing." ICLR 2020.
- Kolda, T.G. and Bader, B.W. (2009). "Tensor Decompositions and Applications." SIAM Review 51(3).

---

*This specification is a living document. Contributions, corrections, and feedback are welcome via GitHub Issues and Discussions.*
