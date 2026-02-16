# Calyx

**A compressed spectral representation format for sampled instruments.**

Calyx is an open specification and reference implementation for encoding multi-sampled instrument libraries as compact spectral hypercube structures. Instead of storing raw audio waveforms, Calyx decomposes sampled sounds into their harmonic components and compresses the resulting spectral data using dimensionality reduction and tensor decomposition techniques.

The goal is to reduce the storage, distribution, and runtime costs of expressive instrument emulation by one to three orders of magnitude compared with conventional PCM-based sample libraries.

## The problem

Modern sampled instrument libraries achieve expressive realism by recording large numbers of audio samples across multiple dimensions: pitch, dynamic level, articulation type, round robin variation, and more. A flagship orchestral library may contain 300 to 500 gigabytes of PCM audio data. This creates compounding costs at every stage of the production chain:

- **For vendors:** warm storage and CDN egress costs scale linearly with library size and customer base. Hosting and distributing tens of terabytes of sample content incurs high operational costs.
- **For consumers:** downloading a large library can take hours. Storing multiple libraries requires dedicated high-capacity SSDs. Loading a full orchestral template into RAM for a scoring session can consume 32-64 GB or more.
- **For musical expression:** conventional samplers crossfade between discrete sample layers to simulate continuous dynamics and articulation changes. This produces audible seams and limits the fluidity of performance.

These problems all stem from the same root: sampled instruments are stored and transmitted as raw audio, with no structural compression that exploits the inherent redundancies in how acoustic instruments produce sound.

## Proposed approach

Calyx represents a sampled instrument as a multi-dimensional tensor (referred to as a spectral hypercube) where each axis corresponds to a musically meaningful parameter:

- **Time:** the evolution of the sound's spectrum over its duration (attack, sustain, release)
- **Pitch:** the fundamental frequency and how the harmonic structure varies across the instrument's range
- **Dynamic level:** how the spectral content changes with playing intensity
- **Articulation:** different playing techniques (bowed, plucked, muted, etc.)
- **Variation:** round robin or stochastic variation within a single note event

Each point in this space encodes the amplitudes (and potentially phases and noise parameters) of the sound's harmonic partials. Because acoustic instruments have strong correlations across all of these dimensions (upper partials tend to behave similarly across nearby pitches; loud and soft versions of the same note share most of their spectral shape), the hypercube is highly compressible.

Calyx applies compression techniques from linear algebra and signal processing, including principal component analysis (PCA), non-negative matrix factorization (NMF), and tensor decomposition methods (Tucker, CP), to reduce the hypercube to a compact representation. This compressed form can be:

1. **Stored and transmitted** as a distribution format, significantly reducing file sizes, download times, and hosting costs.
2. **Decompressed back to PCM** for use with existing sampler engines (Kontakt, SFZ players, etc.).
3. **Played back directly** by a spectral synthesis engine that reconstructs audio from the compressed representation in real time, enabling continuous interpolation across all parameter axes.

## Current status

Calyx is in the **specification and early research** phase. The core concept is documented in the [specification](docs/specification.md) and supporting research documents. A proof-of-concept implementation in Rust is under development.

Contributions are welcome at every level: feedback on the specification, literature references, signal-processing expertise, Rust development, perceptual listening tests, and industry perspectives from sample library developers or users.

## Quick start

*The proof of concept is not yet available. This section will be updated when the PoC reaches a usable state.*

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to participate. In short: all forms of contribution are valued, not just code. If you have experience with spectral analysis, audio compression, tensor methods, Rust audio programming, or sample library production, there are meaningful ways to help shape this project.

## Licensing

Calyx uses a dual-license approach:

- **Source code** is licensed under [Apache 2.0](LICENSE). This includes the patent grant provision, which ensures that contributors and users are protected from patent claims arising from the codec algorithms.
- **Specification, documentation, and diagrams** are licensed under [CC-BY-4.0](LICENSE-DOCS), allowing free use and adaptation with attribution.

This structure is intended to keep the format and its implementations permanently open and available for anyone to use, build on, or integrate into commercial or non-commercial projects.

## Related work and acknowledgments

Calyx builds on ideas from several areas of research and open source audio development:

- **Spectral Modeling Synthesis (SMS):** Xavier Serra and Julius O. Smith's foundational work on sinusoidal plus residual audio models
- **DDSP:** Google Magenta's differentiable digital signal processing framework for neural audio synthesis
- **Tensor decomposition methods:** Tucker, CP, and related techniques for compressing multi-dimensional data
- **Surge Synth Team:** an exemplary model of open source audio project governance and community stewardship
- **Rust audio ecosystem:** Fundsp, CPAL, dasp, and other crates establishing patterns for audio work in Rust

## Contact

*Discussion channels will be listed here as the project community forms. For now, GitHub Issues and Discussions are the primary venues.*
