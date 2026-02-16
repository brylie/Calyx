# Calyx Roadmap

**Status:** Draft
**Last updated:** February 2026

This document outlines the planned phases of research and development for Calyx. Each phase has a clear objective, concrete deliverables, and criteria for moving to the next phase. The roadmap is structured so that each phase builds on the previous one, and early phases are scoped to answer the questions that determine whether later phases are worth pursuing.

Timelines are intentionally omitted. This is an open source research project with variable contributor availability. Progress is measured by milestones, not calendar dates.

---

## Phase 0: Foundation (current)

### Objective

Establish the intellectual and organizational groundwork for the project.

### Deliverables

- [x] Core specification draft (docs/specification.md)
- [x] Literature review with identified sources and gaps (docs/literature-review.md)
- [x] Prioritised use cases (docs/use-cases.md)
- [x] Open questions inventory (docs/open-questions.md)
- [ ] Glossary of domain terminology (docs/glossary.md)
- [ ] Repository published on GitHub with licensing, contribution guidelines, and code of conduct
- [ ] Initial community outreach: posts to relevant forums (KVR developer forum, Rust Audio community, Surge Synth Team Discord) introducing the project and inviting feedback on the specification

### Exit criteria

The specification is complete enough for an informed reader to evaluate the concept's feasibility. The repository is public and welcoming to contributors. At least a few external people have read and commented on the specification.

---

## Phase 1: Analysis proof of concept

### Objective

Demonstrate that a multi-sampled instrument can be decomposed into a spectral hypercube and that the hypercube structure captures the essential timbral content.

### Deliverables

- [ ] Rust-based analysis pipeline that ingests a source instrument (SF2 or SFZ with WAV samples) and produces a populated spectral hypercube
- [ ] Partial tracking implementation (or integration of an existing library via FFI) that extracts harmonic partial amplitudes over time from each sample
- [ ] Stochastic residual extraction: subtract the sinusoidal reconstruction from the original and characterise the residual's spectral envelope
- [ ] Hypercube assembly: populate the multi-dimensional tensor from analysed samples using source metadata (pitch, velocity, articulation mappings)
- [ ] Basic resynthesis from the uncompressed hypercube: reconstruct audio from spectral frames using additive synthesis plus filtered noise, without any compression applied
- [ ] Quality evaluation: compare resynthesised audio against originals using spectral convergence metrics and informal listening tests
- [ ] Documentation of results, including identified issues with the analysis or resynthesis process

### Source material

Start with one or two instruments from openly licensed soundfonts:

- **Salamander Grand Piano** (CC-BY-3.0): multi-velocity piano with rich harmonic content and inharmonicity; good test of partial tracking across dynamics
- **FluidR3 GM SoundFont** (public domain): multiple instrument types available for breadth testing

### Key questions answered

- How many partials are needed for perceptually acceptable resynthesis? (Open question 1.1)
- What frame rate is sufficient? (Open question 1.3)
- How well does the sinusoidal plus residual decomposition capture different instrument types? (Open question 1.5)
- Does the uncompressed hypercube structure make sense, i.e., is the data organized in a way that subsequent compression can exploit?

### Exit criteria

Resynthesised audio from the uncompressed hypercube is perceptually close to the original samples for at least one instrument type. The analysis pipeline runs reliably on the test material. Issues with the spectral frame model (phase problems, transient loss, residual quality) are documented and understood.

---

## Phase 2: Compression proof of concept

### Objective

Demonstrate that the spectral hypercube can be compressed to a small fraction of its uncompressed size while maintaining perceptually acceptable quality. Establish real compression ratio numbers.

### Deliverables

- [ ] PCA and/or NMF implementation for partial amplitude dimensionality reduction
- [ ] At least one tensor decomposition method (Tucker or CP) applied to the reduced hypercube
- [ ] Configurable quality tiers: implement at least two tiers (e.g., Preview and Production) with different rank/component settings
- [ ] Reconstruction pipeline: decompress from the compressed representation back to spectral frames, then to audio
- [ ] Compression benchmarks: measured compression ratios (compressed size vs. original PCM size) for each test instrument at each quality tier
- [ ] Quality benchmarks: spectral convergence, SNR, and informal listening tests comparing compressed resynthesis against originals
- [ ] Comparison baseline: measure the same source material compressed with FLAC and OGG Vorbis for reference
- [ ] Written analysis of results, including where compression works well, where it struggles, and what the practical limits appear to be

### Key questions answered

- What compression ratios are achievable? (Open question 2.5)
- PCA vs. NMF: which produces better results for this application? (Open question 2.1)
- How should decomposition rank be selected for each quality tier? (Open question 2.3)
- Are the compression ratios compelling enough to justify continued development toward a distribution format?

### Exit criteria

Compression ratios and perceptual quality are documented with real numbers. The results are either (a) compelling enough to proceed with container format design, or (b) clearly identify what needs to change in the approach to achieve viable compression. Either outcome is a success; this phase is designed to produce actionable data.

---

## Phase 3: Container format and codec tooling

### Objective

Define the .calyx container format and build command-line tools for compression and decompression, enabling the distribution format use case (Use Case 1).

### Deliverables

- [ ] Container format specification: binary format design with header, basis data, coefficient data, stochastic model, and metadata sections (specification Section 5)
- [ ] `calyx-encode` CLI tool: takes an SF2/SFZ instrument as input, runs analysis and compression, outputs a .calyx file
- [ ] `calyx-decode` CLI tool: takes a .calyx file as input, reconstructs PCM audio, outputs an SFZ instrument definition with WAV files
- [ ] `calyx-info` CLI tool: displays metadata and statistics for a .calyx file (instrument name, axes, compression ratio, quality tier)
- [ ] Round-trip testing: encode a source instrument, decode it, and systematically evaluate quality loss
- [ ] Format documentation: detailed specification of the binary container format, sufficient for independent implementation
- [ ] Initial test suite of .calyx files from various source instruments, published alongside the tools

### Key questions answered

- What metadata must the container preserve? (Open question 4.1)
- How should the format handle versioning and extensibility? (Open question 4.2)
- Is the round-trip quality (PCM to Calyx to PCM) acceptable for practical use?

### Exit criteria

A third party could, using only the published specification and tools, compress a sample library into .calyx format and decompress it back into a usable SFZ instrument. The round-trip quality at the Production tier is perceptually transparent for at least two instrument types.

---

## Phase 4: Community validation and industry feedback

### Objective

Get the format and tools into the hands of sample library developers and users. Collect feedback on quality, usability, and whether the compression ratios deliver meaningful value.

### Deliverables

- [ ] Published release of encode/decode tools with documentation and example files
- [ ] Outreach to indie sample library developers willing to test Calyx with their content
- [ ] Outreach to open source sampler projects (sfizz, Decent Sampler, Shortcircuit XT) for potential integration interest
- [ ] Structured feedback collection: quality reports, workflow friction points, feature requests, format suggestions
- [ ] Listening tests with a broader pool of evaluators, ideally including professional audio engineers and composers
- [ ] Updated specification and tools incorporating feedback

### Key questions answered

- Does the compression quality meet professional standards?
- Is the encode/decode workflow practical for vendors?
- What format or tooling changes do real users need?
- Is there genuine interest in adoption?

### Exit criteria

At least a few external developers have successfully used the tools with their own content and provided substantive feedback. The feedback either validates the approach or identifies specific, addressable problems.

---

## Phase 5: Real-time playback engine

### Objective

Build a synthesis engine that plays back directly from the compressed Calyx representation, enabling the reduced RAM (Use Case 3) and continuous expression (Use Case 4) benefits.

### Deliverables

- [ ] Real-time spectral frame reconstruction from compressed hypercube
- [ ] Additive oscillator bank with efficient partial synthesis
- [ ] Stochastic component generator (filtered noise from compressed residual parameters)
- [ ] MIDI input handling: note on/off, velocity, pitch bend, expression controllers mapped to hypercube axes
- [ ] Polyphony management: multiple simultaneous voices with independent hypercube positions
- [ ] Continuous interpolation across dynamic and articulation axes driven by MIDI controllers
- [ ] Latency and CPU benchmarks on representative hardware
- [ ] Standalone application or audio plugin (VST3/CLAP) for integration with DAWs

### Key questions answered

- What is the computational cost of real-time reconstruction? (Open question 5.1)
- How should the oscillator bank be implemented for efficiency? (Open question 5.2)
- How should note transitions be handled? (Open question 5.3)
- Is continuous articulation morphing perceptually convincing?

### Exit criteria

A musician can load a .calyx instrument into the playback engine, play it from a MIDI controller, and produce musically useful results with continuous expression control. CPU usage is reasonable for typical polyphony on current hardware.

---

## Phase 6: Maturation and ecosystem

### Objective

Grow the project from a working prototype into a robust tool that integrates with the broader music production ecosystem.

### Deliverables (tentative; shaped by outcomes of earlier phases)

- [ ] Stability and performance hardening of all tools and engines
- [ ] Expanded instrument type support, including percussion and non-harmonic sources
- [ ] Integration with one or more existing sampler engines or DAWs
- [ ] Multi-instrument containers (e.g., an entire orchestral section in one .calyx file)
- [ ] Streaming and progressive loading for large instruments
- [ ] Governance formalization if the contributor community has grown
- [ ] Potential standardization efforts if industry adoption warrants

This phase is deliberately open-ended. Its shape will be determined by what is learned in Phases 1 through 5 and by the needs of the community that forms around the project.

---

## Cross-cutting concerns

The following activities run in parallel with the phased work above:

**Literature review maintenance.** The literature review should be updated continuously as new references are discovered and as the research landscape evolves.

**Patent monitoring.** Basic patent landscape awareness should be maintained from Phase 1 onward. A more thorough survey should be conducted before Phase 3 (container format) to ensure the format design does not inadvertently infringe existing patents.

**Quality evaluation methodology.** Develop and refine perceptual quality evaluation methods (objective metrics, listening test protocols) starting in Phase 1 and applying them consistently through all subsequent phases.

**Documentation.** Keep the specification, open questions, and supporting documents current as research produces new findings. Resolved open questions should be documented with their answers.

---

*This roadmap will be updated as phases are completed and as new information changes the plan. Contributions that help advance any phase are welcome; see [CONTRIBUTING.md](../CONTRIBUTING.md) for how to get involved.*
