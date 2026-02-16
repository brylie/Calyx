# Calyx Use Cases

**Status:** Draft
**License:** This document is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) License](https://creativecommons.org/licenses/by/4.0/).

This document describes the primary use cases for Calyx, ordered by estimated value and impact. Each use case identifies who benefits, the magnitude of the benefit, what technical capabilities are required, and what tradeoffs or limitations apply.

The ordering reflects a preliminary assessment. As research and PoC work produce concrete data (compression ratios, perceptual quality measurements, computational benchmarks), the prioritization should be revisited.

---

## Priority framework

Two factors drive the ordering:

1. **Breadth of benefit.** How many people or organizations are affected, and how significantly?
2. **Technical feasibility in early stages.** Can this use case be demonstrated with a PoC, or does it require a mature, production-quality implementation?

Use cases that deliver broad economic benefit and can be validated early are ranked highest. Use cases that require a complete real-time synthesis engine or deep integration with existing tools are ranked lower, not because they are less valuable in the long term, but because they depend on foundational work that the higher-priority use cases will drive.

---

## Use case 1: Compressed distribution format

### Summary

Calyx serves as a lossy compression format for distributing sample library content, reducing file sizes by one to three orders of magnitude compared with PCM audio packaged in ZIP, RAR, or similar general-purpose archives.

### Who benefits

**Sample library vendors** are the primary beneficiaries. Companies that produce and sell sample libraries (Spitfire Audio, Orchestral Tools, Vienna Symphonic Library, Native Instruments, EastWest, Cinesamples, 8Dio, Impact Soundworks, and many smaller developers) bear the cost of hosting and delivering large files to customers. A vendor with a 20-product catalog averaging 100 GB per product maintains 2 TB of distribution content. Each customer download transfers that data through CDN infrastructure at a per-gigabyte cost.

**Consumers** benefit from shorter download times and smaller initial file sizes. A library that takes six hours to download as PCM might take minutes in Calyx format. This is especially significant for users with slow or metered internet connections, and for the growing number of composers working from laptops or portable setups with limited SSD capacity.

**Marketplace platforms** (e.g., Plugin Boutique, Best Service, the vendors' own storefronts) benefit from reduced bandwidth costs and faster delivery.

### Estimated magnitude

Assume a flagship orchestral library of 400 GB (PCM in WAV or lossless FLAC). General-purpose lossless compression (ZIP, FLAC) typically achieves 40-60% of original size, yielding roughly 160-240 GB. Lossy audio compression (OGG, MP3) can reach 5-10% of PCM size, but these are not structured for instrument content and the per-sample overhead is high.

Calyx, by operating in the spectral domain and exploiting cross-pitch, cross-dynamic, and cross-articulation redundancy, aims for 0.1% to 1% of original PCM size for harmonic instrument content. A 400 GB library might compress to 400 MB to 4 GB. Even at the conservative end, this is a 100:1 improvement over uncompressed PCM and a 40:1 to 60:1 improvement over FLAC.

At cloud hosting rates of roughly $0.08/GB for CDN egress (typical at scale), delivering that 400 GB library to 10,000 customers costs approximately $320,000 in bandwidth. At 4 GB, the same delivery costs $3,200. The savings compound with catalog size and customer base.

**These estimates are speculative and need validation through PoC experimentation.** The actual compression ratio will depend on instrument type, quality tier, and the effectiveness of the tensor decomposition on real spectral data. Establishing real numbers is a primary goal of the PoC.

### Technical requirements

- Analysis pipeline: spectral analysis of PCM samples, hypercube population from sample metadata
- Compression pipeline: PCA/NMF dimensionality reduction, tensor decomposition
- Container format: .calyx file structure with embedded metadata
- Decompression pipeline: reconstruct PCM audio from compressed representation, generate compatible instrument definitions (SFZ or similar)

This use case does **not** require a real-time playback engine. The consumer decompresses the Calyx file into conventional samples and loads them into their existing sampler (Kontakt, sfizz, Decent Sampler, etc.).

### Tradeoffs and limitations

- Compression is lossy. Decompressed audio will not be bit-identical to the original. Perceptual transparency at the Production quality tier is the target, but must be validated.
- Transient-heavy and non-harmonic content (percussion, effects) will compress less effectively than harmonic instruments.
- Vendors must trust that the lossy compression does not degrade their product's perceived quality. Listening tests and quality metrics are essential for adoption.
- Decompression adds a step to the customer's workflow. This friction is acceptable if the download time savings are substantial, but should be minimized (ideally a one-click process).

---

## Use case 2: Vendor-side storage optimization

### Summary

Vendors compress their master sample content into Calyx format for warm storage and distribution, archiving the original PCM to cold storage. This reduces ongoing storage costs for active catalog content.

### Who benefits

**Sample library vendors and publishers** with large catalogs. A major vendor maintaining 50+ products at 50-500 GB each may have 5-25 TB of active distribution content on warm cloud storage, plus backups and regional mirrors.

### Estimated magnitude

Cloud storage costs vary by tier and provider. Representative rates (AWS, as of 2024-2025):

| Storage tier                   | Cost per TB/month | Use case                             |
| ------------------------------ | ----------------- | ------------------------------------ |
| S3 Standard (warm)             | ~$23              | Active distribution, frequent access |
| S3 Infrequent Access           | ~$12.50           | Less frequently accessed content     |
| S3 Glacier Deep Archive (cold) | ~$1               | Archival, rare retrieval             |

A vendor with 10 TB of distribution content on S3 Standard pays roughly $230/month ($2,760/year). If Calyx compresses this to 100 GB, the warm storage cost drops to $2.30/month. The original PCM masters move to Glacier Deep Archive at $0.01/GB/month ($10/month for 10 TB).

Total storage cost goes from ~$230/month to ~$12/month. Not transformative for a single vendor in isolation, but meaningful across the industry and compounding over time as catalogs grow.

### Technical requirements

Same as Use Case 1 (analysis, compression, container format). Additionally requires:

- Reliable round-trip quality: vendors need confidence that archiving raw PCM and serving Calyx is not a one-way decision. They should be able to reconstruct PCM from Calyx at any time, understanding that reconstruction is lossy.
- Robust metadata preservation: all instrument mapping, scripting, and playback configuration must survive the compression/decompression cycle intact.

### Tradeoffs and limitations

- Vendors may be reluctant to archive original PCM if the lossy reconstruction is not perceptually transparent. A Reference quality tier with very conservative compression may be necessary for this use case.
- Integration with existing vendor infrastructure (asset management, build pipelines, delivery systems) requires tooling beyond the core codec.
- Cold storage retrieval has latency (hours for Glacier Deep Archive). Vendors need the original PCM only for re-mastering or format migration, not for daily operations, so this is acceptable.

---

## Use case 3: Reduced RAM consumption during playback

### Summary

A sampler or synthesis engine loads the compressed Calyx representation into RAM instead of full PCM sample data, reducing memory consumption by a factor corresponding to the compression ratio. Playback reconstructs audio from the compressed representation in real time.

### Who benefits

**Composers and producers** working with large orchestral templates. A full scoring template with dozens of instruments, each with multiple articulations loaded, can consume 32-64 GB of RAM or more. This is often the binding constraint on template complexity, forcing users to choose between instrument coverage and system stability.

**Users with limited hardware.** Laptop-based composers, students, hobbyists, and musicians in resource-constrained environments benefit disproportionately from reduced RAM requirements.

### Estimated magnitude

If a Calyx representation is 100:1 smaller than PCM, a template that currently requires 48 GB of RAM might fit in 480 MB. Even accounting for runtime overhead (oscillator state, buffers, decompression workspace), the RAM savings would be dramatic.

However, this estimate assumes the playback engine can work directly from the compressed representation without expanding it to PCM in memory. If intermediate buffers or look-ahead decompression are needed, the effective savings will be lower.

### Technical requirements

This use case requires a **real-time spectral synthesis engine** capable of:

- Evaluating tensor decomposition at a spectral frame position in real time (at the spectral frame rate, e.g., 100-200 Hz)
- Running an additive oscillator bank to produce audio from reconstructed spectral frames (at the audio sample rate, e.g., 44,100-96,000 Hz)
- Generating the stochastic/residual component from its compressed parameters
- Interpolating smoothly across hypercube axes as performance parameters change

This is substantially more complex than the codec-only requirements of Use Cases 1 and 2. It is the eventual goal of the project but depends on the foundational compression work.

### Tradeoffs and limitations

- Real-time decompression adds CPU load. The tradeoff is RAM for CPU. On modern hardware with many cores, this is generally favorable, but must be benchmarked.
- Latency: reconstructing a spectral frame and synthesizing audio introduces processing delay. For live performance, this latency must be below perceptual thresholds (ideally under 10 ms).
- Quality: real-time synthesis from compressed spectral data may not match the quality of pre-rendered PCM playback, particularly for transients and noise-heavy content.
- Adoption: this use case requires either integration into existing sampler engines (plugin SDK, format support) or a standalone Calyx player that composers adopt alongside their existing tools.

---

## Use case 4: Continuous articulation and expression morphing

### Summary

Because the spectral hypercube represents articulation and dynamic as continuous axes rather than discrete layers, a Calyx playback engine can interpolate smoothly between articulations and dynamic levels. This eliminates the crossfade artifacts inherent in conventional samplers and enables more fluid, expressive performance.

### Who benefits

**Composers and performers** seeking more realistic and expressive results from virtual instruments. The discontinuities between velocity layers and the abrupt switches between articulations are among the most commonly cited limitations of sampled instruments.

**Sample library developers** who could differentiate their products by offering Calyx-native instruments with continuous expression control, rather than competing solely on sample count and recording quality.

### Estimated magnitude

This benefit is qualitative rather than quantitative. It does not save money or storage; it improves the musical result. The magnitude depends on how audible the improvement is compared with state-of-the-art conventional samplers (which use sophisticated crossfading and scripting to minimize discontinuities).

For solo, exposed instrument lines (a solo violin melody, a solo cello passage), the improvement could be significant and immediately audible. In dense orchestral textures where individual instruments are partially masked, the benefit may be subtle.

### Technical requirements

All requirements of Use Case 3 (real-time spectral synthesis engine), plus:

- Smooth interpolation across articulation and dynamic axes without audible artifacts at the interpolation boundaries
- Mapping of MIDI controllers (expression, modwheel, keyswitches, aftertouch) to hypercube axis positions
- Low-latency response to parameter changes for real-time performance feel

### Tradeoffs and limitations

- Interpolating between articulations that are physically dissimilar (e.g., arco to pizzicato) may produce unrealistic intermediate states. The hypercube doesn't model the physical transition between techniques, only the endpoint states. Transition modeling may require additional logic beyond simple interpolation.
- This use case is the furthest from being demonstrable with a PoC. It requires the full synthesis engine plus thoughtful musical interaction design.
- Evaluation is subjective. Unlike compression ratios (measurable) or RAM usage (measurable), "more expressive" requires listening tests and musician feedback.

---

## Use case 5: Disk streaming from compressed format

### Summary

Instead of loading the full compressed representation into RAM, a playback engine streams Calyx data from disk on demand, decompressing and synthesizing in real time. This further reduces RAM consumption at the cost of increased disk I/O.

### Who benefits

**Users with very large templates** that exceed available RAM even with Calyx's compression. Also potentially useful for very low-memory systems (embedded devices, mobile).

### Estimated magnitude

The additional RAM saving beyond Use Case 3 is modest in most scenarios. If the compressed representation is already 100:1 smaller than PCM, it likely fits comfortably in RAM for most instruments. Disk streaming becomes relevant only for extremely large multi-instrument setups or severely constrained hardware.

The primary risk is that disk I/O introduces latency and depends on storage hardware performance. SSD random read latency is typically 0.1-0.2 ms, which is acceptable, but HDD latency (5-10 ms) would be problematic for responsive playback.

### Technical requirements

All requirements of Use Case 3, plus:

- Efficient random access into the .calyx container format (seeking to arbitrary hypercube positions without reading the entire file)
- Prefetching and caching strategies to minimize latency for likely note events
- Graceful degradation when disk I/O is slower than expected (e.g., falling back to a lower quality tier or using a cached preview representation)

### Tradeoffs and limitations

- Increased disk I/O wear, particularly on SSDs with write-cycle limits (though read-only access is less damaging than write-heavy workloads)
- Latency sensitivity: disk access adds unpredictable delay compared with RAM access
- Complexity: the streaming engine must manage prefetching, caching, and fallback logic, significantly increasing implementation complexity
- Marginal benefit: if RAM-resident playback (Use Case 3) already reduces memory requirements by two orders of magnitude, the incremental value of disk streaming is limited for most users

This use case is lowest priority because its benefit is marginal in most scenarios and its implementation complexity is high. It should be revisited if real-world usage reveals that RAM-resident Calyx representations are still too large for common workflows.

---

## Summary matrix

| Use case               | Primary beneficiary   | Benefit type                     | Estimated impact | Requires real-time engine | Priority       |
| ---------------------- | --------------------- | -------------------------------- | ---------------- | ------------------------- | -------------- |
| 1. Distribution format | Vendors, consumers    | Economic (egress, download time) | High             | No                        | **Highest**    |
| 2. Vendor storage      | Vendors               | Economic (storage costs)         | Medium           | No                        | **High**       |
| 3. Reduced RAM         | Composers, producers  | Technical (workflow)             | High             | Yes                       | **Medium**     |
| 4. Expression morphing | Composers, performers | Musical (quality)                | Variable         | Yes                       | **Medium-low** |
| 5. Disk streaming      | Edge cases            | Technical (RAM)                  | Low-marginal     | Yes                       | **Low**        |

Use Cases 1 and 2 can be validated and delivered with the compression codec alone. Use Cases 3 and 4 require a real-time synthesis engine and represent the project's longer-term vision. Use Case 5 is a potential optimization that should be deferred until the core system is proven.

---

*This document will be updated as PoC results provide concrete data to refine the estimates and prioritization. Contributions of additional use cases, industry perspective, and corrections to the estimates are welcome.*
