# Contributing to Calyx

Thank you for your interest in Calyx. This project is in its early stages (specification drafting and proof-of-concept development), which means contributions at every level are valuable, not just code.

## Ways to contribute

### Feedback on the specification

The [specification](docs/specification.md) is the core intellectual document of the project. If you have expertise in spectral analysis, audio compression, tensor methods, or sampler technology, your review and feedback on the specification would be especially helpful.

You can provide feedback by:

- Opening a GitHub Issue with the "specification" label
- Starting a GitHub Discussion for broader design questions
- Submitting a Pull Request with proposed changes to the specification text

### Literature references

The [literature review](docs/literature-review.md) has many [TODO] markers where additional references are needed. If you know of relevant papers, tools, or prior art that should be included, please open an Issue or PR with the reference and a brief note on its relevance.

### Answering open questions

The [open questions](docs/open-questions.md) document tracks unresolved research and design questions. Each question lists the kind of expertise that would help answer it. If you have relevant knowledge or experience, even partial answers, preliminary data, or pointers to existing work, are useful.

### Perceptual evaluation

As the proof of concept produces compressed audio, we will need listeners to evaluate quality. Formal listening tests will be organized as the project matures, but informal feedback (particularly from audio engineers, musicians, and sample library users) is welcome at any stage.

### Code contributions

The proof-of-concept implementation is written in Rust. Code contributions are welcome for:

- The analysis pipeline (spectral analysis, partial tracking, hypercube population)
- The compression pipeline (PCA, NMF, tensor decomposition)
- The decompression and resynthesis pipeline
- Input format support (SF2, SFZ parsing)
- Testing and benchmarking infrastructure
- Documentation improvements and examples

### Industry perspective

If you develop, publish, or regularly use sample libraries, your perspective on the use cases, priorities, and practical requirements described in [use-cases.md](docs/use-cases.md) is valuable. Understanding real-world workflows and pain points helps ensure Calyx solves problems people actually have.

## Getting started with code contributions

### Prerequisites

- Rust (stable toolchain, latest stable release recommended)
- Git

### Building the project

```bash
git clone https://github.com/[org]/calyx.git
cd calyx/poc
cargo build
cargo test
```

### Code style

- Follow standard Rust conventions (`cargo fmt`, `cargo clippy`)
- Write tests for new functionality
- Document public APIs with doc comments
- Keep unsafe code to a minimum; justify and document any unsafe blocks

### Pull request process

1. Fork the repository and create a branch from `main`.
2. Make your changes. Include tests if applicable.
3. Run `cargo fmt` and `cargo clippy` before committing.
4. Write a clear commit message describing what the change does and why.
5. Open a Pull Request against `main`. Describe what the PR addresses and link to any related Issues.
6. Be open to feedback. PRs may go through a round or two of review before merging.

For larger changes (new compression methods, significant specification revisions, architectural decisions), please open an Issue or Discussion first to talk through the approach before investing time in implementation. This avoids wasted effort if the direction needs adjustment.

## Documentation contributions

Documentation is as important as code in this project. The specification, literature review, and supporting documents are intended to be collaborative resources.

- Specification and documentation files are in the `docs/` directory, written in Markdown.
- Documentation is licensed under CC-BY-4.0, and source code under Apache 2.0. By contributing, you agree to license your contributions under these terms.

## Communication

- **GitHub Issues:** for specific bugs, questions, or tasks
- **GitHub Discussions:** for open-ended design conversations, ideas, and general questions

*Additional communication channels (Discord, forum) may be established as the community grows.*

## Decision-making

Calyx is currently maintained by a single project lead. Decisions about specification changes, design direction, and priorities are made transparently through GitHub Issues and Discussions. The goal is to be responsive to community input while maintaining coherence in the project's direction.

As the project grows and attracts regular contributors, the governance model will be revisited. See [open-questions.md, Section 7.2](docs/open-questions.md#72-what-governance-model-should-the-project-adopt-as-it-grows) for more on this topic.

## Code of Conduct

This project follows a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold a welcoming, respectful, and constructive environment for everyone.

## Licensing

- **Source code:** Apache 2.0 ([LICENSE-CODE](LICENSE))
- **Specification and documentation:** CC-BY-4.0 ([LICENSE-DOCS](LICENSE-content))

By submitting a contribution (code, documentation, or other material), you agree to license it under the applicable license above.

If you have questions about licensing or intellectual property, please open an Issue.
