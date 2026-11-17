//! Pure Rust `.calyx` binary container format encoding and decoding.
//!
//! Implements the extensible chunk-based `.calyx` binary container format strictly
//! from first principles with zero external dependencies and safe Rust (`#![deny(unsafe_code)]`).
//!
//! # Container Format Specification
//!
//! A `.calyx` file consists of:
//! 1. An 18-byte fixed header:
//!    - Magic identification: `b"CALX"` (4 bytes)
//!    - Major version: `u16` Little-Endian (`0`)
//!    - Minor version: `u16` Little-Endian (`1`)
//!    - Quality tier: `u8` (0 = Preview, 1 = Production, 2 = Reference)
//!    - Flags: `u8` (bit 0: Tucker present, bit 1: CP present, bit 2: Residual present)
//!    - Sample rate: `f32` Little-Endian (e.g. 48000.0)
//!    - Hop size: `u16` Little-Endian (e.g. 256)
//!    - Number of partials: `u16` Little-Endian (e.g. 16 or 24)
//! 2. A sequence of extensible FourCC chunks:
//!    - Chunk ID: 4 ASCII bytes (e.g. `b"META"`, `b"AXES"`, `b"TUCK"`, `b"CPMD"`, `b"NOIS"`)
//!    - Payload length: `u32` Little-Endian byte count
//!    - Payload: raw serialized data
//!    - Checksum: `u32` Little-Endian Adler-32 integrity checksum of payload
//!
//! Any reader encountering an unknown chunk safely skips past it using the payload length
//! and verifies checksum integrity, ensuring full forward and backward extensibility.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::error::DspError;
use crate::hypercube::coordinates::HypercubeLayout;
use crate::hypercube::cp::CpModel;
use crate::hypercube::tucker::TuckerDecomposition;
use crate::math::{Matrix, Tensor};
use crate::spectral::traits::NoiseBandEnvelope;

/// FourCC magic byte sequence identifying a Calyx container.
pub const CALYX_MAGIC: [u8; 4] = *b"CALX";

/// Current major format version.
pub const CALYX_MAJOR_VERSION: u16 = 0;

/// Current minor format version.
pub const CALYX_MINOR_VERSION: u16 = 1;

/// Fixed header size in bytes.
pub const HEADER_SIZE: usize = 18;

/// FourCC identifier for the Metadata chunk (`b"META"`).
pub const CHUNK_META: [u8; 4] = *b"META";

/// FourCC identifier for the Axes / HypercubeLayout chunk (`b"AXES"`).
pub const CHUNK_AXES: [u8; 4] = *b"AXES";

/// FourCC identifier for the Tucker decomposition chunk (`b"TUCK"`).
pub const CHUNK_TUCK: [u8; 4] = *b"TUCK";

/// FourCC identifier for the CP-ALS model chunk (`b"CPMD"`).
pub const CHUNK_CPMD: [u8; 4] = *b"CPMD";

/// FourCC identifier for the Stochastic Residual noise chunk (`b"NOIS"`).
pub const CHUNK_NOIS: [u8; 4] = *b"NOIS";

/// Bitflag indicating the presence of Tucker decomposition data.
pub const FLAG_HAS_TUCKER: u8 = 1 << 0;

/// Bitflag indicating the presence of CP-ALS decomposition data.
pub const FLAG_HAS_CP: u8 = 1 << 1;

/// Bitflag indicating the presence of stochastic residual noise parameters.
pub const FLAG_HAS_RESIDUAL: u8 = 1 << 2;

/// Maximum allowable single chunk payload size (512 MB) to prevent out-of-memory panics.
const MAX_CHUNK_PAYLOAD_BYTES: u32 = 512 * 1024 * 1024;

/// Quality tiers defined in Calyx specification §4.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum QualityTier {
    /// Preview tier (auditioning, low RAM, fast loading, high compression).
    Preview = 0,
    /// Production tier (arranging, scoring, standard fidelity).
    Production = 1,
    /// Reference tier (solo instrument, archival, maximum fidelity).
    Reference = 2,
}

impl QualityTier {
    /// Parses a raw `u8` integer into a validated `QualityTier`.
    pub fn from_u8(val: u8) -> Result<Self, DspError> {
        match val {
            0 => Ok(QualityTier::Preview),
            1 => Ok(QualityTier::Production),
            2 => Ok(QualityTier::Reference),
            _ => Err(DspError::invalid_parameter(
                "quality_tier",
                format!("Unsupported quality tier code: {}", val),
            )),
        }
    }

    /// Returns the raw `u8` code for this tier.
    #[inline]
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }
}

impl From<crate::metrics::report::QualityTier> for QualityTier {
    fn from(tier: crate::metrics::report::QualityTier) -> Self {
        match tier {
            crate::metrics::report::QualityTier::Preview => QualityTier::Preview,
            crate::metrics::report::QualityTier::Production => QualityTier::Production,
            crate::metrics::report::QualityTier::Reference
            | crate::metrics::report::QualityTier::Strict => QualityTier::Reference,
        }
    }
}

impl fmt::Display for QualityTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QualityTier::Preview => write!(f, "Preview"),
            QualityTier::Production => write!(f, "Production"),
            QualityTier::Reference => write!(f, "Reference"),
        }
    }
}

/// 18-byte fixed binary container header.
#[derive(Debug, Clone, PartialEq)]
pub struct CalyxContainerHeader {
    /// Magic byte identifier (`CALX`).
    pub magic: [u8; 4],
    /// Major container version.
    pub major_version: u16,
    /// Minor container version.
    pub minor_version: u16,
    /// Quality tier code (0 = Preview, 1 = Production, 2 = Reference).
    pub quality_tier: u8,
    /// Bitflags describing optional content.
    pub flags: u8,
    /// Audio sample rate in Hz (e.g. 48000.0).
    pub sample_rate: f32,
    /// STFT analysis hop size in samples (e.g. 256).
    pub hop_size: u16,
    /// Target number of sinusoidal partials per frame.
    pub num_partials: u16,
}

impl CalyxContainerHeader {
    /// Creates a new container header with standard magic and version.
    pub fn new(
        quality_tier: QualityTier,
        sample_rate: f32,
        hop_size: u16,
        num_partials: u16,
    ) -> Self {
        Self {
            magic: CALYX_MAGIC,
            major_version: CALYX_MAJOR_VERSION,
            minor_version: CALYX_MINOR_VERSION,
            quality_tier: quality_tier.as_u8(),
            flags: 0,
            sample_rate,
            hop_size,
            num_partials,
        }
    }

    /// Serializes this 18-byte header into a fixed-size byte array.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.major_version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.minor_version.to_le_bytes());
        buf[8] = self.quality_tier;
        buf[9] = self.flags;
        buf[10..14].copy_from_slice(&self.sample_rate.to_le_bytes());
        buf[14..16].copy_from_slice(&self.hop_size.to_le_bytes());
        buf[16..18].copy_from_slice(&self.num_partials.to_le_bytes());
        buf
    }

    /// Parses an 18-byte slice into a validated `CalyxContainerHeader`.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] if the slice is shorter than 18 bytes,
    /// magic bytes do not match `CALX`, or major version is unsupported.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, DspError> {
        if bytes.len() < HEADER_SIZE {
            return Err(DspError::invalid_parameter(
                "container_header",
                format!(
                    "Header byte slice too short: {} bytes (expected {})",
                    bytes.len(),
                    HEADER_SIZE
                ),
            ));
        }

        let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
        if magic != CALYX_MAGIC {
            return Err(DspError::invalid_parameter(
                "magic",
                format!(
                    "Invalid container magic bytes: expected {:?}, got {:?}",
                    CALYX_MAGIC, magic
                ),
            ));
        }

        let major_version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if major_version > CALYX_MAJOR_VERSION {
            return Err(DspError::invalid_parameter(
                "major_version",
                format!(
                    "Unsupported major version {}. Reader supports up to {}.",
                    major_version, CALYX_MAJOR_VERSION
                ),
            ));
        }

        let minor_version = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        let quality_tier = bytes[8];
        QualityTier::from_u8(quality_tier)?;

        let flags = bytes[9];
        let sample_rate = f32::from_le_bytes(bytes[10..14].try_into().unwrap());
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(DspError::invalid_parameter(
                "sample_rate",
                format!("Invalid sample rate in header: {}", sample_rate),
            ));
        }

        let hop_size = u16::from_le_bytes(bytes[14..16].try_into().unwrap());
        if hop_size == 0 {
            return Err(DspError::invalid_parameter(
                "hop_size",
                "Hop size must be non-zero",
            ));
        }

        let num_partials = u16::from_le_bytes(bytes[16..18].try_into().unwrap());
        if num_partials == 0 {
            return Err(DspError::invalid_parameter(
                "num_partials",
                "Num partials must be non-zero",
            ));
        }

        Ok(Self {
            magic,
            major_version,
            minor_version,
            quality_tier,
            flags,
            sample_rate,
            hop_size,
            num_partials,
        })
    }

    /// Returns the parsed quality tier.
    #[inline]
    pub fn tier(&self) -> Result<QualityTier, DspError> {
        QualityTier::from_u8(self.quality_tier)
    }
}

/// Definition of an axis in the spectral hypercube coordinate system.
#[derive(Debug, Clone, PartialEq)]
pub struct AxisDefinition {
    /// Semantic name of the axis (e.g. "time", "pitch", "dynamic", "articulation", "variation").
    pub name: String,
    /// Number of discrete points / dimension size along this axis.
    pub dim: usize,
    /// Minimum physical or normalized coordinate bound (e.g. 0.0 or MIDI note 21.0).
    pub min_bound: f64,
    /// Maximum physical or normalized coordinate bound (e.g. 1.0 or MIDI note 108.0).
    pub max_bound: f64,
}

impl AxisDefinition {
    /// Constructs a new axis definition with dimension and bounds.
    pub fn new(
        name: impl Into<String>,
        dim: usize,
        min_bound: f64,
        max_bound: f64,
    ) -> Result<Self, DspError> {
        if dim == 0 {
            return Err(DspError::invalid_parameter(
                "dim",
                "Axis dimension must be greater than zero",
            ));
        }
        Ok(Self {
            name: name.into(),
            dim,
            min_bound,
            max_bound,
        })
    }
}

/// Serialized matrix representation preserving exact 64-bit IEEE float data.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixData {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row-major continuous floating-point coefficients.
    pub data: Vec<f64>,
}

impl MatrixData {
    /// Constructs a new `MatrixData` with dimension and element validation.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, DspError> {
        let total = rows.checked_mul(cols).ok_or_else(|| {
            DspError::invalid_parameter("dimensions", "Matrix dimensions overflow")
        })?;
        if total != data.len() {
            return Err(DspError::invalid_parameter(
                "data",
                format!(
                    "Matrix dimension mismatch: {} x {} = {} elements, but provided data has {}",
                    rows,
                    cols,
                    total,
                    data.len()
                ),
            ));
        }
        Ok(Self { rows, cols, data })
    }

    /// Converts this serialized matrix to a domain [`Matrix<f64>`].
    pub fn to_matrix(&self) -> Result<Matrix<f64>, DspError> {
        Matrix::from_vec(self.rows, self.cols, self.data.clone())
    }

    /// Constructs a `MatrixData` from a domain [`Matrix<f64>`].
    pub fn from_matrix(matrix: &Matrix<f64>) -> Self {
        Self {
            rows: matrix.rows(),
            cols: matrix.cols(),
            data: matrix.as_slice().to_vec(),
        }
    }
}

/// Serialized Tucker-HOSVD decomposition container.
#[derive(Debug, Clone, PartialEq)]
pub struct TuckerContainerData {
    /// Shape vector of the dense core tensor.
    pub core_shape: Vec<usize>,
    /// Continuous row-major / C-order elements of the core tensor.
    pub core_data: Vec<f64>,
    /// Orthogonal mode factor matrices $U^{(0)}, \dots, U^{(N-1)}$.
    pub factors: Vec<MatrixData>,
}

impl TuckerContainerData {
    /// Constructs a `TuckerContainerData` from a [`TuckerDecomposition`].
    pub fn from_decomposition(tucker: &TuckerDecomposition) -> Self {
        let core_shape = tucker.core.shape().to_vec();
        let core_data = tucker.core.data().to_vec();
        let factors = tucker.factors.iter().map(MatrixData::from_matrix).collect();
        Self {
            core_shape,
            core_data,
            factors,
        }
    }

    /// Converts this container data back into a domain [`TuckerDecomposition`].
    pub fn to_decomposition(&self) -> Result<TuckerDecomposition, DspError> {
        let core = Tensor::from_vec(self.core_shape.clone(), self.core_data.clone())?;
        let mut factors = Vec::with_capacity(self.factors.len());
        for f in &self.factors {
            factors.push(f.to_matrix()?);
        }
        Ok(TuckerDecomposition { core, factors })
    }
}

/// Serialized Canonical Polyadic (CP-ALS) decomposition container.
#[derive(Debug, Clone, PartialEq)]
pub struct CpContainerData {
    /// CP rank $R$ (number of rank-one components).
    pub rank: usize,
    /// Component scaling weights $\lambda_1, \dots, \lambda_R$.
    pub weights: Vec<f64>,
    /// Mode factor matrices $A^{(0)}, \dots, A^{(N-1)}$ of shape $I_n \times R$.
    pub factors: Vec<MatrixData>,
}

impl CpContainerData {
    /// Constructs a `CpContainerData` from a domain [`CpModel`].
    pub fn from_model(model: &CpModel) -> Self {
        let rank = if let Some(first) = model.factors.first() {
            first.cols()
        } else {
            model.weights.len()
        };
        let factors = model.factors.iter().map(MatrixData::from_matrix).collect();
        Self {
            rank,
            weights: model.weights.clone(),
            factors,
        }
    }

    /// Converts this container data back into a domain [`CpModel`].
    pub fn to_model(&self) -> Result<CpModel, DspError> {
        let mut factors = Vec::with_capacity(self.factors.len());
        for f in &self.factors {
            factors.push(f.to_matrix()?);
        }
        Ok(CpModel {
            factors,
            weights: self.weights.clone(),
        })
    }
}

/// Serialized stochastic residual filterbank parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualContainerData {
    /// Number of critical perceptual frequency bands.
    pub num_bands: usize,
    /// Center frequencies in Hz for each critical filterbank band.
    pub center_frequencies: Vec<f64>,
    /// Sub-band noise energies.
    pub band_energies: Vec<f64>,
    /// Stochastic noise energy fraction relative to total signal energy in $[0.0, 1.0]$.
    pub noise_ratio: f64,
}

impl ResidualContainerData {
    /// Constructs a `ResidualContainerData` from a domain [`NoiseBandEnvelope`].
    pub fn from_envelope(envelope: &NoiseBandEnvelope, center_frequencies: Vec<f64>) -> Self {
        Self {
            num_bands: envelope.band_energies.len(),
            center_frequencies,
            band_energies: envelope.band_energies.clone(),
            noise_ratio: envelope.noise_ratio,
        }
    }

    /// Converts this container data back into a domain [`NoiseBandEnvelope`].
    pub fn to_envelope(&self) -> NoiseBandEnvelope {
        NoiseBandEnvelope {
            band_energies: self.band_energies.clone(),
            noise_ratio: self.noise_ratio,
        }
    }
}

/// High-level container struct representing an entire `.calyx` binary file.
#[derive(Debug, Clone, PartialEq)]
pub struct CalyxContainer {
    /// 18-byte fixed container header.
    pub header: CalyxContainerHeader,
    /// Arbitrary key-value string metadata (instrument name, author, copyright, notes, etc.).
    pub metadata: HashMap<String, String>,
    /// 5-axis acoustic hypercube layout specification.
    pub layout: HypercubeLayout,
    /// Semantic definitions and physical bounds for each hypercube axis.
    pub axes: Vec<AxisDefinition>,
    /// Optional Tucker decomposition data (offline authoring representation).
    pub tucker_data: Option<TuckerContainerData>,
    /// Optional CP-ALS decomposition data (runtime real-time voice representation).
    pub cp_data: Option<CpContainerData>,
    /// Optional stochastic residual noise parameters.
    pub residual_data: Option<ResidualContainerData>,
}

impl CalyxContainer {
    /// Creates a new `CalyxContainer` with header and layout.
    pub fn new(header: CalyxContainerHeader, layout: HypercubeLayout) -> Self {
        let mut axes = Vec::with_capacity(layout.shape().len());
        let default_names = ["time", "pitch", "dynamic", "articulation", "variation"];
        for (i, &dim) in layout.shape().iter().enumerate() {
            let name = if i < default_names.len() {
                default_names[i].to_string()
            } else {
                format!("axis_{}", i)
            };
            axes.push(AxisDefinition {
                name,
                dim,
                min_bound: 0.0,
                max_bound: 1.0,
            });
        }

        Self {
            header,
            metadata: HashMap::new(),
            layout,
            axes,
            tucker_data: None,
            cp_data: None,
            residual_data: None,
        }
    }

    /// Builder method attaching metadata key-value pairs.
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Builder method inserting a single metadata key-value pair.
    pub fn with_metadata_entry(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Builder method replacing the axis definitions.
    pub fn with_axes(mut self, axes: Vec<AxisDefinition>) -> Result<Self, DspError> {
        if axes.len() != self.layout.shape().len() {
            return Err(DspError::invalid_parameter(
                "axes",
                format!(
                    "Axis definition count {} does not match layout dimension count {}",
                    axes.len(),
                    self.layout.shape().len()
                ),
            ));
        }
        for (i, (axis, &dim)) in axes.iter().zip(self.layout.shape().iter()).enumerate() {
            if axis.dim != dim {
                return Err(DspError::invalid_parameter(
                    "axes",
                    format!(
                        "Axis {} '{}' dimension {} does not match layout dimension {}",
                        i, axis.name, axis.dim, dim
                    ),
                ));
            }
        }
        self.axes = axes;
        Ok(self)
    }

    /// Builder method attaching Tucker decomposition data.
    pub fn with_tucker(mut self, tucker: TuckerContainerData) -> Self {
        self.tucker_data = Some(tucker);
        self.header.flags |= FLAG_HAS_TUCKER;
        self
    }

    /// Builder method attaching CP decomposition data.
    pub fn with_cp(mut self, cp: CpContainerData) -> Self {
        self.cp_data = Some(cp);
        self.header.flags |= FLAG_HAS_CP;
        self
    }

    /// Builder method attaching stochastic residual data.
    pub fn with_residual(mut self, residual: ResidualContainerData) -> Self {
        self.residual_data = Some(residual);
        self.header.flags |= FLAG_HAS_RESIDUAL;
        self
    }

    /// Serializes this container and all contained chunks to any byte writer.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<(), DspError> {
        // Sync flags with present chunks
        let mut header = self.header.clone();
        header.flags = 0;
        if self.tucker_data.is_some() {
            header.flags |= FLAG_HAS_TUCKER;
        }
        if self.cp_data.is_some() {
            header.flags |= FLAG_HAS_CP;
        }
        if self.residual_data.is_some() {
            header.flags |= FLAG_HAS_RESIDUAL;
        }

        // 1. Write Header (18 bytes)
        writer.write_all(&header.to_bytes()).map_err(|e| {
            DspError::invalid_parameter("io", format!("Failed writing container header: {}", e))
        })?;

        // 2. Write META chunk if metadata is non-empty
        if !self.metadata.is_empty() {
            let mut payload = Vec::new();
            // Sort keys for deterministic output
            let mut sorted_entries: Vec<(&String, &String)> = self.metadata.iter().collect();
            sorted_entries.sort_by_key(|(k, _)| *k);

            let num_entries = sorted_entries.len() as u32;
            payload.extend_from_slice(&num_entries.to_le_bytes());

            for (k, v) in sorted_entries {
                let k_bytes = k.as_bytes();
                payload.extend_from_slice(&(k_bytes.len() as u32).to_le_bytes());
                payload.extend_from_slice(k_bytes);

                let v_bytes = v.as_bytes();
                payload.extend_from_slice(&(v_bytes.len() as u32).to_le_bytes());
                payload.extend_from_slice(v_bytes);
            }
            write_chunk(writer, CHUNK_META, &payload)?;
        }

        // 3. Write AXES chunk (mandatory for hypercube layout)
        {
            let mut payload = Vec::new();
            let num_axes = self.axes.len() as u32;
            payload.extend_from_slice(&num_axes.to_le_bytes());

            for axis in &self.axes {
                let name_bytes = axis.name.as_bytes();
                payload.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
                payload.extend_from_slice(name_bytes);
                payload.extend_from_slice(&(axis.dim as u32).to_le_bytes());
                payload.extend_from_slice(&axis.min_bound.to_le_bytes());
                payload.extend_from_slice(&axis.max_bound.to_le_bytes());
            }
            write_chunk(writer, CHUNK_AXES, &payload)?;
        }

        // 4. Write TUCK chunk if Tucker data is present
        if let Some(tucker) = &self.tucker_data {
            let mut payload = Vec::new();

            // Core shape
            let ndim = tucker.core_shape.len() as u32;
            payload.extend_from_slice(&ndim.to_le_bytes());
            for &dim in &tucker.core_shape {
                payload.extend_from_slice(&(dim as u32).to_le_bytes());
            }

            // Core data
            let core_len = tucker.core_data.len() as u32;
            payload.extend_from_slice(&core_len.to_le_bytes());
            for &val in &tucker.core_data {
                payload.extend_from_slice(&val.to_le_bytes());
            }

            // Mode factor matrices
            let num_factors = tucker.factors.len() as u32;
            payload.extend_from_slice(&num_factors.to_le_bytes());
            for f in &tucker.factors {
                payload.extend_from_slice(&(f.rows as u32).to_le_bytes());
                payload.extend_from_slice(&(f.cols as u32).to_le_bytes());
                for &val in &f.data {
                    payload.extend_from_slice(&val.to_le_bytes());
                }
            }
            write_chunk(writer, CHUNK_TUCK, &payload)?;
        }

        // 5. Write CPMD chunk if CP data is present
        if let Some(cp) = &self.cp_data {
            let mut payload = Vec::new();

            payload.extend_from_slice(&(cp.rank as u32).to_le_bytes());

            let num_weights = cp.weights.len() as u32;
            payload.extend_from_slice(&num_weights.to_le_bytes());
            for &w in &cp.weights {
                payload.extend_from_slice(&w.to_le_bytes());
            }

            let num_factors = cp.factors.len() as u32;
            payload.extend_from_slice(&num_factors.to_le_bytes());
            for f in &cp.factors {
                payload.extend_from_slice(&(f.rows as u32).to_le_bytes());
                payload.extend_from_slice(&(f.cols as u32).to_le_bytes());
                for &val in &f.data {
                    payload.extend_from_slice(&val.to_le_bytes());
                }
            }
            write_chunk(writer, CHUNK_CPMD, &payload)?;
        }

        // 6. Write NOIS chunk if residual data is present
        if let Some(residual) = &self.residual_data {
            let mut payload = Vec::new();

            payload.extend_from_slice(&(residual.num_bands as u32).to_le_bytes());

            let num_freqs = residual.center_frequencies.len() as u32;
            payload.extend_from_slice(&num_freqs.to_le_bytes());
            for &freq in &residual.center_frequencies {
                payload.extend_from_slice(&freq.to_le_bytes());
            }

            let num_energies = residual.band_energies.len() as u32;
            payload.extend_from_slice(&num_energies.to_le_bytes());
            for &e in &residual.band_energies {
                payload.extend_from_slice(&e.to_le_bytes());
            }

            payload.extend_from_slice(&residual.noise_ratio.to_le_bytes());

            write_chunk(writer, CHUNK_NOIS, &payload)?;
        }

        writer.flush().map_err(|e| {
            DspError::invalid_parameter("io", format!("Failed flushing container writer: {}", e))
        })?;

        Ok(())
    }

    /// Reads and deserializes a `CalyxContainer` from any byte stream.
    ///
    /// Extensible FourCC chunks: any unrecognized chunks are verified for checksum
    /// integrity and safely skipped, allowing older engines to open newer container files.
    pub fn read<R: Read>(reader: &mut R) -> Result<Self, DspError> {
        // 1. Read and parse 18-byte Header
        let mut header_buf = [0u8; HEADER_SIZE];
        reader.read_exact(&mut header_buf).map_err(|e| {
            DspError::invalid_parameter(
                "container_header",
                format!("Failed to read 18-byte container header: {}", e),
            )
        })?;
        let header = CalyxContainerHeader::from_slice(&header_buf)?;

        let mut metadata = HashMap::new();
        let mut axes: Option<Vec<AxisDefinition>> = None;
        let mut tucker_data = None;
        let mut cp_data = None;
        let mut residual_data = None;

        // 2. Read Chunks until EOF
        loop {
            let mut chunk_id = [0u8; 4];
            match reader.read_exact(&mut chunk_id) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(e) => {
                    return Err(DspError::invalid_parameter(
                        "chunk_header",
                        format!("Failed reading chunk FourCC identifier: {}", e),
                    ));
                }
            }

            let mut len_buf = [0u8; 4];
            reader.read_exact(&mut len_buf).map_err(|e| {
                DspError::invalid_parameter(
                    "chunk_header",
                    format!(
                        "Failed reading chunk payload length for {:?}: {}",
                        chunk_id, e
                    ),
                )
            })?;
            let payload_len = u32::from_le_bytes(len_buf);
            if payload_len > MAX_CHUNK_PAYLOAD_BYTES {
                return Err(DspError::invalid_parameter(
                    "chunk_header",
                    format!(
                        "Chunk payload length {} exceeds maximum allowed limit {}",
                        payload_len, MAX_CHUNK_PAYLOAD_BYTES
                    ),
                ));
            }

            let mut payload = vec![0u8; payload_len as usize];
            reader.read_exact(&mut payload).map_err(|e| {
                DspError::invalid_parameter(
                    "chunk_payload",
                    format!(
                        "Failed reading {} payload bytes for chunk {:?}: {}",
                        payload_len, chunk_id, e
                    ),
                )
            })?;

            let mut checksum_buf = [0u8; 4];
            reader.read_exact(&mut checksum_buf).map_err(|e| {
                DspError::invalid_parameter(
                    "chunk_checksum",
                    format!("Failed reading checksum for chunk {:?}: {}", chunk_id, e),
                )
            })?;
            let expected_checksum = u32::from_le_bytes(checksum_buf);
            let calculated_checksum = adler32(&payload);
            if calculated_checksum != expected_checksum {
                return Err(DspError::invalid_parameter(
                    "checksum",
                    format!(
                        "Checksum mismatch in chunk {:?}: expected {}, calculated {}",
                        chunk_id, expected_checksum, calculated_checksum
                    ),
                ));
            }

            // Dispatch chunk by FourCC
            match chunk_id {
                CHUNK_META => {
                    metadata = parse_meta_chunk(&payload)?;
                }
                CHUNK_AXES => {
                    axes = Some(parse_axes_chunk(&payload)?);
                }
                CHUNK_TUCK => {
                    tucker_data = Some(parse_tuck_chunk(&payload)?);
                }
                CHUNK_CPMD => {
                    cp_data = Some(parse_cpmd_chunk(&payload)?);
                }
                CHUNK_NOIS => {
                    residual_data = Some(parse_nois_chunk(&payload)?);
                }
                _ => {
                    // Unrecognized chunk: safely ignored for forward compatibility
                }
            }
        }

        // Validate that mandatory AXES chunk was parsed
        let axes = axes.ok_or_else(|| {
            DspError::invalid_parameter("axes", "Missing mandatory AXES chunk in Calyx container")
        })?;

        // Enforce promised header flags
        if (header.flags & FLAG_HAS_TUCKER != 0) && tucker_data.is_none() {
            return Err(DspError::invalid_parameter(
                "container",
                "Header declares Tucker decomposition (FLAG_HAS_TUCKER), but TUCK chunk is missing",
            ));
        }
        if (header.flags & FLAG_HAS_CP != 0) && cp_data.is_none() {
            return Err(DspError::invalid_parameter(
                "container",
                "Header declares CP decomposition (FLAG_HAS_CP), but CPMD chunk is missing",
            ));
        }
        if (header.flags & FLAG_HAS_RESIDUAL != 0) && residual_data.is_none() {
            return Err(DspError::invalid_parameter(
                "container",
                "Header declares residual noise (FLAG_HAS_RESIDUAL), but NOIS chunk is missing",
            ));
        }

        let shape: Vec<usize> = axes.iter().map(|a| a.dim).collect();
        let layout = HypercubeLayout::new(shape)?;

        Ok(Self {
            header,
            metadata,
            layout,
            axes,
            tucker_data,
            cp_data,
            residual_data,
        })
    }

    /// Serializes this container to a file on the filesystem.
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), DspError> {
        let p = path.as_ref();
        let file = File::create(p).map_err(|e| {
            DspError::invalid_parameter(
                "path",
                format!("Failed to create container file '{}': {}", p.display(), e),
            )
        })?;
        let mut writer = BufWriter::new(file);
        self.write(&mut writer)
    }

    /// Loads and deserializes a `CalyxContainer` from a file on the filesystem.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, DspError> {
        let p = path.as_ref();
        let file = File::open(p).map_err(|e| {
            DspError::invalid_parameter(
                "path",
                format!("Failed to open container file '{}': {}", p.display(), e),
            )
        })?;
        let mut reader = BufReader::new(file);
        Self::read(&mut reader)
    }
}

// ============================================================================
// Internal Chunk Encoding & Decoding Helpers
// ============================================================================

/// Computes the Adler-32 checksum of a byte slice in pure safe Rust.
#[inline]
pub fn adler32(data: &[u8]) -> u32 {
    let mut s1: u32 = 1;
    let mut s2: u32 = 0;
    const MOD_ADLER: u32 = 65521;

    for &byte in data {
        s1 = (s1 + byte as u32) % MOD_ADLER;
        s2 = (s2 + s1) % MOD_ADLER;
    }

    (s2 << 16) | s1
}

/// Helper writing a FourCC chunk with payload length, payload, and Adler-32 checksum.
fn write_chunk<W: Write>(
    writer: &mut W,
    chunk_id: [u8; 4],
    payload: &[u8],
) -> Result<(), DspError> {
    writer.write_all(&chunk_id).map_err(|e| {
        DspError::invalid_parameter(
            "io",
            format!("Failed writing chunk ID {:?}: {}", chunk_id, e),
        )
    })?;

    let len = payload.len() as u32;
    writer.write_all(&len.to_le_bytes()).map_err(|e| {
        DspError::invalid_parameter(
            "io",
            format!("Failed writing chunk length for {:?}: {}", chunk_id, e),
        )
    })?;

    writer.write_all(payload).map_err(|e| {
        DspError::invalid_parameter(
            "io",
            format!("Failed writing payload for chunk {:?}: {}", chunk_id, e),
        )
    })?;

    let checksum = adler32(payload);
    writer.write_all(&checksum.to_le_bytes()).map_err(|e| {
        DspError::invalid_parameter(
            "io",
            format!("Failed writing checksum for chunk {:?}: {}", chunk_id, e),
        )
    })?;

    Ok(())
}

/// Parses the payload of a `b"META"` chunk into key-value pairs.
fn parse_meta_chunk(payload: &[u8]) -> Result<HashMap<String, String>, DspError> {
    let mut offset = 0;
    if payload.len() < 4 {
        return Err(DspError::invalid_parameter(
            "meta_chunk",
            "Truncated META chunk payload",
        ));
    }

    let num_entries = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    // Capacity allocation safeguard: clamp pre-allocation against remaining payload bytes
    let remaining = payload.len() - offset;
    let cap = num_entries.min(remaining / 8);
    let mut map = HashMap::with_capacity(cap);

    for _ in 0..num_entries {
        if offset
            .checked_add(4)
            .map_or(true, |end| end > payload.len())
        {
            return Err(DspError::invalid_parameter(
                "meta_chunk",
                "Truncated key length in META chunk",
            ));
        }
        let k_len = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if offset
            .checked_add(k_len)
            .map_or(true, |end| end > payload.len())
        {
            return Err(DspError::invalid_parameter(
                "meta_chunk",
                "Truncated key bytes in META chunk",
            ));
        }
        let key = std::str::from_utf8(&payload[offset..offset + k_len]).map_err(|e| {
            DspError::invalid_parameter(
                "meta_chunk",
                format!("Invalid UTF-8 in metadata key: {}", e),
            )
        })?;
        offset += k_len;

        if offset
            .checked_add(4)
            .map_or(true, |end| end > payload.len())
        {
            return Err(DspError::invalid_parameter(
                "meta_chunk",
                "Truncated value length in META chunk",
            ));
        }
        let v_len = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if offset
            .checked_add(v_len)
            .map_or(true, |end| end > payload.len())
        {
            return Err(DspError::invalid_parameter(
                "meta_chunk",
                "Truncated value bytes in META chunk",
            ));
        }
        let val = std::str::from_utf8(&payload[offset..offset + v_len]).map_err(|e| {
            DspError::invalid_parameter(
                "meta_chunk",
                format!("Invalid UTF-8 in metadata value: {}", e),
            )
        })?;
        offset += v_len;

        map.insert(key.to_string(), val.to_string());
    }

    Ok(map)
}

/// Parses the payload of an `b"AXES"` chunk into a list of `AxisDefinition`s.
fn parse_axes_chunk(payload: &[u8]) -> Result<Vec<AxisDefinition>, DspError> {
    let mut offset = 0;
    if payload.len() < 4 {
        return Err(DspError::invalid_parameter(
            "axes_chunk",
            "Truncated AXES chunk payload",
        ));
    }

    let num_axes = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    if num_axes == 0 {
        return Err(DspError::invalid_parameter(
            "axes_chunk",
            "AXES chunk specifies zero axes",
        ));
    }

    // Capacity allocation safeguard: each axis takes at least 24 bytes (4 name_len + 4 dim + 8 min + 8 max)
    let remaining = payload.len() - offset;
    let cap = num_axes.min(remaining / 24);
    let mut axes = Vec::with_capacity(cap);

    for _ in 0..num_axes {
        if offset
            .checked_add(4)
            .map_or(true, |end| end > payload.len())
        {
            return Err(DspError::invalid_parameter(
                "axes_chunk",
                "Truncated axis name length in AXES chunk",
            ));
        }
        let name_len = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if offset
            .checked_add(name_len)
            .map_or(true, |end| end > payload.len())
        {
            return Err(DspError::invalid_parameter(
                "axes_chunk",
                "Truncated axis name bytes in AXES chunk",
            ));
        }
        let name = std::str::from_utf8(&payload[offset..offset + name_len]).map_err(|e| {
            DspError::invalid_parameter("axes_chunk", format!("Invalid UTF-8 in axis name: {}", e))
        })?;
        offset += name_len;

        if offset
            .checked_add(20)
            .map_or(true, |end| end > payload.len())
        {
            // 4 (dim) + 8 (min) + 8 (max) = 20
            return Err(DspError::invalid_parameter(
                "axes_chunk",
                "Truncated axis dimensions or bounds in AXES chunk",
            ));
        }
        let dim = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        let min_bound = f64::from_le_bytes(payload[offset..offset + 8].try_into().unwrap());
        offset += 8;

        let max_bound = f64::from_le_bytes(payload[offset..offset + 8].try_into().unwrap());
        offset += 8;

        axes.push(AxisDefinition::new(name, dim, min_bound, max_bound)?);
    }

    Ok(axes)
}

/// Parses the payload of a `b"TUCK"` chunk into a `TuckerContainerData`.
fn parse_tuck_chunk(payload: &[u8]) -> Result<TuckerContainerData, DspError> {
    let mut offset = 0;
    if payload.len() < 4 {
        return Err(DspError::invalid_parameter(
            "tuck_chunk",
            "Truncated TUCK chunk payload",
        ));
    }

    let ndim = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    let ndim_bytes = ndim.checked_mul(4).ok_or_else(|| {
        DspError::invalid_parameter("tuck_chunk", "Core shape dimensions byte size overflow")
    })?;
    if offset
        .checked_add(ndim_bytes)
        .map_or(true, |end| end > payload.len())
    {
        return Err(DspError::invalid_parameter(
            "tuck_chunk",
            "Truncated core shape in TUCK chunk",
        ));
    }
    // Capacity allocation safeguard: clamp pre-allocation against remaining payload bytes
    let remaining = payload.len() - offset;
    let cap = ndim.min(remaining / 4);
    let mut core_shape = Vec::with_capacity(cap);
    for _ in 0..ndim {
        let dim = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        core_shape.push(dim);
    }

    if offset
        .checked_add(4)
        .map_or(true, |end| end > payload.len())
    {
        return Err(DspError::invalid_parameter(
            "tuck_chunk",
            "Truncated core data length in TUCK chunk",
        ));
    }
    let core_len = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    let core_bytes = core_len
        .checked_mul(8)
        .ok_or_else(|| DspError::invalid_parameter("tuck_chunk", "Core data byte size overflow"))?;
    if offset
        .checked_add(core_bytes)
        .map_or(true, |end| end > payload.len())
    {
        return Err(DspError::invalid_parameter(
            "tuck_chunk",
            "Truncated core data elements in TUCK chunk",
        ));
    }
    let remaining = payload.len() - offset;
    let cap = core_len.min(remaining / 8);
    let mut core_data = Vec::with_capacity(cap);
    for _ in 0..core_len {
        let val = f64::from_le_bytes(payload[offset..offset + 8].try_into().unwrap());
        offset += 8;
        core_data.push(val);
    }

    if offset
        .checked_add(4)
        .map_or(true, |end| end > payload.len())
    {
        return Err(DspError::invalid_parameter(
            "tuck_chunk",
            "Truncated factor count in TUCK chunk",
        ));
    }
    let num_factors = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    let remaining = payload.len() - offset;
    let cap = num_factors.min(remaining / 8);
    let mut factors = Vec::with_capacity(cap);
    for _ in 0..num_factors {
        if offset
            .checked_add(8)
            .map_or(true, |end| end > payload.len())
        {
            return Err(DspError::invalid_parameter(
                "tuck_chunk",
                "Truncated matrix dimensions in TUCK factor",
            ));
        }
        let rows = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let cols = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        let total = rows.checked_mul(cols).ok_or_else(|| {
            DspError::invalid_parameter("tuck_chunk", "Matrix dimensions overflow in TUCK factor")
        })?;
        let bytes_needed = total.checked_mul(8).ok_or_else(|| {
            DspError::invalid_parameter("tuck_chunk", "Matrix byte size overflow in TUCK factor")
        })?;
        if offset
            .checked_add(bytes_needed)
            .map_or(true, |end| end > payload.len())
        {
            return Err(DspError::invalid_parameter(
                "tuck_chunk",
                "Truncated matrix data in TUCK factor",
            ));
        }
        let remaining = payload.len() - offset;
        let cap = total.min(remaining / 8);
        let mut data = Vec::with_capacity(cap);
        for _ in 0..total {
            let val = f64::from_le_bytes(payload[offset..offset + 8].try_into().unwrap());
            offset += 8;
            data.push(val);
        }
        factors.push(MatrixData::new(rows, cols, data)?);
    }

    Ok(TuckerContainerData {
        core_shape,
        core_data,
        factors,
    })
}

/// Parses the payload of a `b"CPMD"` chunk into a `CpContainerData`.
fn parse_cpmd_chunk(payload: &[u8]) -> Result<CpContainerData, DspError> {
    let mut offset = 0;
    if payload.len() < 8 {
        return Err(DspError::invalid_parameter(
            "cpmd_chunk",
            "Truncated CPMD chunk payload",
        ));
    }

    let rank = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    let num_weights = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    let weights_bytes = num_weights.checked_mul(8).ok_or_else(|| {
        DspError::invalid_parameter("cpmd_chunk", "Weights byte size overflow in CPMD chunk")
    })?;
    if offset
        .checked_add(weights_bytes)
        .map_or(true, |end| end > payload.len())
    {
        return Err(DspError::invalid_parameter(
            "cpmd_chunk",
            "Truncated weights in CPMD chunk",
        ));
    }
    // Capacity allocation safeguard: clamp pre-allocation against remaining payload bytes
    let remaining = payload.len() - offset;
    let cap = num_weights.min(remaining / 8);
    let mut weights = Vec::with_capacity(cap);
    for _ in 0..num_weights {
        let w = f64::from_le_bytes(payload[offset..offset + 8].try_into().unwrap());
        offset += 8;
        weights.push(w);
    }

    if offset
        .checked_add(4)
        .map_or(true, |end| end > payload.len())
    {
        return Err(DspError::invalid_parameter(
            "cpmd_chunk",
            "Truncated factor count in CPMD chunk",
        ));
    }
    let num_factors = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    let remaining = payload.len() - offset;
    let cap = num_factors.min(remaining / 8);
    let mut factors = Vec::with_capacity(cap);
    for _ in 0..num_factors {
        if offset
            .checked_add(8)
            .map_or(true, |end| end > payload.len())
        {
            return Err(DspError::invalid_parameter(
                "cpmd_chunk",
                "Truncated matrix dimensions in CPMD factor",
            ));
        }
        let rows = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let cols = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        let total = rows.checked_mul(cols).ok_or_else(|| {
            DspError::invalid_parameter("cpmd_chunk", "Matrix dimensions overflow in CPMD factor")
        })?;
        let bytes_needed = total.checked_mul(8).ok_or_else(|| {
            DspError::invalid_parameter("cpmd_chunk", "Matrix byte size overflow in CPMD factor")
        })?;
        if offset
            .checked_add(bytes_needed)
            .map_or(true, |end| end > payload.len())
        {
            return Err(DspError::invalid_parameter(
                "cpmd_chunk",
                "Truncated matrix data in CPMD factor",
            ));
        }
        let remaining = payload.len() - offset;
        let cap = total.min(remaining / 8);
        let mut data = Vec::with_capacity(cap);
        for _ in 0..total {
            let val = f64::from_le_bytes(payload[offset..offset + 8].try_into().unwrap());
            offset += 8;
            data.push(val);
        }
        factors.push(MatrixData::new(rows, cols, data)?);
    }

    Ok(CpContainerData {
        rank,
        weights,
        factors,
    })
}

/// Parses the payload of a `b"NOIS"` chunk into a `ResidualContainerData`.
fn parse_nois_chunk(payload: &[u8]) -> Result<ResidualContainerData, DspError> {
    let mut offset = 0;
    if payload.len() < 8 {
        return Err(DspError::invalid_parameter(
            "nois_chunk",
            "Truncated NOIS chunk payload",
        ));
    }

    let num_bands = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    let num_freqs = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    let freqs_bytes = num_freqs.checked_mul(8).ok_or_else(|| {
        DspError::invalid_parameter(
            "nois_chunk",
            "Center frequencies byte size overflow in NOIS chunk",
        )
    })?;
    if offset
        .checked_add(freqs_bytes)
        .map_or(true, |end| end > payload.len())
    {
        return Err(DspError::invalid_parameter(
            "nois_chunk",
            "Truncated center frequencies in NOIS chunk",
        ));
    }
    // Capacity allocation safeguard: clamp pre-allocation against remaining payload bytes
    let remaining = payload.len() - offset;
    let cap = num_freqs.min(remaining / 8);
    let mut center_frequencies = Vec::with_capacity(cap);
    for _ in 0..num_freqs {
        let f = f64::from_le_bytes(payload[offset..offset + 8].try_into().unwrap());
        offset += 8;
        center_frequencies.push(f);
    }

    if offset
        .checked_add(4)
        .map_or(true, |end| end > payload.len())
    {
        return Err(DspError::invalid_parameter(
            "nois_chunk",
            "Truncated band energy count in NOIS chunk",
        ));
    }
    let num_energies = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    let energies_bytes = num_energies.checked_mul(8).ok_or_else(|| {
        DspError::invalid_parameter(
            "nois_chunk",
            "Band energies byte size overflow in NOIS chunk",
        )
    })?;
    if offset
        .checked_add(energies_bytes)
        .map_or(true, |end| end > payload.len())
    {
        return Err(DspError::invalid_parameter(
            "nois_chunk",
            "Truncated band energies in NOIS chunk",
        ));
    }
    let remaining = payload.len() - offset;
    let cap = num_energies.min(remaining / 8);
    let mut band_energies = Vec::with_capacity(cap);
    for _ in 0..num_energies {
        let e = f64::from_le_bytes(payload[offset..offset + 8].try_into().unwrap());
        offset += 8;
        band_energies.push(e);
    }

    if offset
        .checked_add(8)
        .map_or(true, |end| end > payload.len())
    {
        return Err(DspError::invalid_parameter(
            "nois_chunk",
            "Truncated noise ratio in NOIS chunk",
        ));
    }
    let noise_ratio = f64::from_le_bytes(payload[offset..offset + 8].try_into().unwrap());

    Ok(ResidualContainerData {
        num_bands,
        center_frequencies,
        band_energies,
        noise_ratio,
    })
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypercube::cp::CpAls;
    use crate::hypercube::tucker::TuckerDecomposition;
    use crate::math::Tensor;
    use std::io::Cursor;

    #[test]
    fn test_header_serialization_roundtrip() {
        let header = CalyxContainerHeader::new(QualityTier::Production, 48000.0, 256, 16);
        let bytes = header.to_bytes();
        assert_eq!(&bytes[0..4], b"CALX");
        assert_eq!(bytes.len(), HEADER_SIZE);

        let parsed = CalyxContainerHeader::from_slice(&bytes).unwrap();
        assert_eq!(parsed.magic, CALYX_MAGIC);
        assert_eq!(parsed.major_version, CALYX_MAJOR_VERSION);
        assert_eq!(parsed.minor_version, CALYX_MINOR_VERSION);
        assert_eq!(parsed.tier().unwrap(), QualityTier::Production);
        assert_eq!(parsed.sample_rate, 48000.0);
        assert_eq!(parsed.hop_size, 256);
        assert_eq!(parsed.num_partials, 16);
    }

    fn create_minimal_valid_container(flags: u8) -> Vec<u8> {
        let mut header = CalyxContainerHeader::new(QualityTier::Production, 48000.0, 256, 16);
        header.flags = flags;
        let mut buf = Vec::new();
        buf.extend_from_slice(&header.to_bytes());

        // AXES chunk
        let mut axes = Vec::new();
        axes.extend_from_slice(&1u32.to_le_bytes()); // num_axes = 1
        axes.extend_from_slice(&4u32.to_le_bytes()); // name_len = 4
        axes.extend_from_slice(b"time");
        axes.extend_from_slice(&2u32.to_le_bytes()); // dim = 2
        axes.extend_from_slice(&0.0f64.to_le_bytes()); // min = 0.0
        axes.extend_from_slice(&1.0f64.to_le_bytes()); // max = 1.0
        write_chunk(&mut buf, CHUNK_AXES, &axes).unwrap();

        buf
    }

    #[test]
    fn test_header_validation_errors() {
        // Corrupt magic
        let mut bad_magic =
            CalyxContainerHeader::new(QualityTier::Preview, 44100.0, 128, 8).to_bytes();
        bad_magic[0] = b'B';
        assert!(matches!(
            CalyxContainerHeader::from_slice(&bad_magic),
            Err(DspError::InvalidParameter { name: "magic", .. })
        ));

        // Truncated header
        let truncated = [0u8; 10];
        assert!(matches!(
            CalyxContainerHeader::from_slice(&truncated),
            Err(DspError::InvalidParameter {
                name: "container_header",
                ..
            })
        ));

        // Unsupported major version
        let mut bad_ver =
            CalyxContainerHeader::new(QualityTier::Preview, 44100.0, 128, 8).to_bytes();
        bad_ver[4] = 99; // major version 99
        assert!(matches!(
            CalyxContainerHeader::from_slice(&bad_ver),
            Err(DspError::InvalidParameter {
                name: "major_version",
                ..
            })
        ));

        // Unsupported quality tier
        let mut bad_tier =
            CalyxContainerHeader::new(QualityTier::Preview, 44100.0, 128, 8).to_bytes();
        bad_tier[8] = 99;
        assert!(matches!(
            CalyxContainerHeader::from_slice(&bad_tier),
            Err(DspError::InvalidParameter {
                name: "quality_tier",
                ..
            })
        ));
        bad_tier[8] = 255;
        assert!(matches!(
            CalyxContainerHeader::from_slice(&bad_tier),
            Err(DspError::InvalidParameter {
                name: "quality_tier",
                ..
            })
        ));

        // Invalid sample rates: 0.0, negative, infinity, NaN
        let mut bad_sr =
            CalyxContainerHeader::new(QualityTier::Preview, 44100.0, 128, 8).to_bytes();
        bad_sr[10..14].copy_from_slice(&0.0f32.to_le_bytes());
        assert!(matches!(
            CalyxContainerHeader::from_slice(&bad_sr),
            Err(DspError::InvalidParameter {
                name: "sample_rate",
                ..
            })
        ));

        bad_sr[10..14].copy_from_slice(&(-44100.0f32).to_le_bytes());
        assert!(matches!(
            CalyxContainerHeader::from_slice(&bad_sr),
            Err(DspError::InvalidParameter {
                name: "sample_rate",
                ..
            })
        ));

        bad_sr[10..14].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(matches!(
            CalyxContainerHeader::from_slice(&bad_sr),
            Err(DspError::InvalidParameter {
                name: "sample_rate",
                ..
            })
        ));

        bad_sr[10..14].copy_from_slice(&f32::NEG_INFINITY.to_le_bytes());
        assert!(matches!(
            CalyxContainerHeader::from_slice(&bad_sr),
            Err(DspError::InvalidParameter {
                name: "sample_rate",
                ..
            })
        ));

        bad_sr[10..14].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(matches!(
            CalyxContainerHeader::from_slice(&bad_sr),
            Err(DspError::InvalidParameter {
                name: "sample_rate",
                ..
            })
        ));

        // Zero hop size
        let mut bad_hop =
            CalyxContainerHeader::new(QualityTier::Preview, 44100.0, 128, 8).to_bytes();
        bad_hop[14..16].copy_from_slice(&0u16.to_le_bytes());
        assert!(matches!(
            CalyxContainerHeader::from_slice(&bad_hop),
            Err(DspError::InvalidParameter {
                name: "hop_size",
                ..
            })
        ));

        // Zero num partials
        let mut bad_partials =
            CalyxContainerHeader::new(QualityTier::Preview, 44100.0, 128, 8).to_bytes();
        bad_partials[16..18].copy_from_slice(&0u16.to_le_bytes());
        assert!(matches!(
            CalyxContainerHeader::from_slice(&bad_partials),
            Err(DspError::InvalidParameter {
                name: "num_partials",
                ..
            })
        ));
    }

    #[test]
    fn test_adler32_checksum_correctness() {
        assert_eq!(adler32(b""), 1);
        let data = b"Calyx Spectral Synthesizer";
        let c1 = adler32(data);
        assert_ne!(c1, 0);

        // Bitflip changes checksum
        let mut modified = data.to_vec();
        modified[0] ^= 0x01;
        let c2 = adler32(&modified);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_tucker_container_roundtrip() {
        // 1. Create a 3D test tensor
        let shape = vec![4, 3, 5];
        let mut data = Vec::with_capacity(60);
        for i in 0..60 {
            data.push((i as f64 * 0.1).sin());
        }
        let tensor = Tensor::from_vec(shape.clone(), data).unwrap();

        // 2. Tucker HOSVD decomposition
        let tucker = TuckerDecomposition::decompose(&tensor, &[2, 2, 3]).unwrap();
        let tucker_data = TuckerContainerData::from_decomposition(&tucker);

        // 3. Assemble container
        let header = CalyxContainerHeader::new(QualityTier::Production, 48000.0, 256, 16);
        let layout = HypercubeLayout::new(shape).unwrap();
        let mut meta = HashMap::new();
        meta.insert("name".to_string(), "Violin Solo Legato".to_string());
        meta.insert("author".to_string(), "Calyx DSP Research".to_string());

        let container = CalyxContainer::new(header, layout)
            .with_metadata(meta)
            .with_tucker(tucker_data);

        // 4. Serialize to memory
        let mut buffer = Vec::new();
        container.write(&mut buffer).unwrap();
        assert!(!buffer.is_empty());

        // 5. Deserialize from memory
        let mut cursor = Cursor::new(&buffer);
        let recovered = CalyxContainer::read(&mut cursor).unwrap();

        // 6. Assert structural equality
        assert_eq!(recovered.header.magic, CALYX_MAGIC);
        assert_eq!(recovered.header.tier().unwrap(), QualityTier::Production);
        assert_eq!(
            recovered.metadata.get("name").unwrap(),
            "Violin Solo Legato"
        );
        assert_eq!(
            recovered.metadata.get("author").unwrap(),
            "Calyx DSP Research"
        );
        assert_eq!(recovered.layout.shape(), &[4, 3, 5]);

        let recovered_tucker_data = recovered
            .tucker_data
            .expect("Tucker data should be present");
        let recovered_tucker = recovered_tucker_data.to_decomposition().unwrap();

        // 7. Verify mathematical equivalence (Frobenius error == 0.0)
        let orig_recon = tucker.reconstruct_tensor().unwrap();
        let recovered_recon = recovered_tucker.reconstruct_tensor().unwrap();

        let diff: f64 = orig_recon
            .data()
            .iter()
            .zip(recovered_recon.data().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert_eq!(
            diff, 0.0,
            "Tucker reconstruction must match with 0.0 numerical error"
        );
    }

    #[test]
    fn test_cp_container_roundtrip() {
        // 1. Create a 3D test tensor
        let shape = vec![3, 3, 3];
        let mut data = Vec::with_capacity(27);
        for i in 0..27 {
            data.push(1.0 + (i as f64) * 0.2);
        }
        let tensor = Tensor::from_vec(shape.clone(), data).unwrap();

        // 2. CP-ALS decomposition (rank 2)
        let cp_model = CpAls::decompose(&tensor, 2, 20).unwrap();
        let cp_data = CpContainerData::from_model(&cp_model);

        // 3. Assemble container
        let header = CalyxContainerHeader::new(QualityTier::Reference, 96000.0, 512, 32);
        let layout = HypercubeLayout::new(shape).unwrap();
        let container = CalyxContainer::new(header, layout).with_cp(cp_data);

        // 4. Serialize to buffer
        let mut buf = Vec::new();
        container.write(&mut buf).unwrap();

        // 5. Deserialize
        let mut cursor = Cursor::new(&buf);
        let recovered = CalyxContainer::read(&mut cursor).unwrap();

        assert_eq!(recovered.header.tier().unwrap(), QualityTier::Reference);
        assert_eq!(recovered.header.sample_rate, 96000.0);
        let rec_cp = recovered
            .cp_data
            .expect("CP data must exist")
            .to_model()
            .unwrap();

        // 6. Verify bitwise float exactness
        assert_eq!(rec_cp.weights.len(), cp_model.weights.len());
        for (w_orig, w_rec) in cp_model.weights.iter().zip(rec_cp.weights.iter()) {
            assert_eq!(w_orig.to_bits(), w_rec.to_bits());
        }

        for (f_orig, f_rec) in cp_model.factors.iter().zip(rec_cp.factors.iter()) {
            assert_eq!(f_orig.as_slice(), f_rec.as_slice());
        }
    }

    #[test]
    fn test_full_hybrid_container_roundtrip() {
        let layout = HypercubeLayout::new(vec![2, 3, 4]).unwrap();
        let header = CalyxContainerHeader::new(QualityTier::Production, 48000.0, 256, 16);

        let tucker_data = TuckerContainerData {
            core_shape: vec![2, 2, 2],
            core_data: vec![0.5, -0.2, 0.1, 0.9, -0.4, 0.3, 0.8, -0.1],
            factors: vec![
                MatrixData::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
                MatrixData::new(3, 2, vec![0.7, 0.3, 0.1, 0.9, 0.5, 0.5]).unwrap(),
                MatrixData::new(4, 2, vec![0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4]).unwrap(),
            ],
        };

        let cp_data = CpContainerData {
            rank: 2,
            weights: vec![1.2, 0.8],
            factors: vec![
                MatrixData::new(2, 2, vec![0.8, 0.6, 0.6, 0.8]).unwrap(),
                MatrixData::new(3, 2, vec![0.5, 0.5, 0.7, 0.3, 0.1, 0.9]).unwrap(),
                MatrixData::new(4, 2, vec![0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5]).unwrap(),
            ],
        };

        let residual_data = ResidualContainerData {
            num_bands: 4,
            center_frequencies: vec![250.0, 1000.0, 4000.0, 12000.0],
            band_energies: vec![0.02, 0.05, 0.01, 0.003],
            noise_ratio: 0.083,
        };

        let container = CalyxContainer::new(header, layout)
            .with_metadata_entry("preset", "Concert Grand Piano")
            .with_tucker(tucker_data.clone())
            .with_cp(cp_data.clone())
            .with_residual(residual_data.clone());

        let mut buf = Vec::new();
        container.write(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let recovered = CalyxContainer::read(&mut cursor).unwrap();

        assert_eq!(
            recovered.metadata.get("preset").unwrap(),
            "Concert Grand Piano"
        );
        assert_eq!(recovered.tucker_data.unwrap(), tucker_data);
        assert_eq!(recovered.cp_data.unwrap(), cp_data);
        assert_eq!(recovered.residual_data.unwrap(), residual_data);
    }

    #[test]
    fn test_float_precision_preservation_exactness() {
        // Ensure standard floats, subnormals, infinities, and edge values preserve exact bit patterns
        let values = vec![
            0.0_f64,
            -0.0_f64,
            1.0,
            -1.0,
            std::f64::consts::PI,
            std::f64::consts::E,
            1e-308, // subnormal / small
            1e308,  // large
            f64::MIN_POSITIVE,
            f64::EPSILON,
        ];

        let matrix_data = MatrixData::new(1, values.len(), values.clone()).unwrap();
        let cp_data = CpContainerData {
            rank: values.len(),
            weights: values.clone(),
            factors: vec![matrix_data],
        };

        let layout = HypercubeLayout::new(vec![1]).unwrap();
        let header = CalyxContainerHeader::new(QualityTier::Reference, 48000.0, 256, 16);
        let container = CalyxContainer::new(header, layout).with_cp(cp_data);

        let mut buf = Vec::new();
        container.write(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let recovered = CalyxContainer::read(&mut cursor).unwrap();
        let rec_cp = recovered.cp_data.unwrap();

        for (&orig, &rec) in values.iter().zip(rec_cp.weights.iter()) {
            assert_eq!(
                orig.to_bits(),
                rec.to_bits(),
                "Float value {} bitwise preservation failed: expected {:x}, got {:x}",
                orig,
                orig.to_bits(),
                rec.to_bits()
            );
        }
    }

    #[test]
    fn test_corrupt_checksum_rejection() {
        let layout = HypercubeLayout::new(vec![2, 2]).unwrap();
        let header = CalyxContainerHeader::new(QualityTier::Preview, 44100.0, 256, 8);
        let container =
            CalyxContainer::new(header, layout).with_metadata_entry("author", "Tamper Test");

        let mut buf = Vec::new();
        container.write(&mut buf).unwrap();

        // Flip a byte in the META chunk payload (around offset 28)
        let payload_byte_idx = buf.len() - 8;
        buf[payload_byte_idx] ^= 0x42;

        let mut cursor = Cursor::new(&buf);
        let result = CalyxContainer::read(&mut cursor);
        assert!(matches!(
            result,
            Err(DspError::InvalidParameter {
                name: "checksum",
                ..
            })
        ));
    }

    #[test]
    fn test_truncated_container_rejection() {
        let layout = HypercubeLayout::new(vec![2, 2]).unwrap();
        let header = CalyxContainerHeader::new(QualityTier::Preview, 44100.0, 256, 8);
        let container = CalyxContainer::new(header, layout);

        let mut buf = Vec::new();
        container.write(&mut buf).unwrap();

        // Truncate at various byte offsets
        for cut in [0, 5, 17, 22, buf.len() - 3] {
            let truncated = &buf[..cut];
            let mut cursor = Cursor::new(truncated);
            assert!(CalyxContainer::read(&mut cursor).is_err());
        }
    }

    #[test]
    fn test_forward_compatibility_unknown_chunk() {
        let layout = HypercubeLayout::new(vec![2, 2]).unwrap();
        let header = CalyxContainerHeader::new(QualityTier::Preview, 44100.0, 256, 8);
        let container =
            CalyxContainer::new(header, layout).with_metadata_entry("tag", "forward_compat");

        let mut buf = Vec::new();
        container.write(&mut buf).unwrap();

        // Inject an unknown future chunk `b"CSTM"` at the end
        let custom_payload = b"Future custom neural vocoder weights or GUI parameters";
        write_chunk(&mut buf, *b"CSTM", custom_payload).unwrap();

        // Ensure reader smoothly reads container, parses known chunks, and ignores CSTM
        let mut cursor = Cursor::new(&buf);
        let recovered = CalyxContainer::read(&mut cursor).unwrap();
        assert_eq!(recovered.metadata.get("tag").unwrap(), "forward_compat");
    }

    #[test]
    fn test_file_save_and_load_roundtrip() {
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("test_container_save_load.calyx");

        let layout = HypercubeLayout::new(vec![3, 4]).unwrap();
        let header = CalyxContainerHeader::new(QualityTier::Production, 48000.0, 256, 16);
        let container =
            CalyxContainer::new(header, layout).with_metadata_entry("instrument", "French Horn");

        container.save_to_file(&path).unwrap();
        let loaded = CalyxContainer::load_from_file(&path).unwrap();

        assert_eq!(loaded.metadata.get("instrument").unwrap(), "French Horn");
        assert_eq!(loaded.layout.shape(), &[3, 4]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_promised_header_flags_enforcement() {
        // 1. Missing Tucker when FLAG_HAS_TUCKER is set
        let buf_tuck = create_minimal_valid_container(FLAG_HAS_TUCKER);
        let mut cursor = Cursor::new(&buf_tuck);
        let res = CalyxContainer::read(&mut cursor);
        assert!(matches!(
            res,
            Err(DspError::InvalidParameter {
                name: "container",
                ..
            })
        ));

        // 2. Missing CP when FLAG_HAS_CP is set
        let buf_cp = create_minimal_valid_container(FLAG_HAS_CP);
        let mut cursor = Cursor::new(&buf_cp);
        let res = CalyxContainer::read(&mut cursor);
        assert!(matches!(
            res,
            Err(DspError::InvalidParameter {
                name: "container",
                ..
            })
        ));

        // 3. Missing Residual when FLAG_HAS_RESIDUAL is set
        let buf_nois = create_minimal_valid_container(FLAG_HAS_RESIDUAL);
        let mut cursor = Cursor::new(&buf_nois);
        let res = CalyxContainer::read(&mut cursor);
        assert!(matches!(
            res,
            Err(DspError::InvalidParameter {
                name: "container",
                ..
            })
        ));

        // 4. Truncated container with all flags promised
        let mut buf_all =
            create_minimal_valid_container(FLAG_HAS_TUCKER | FLAG_HAS_CP | FLAG_HAS_RESIDUAL);
        // Truncated at end of AXES chunk
        let mut cursor = Cursor::new(&buf_all);
        assert!(matches!(
            CalyxContainer::read(&mut cursor),
            Err(DspError::InvalidParameter {
                name: "container",
                ..
            })
        ));

        // 5. Corrupted chunk identifier (e.g. b"TUCK" -> b"SUCK") with Adler32 valid for payload
        let mut tuck_payload = Vec::new();
        tuck_payload.extend_from_slice(&1u32.to_le_bytes()); // ndim = 1
        tuck_payload.extend_from_slice(&2u32.to_le_bytes()); // core_shape = [2]
        tuck_payload.extend_from_slice(&2u32.to_le_bytes()); // core_len = 2
        tuck_payload.extend_from_slice(&1.0f64.to_le_bytes());
        tuck_payload.extend_from_slice(&2.0f64.to_le_bytes());
        tuck_payload.extend_from_slice(&1u32.to_le_bytes()); // num_factors = 1
        tuck_payload.extend_from_slice(&2u32.to_le_bytes()); // rows = 2
        tuck_payload.extend_from_slice(&1u32.to_le_bytes()); // cols = 1
        tuck_payload.extend_from_slice(&1.0f64.to_le_bytes());
        tuck_payload.extend_from_slice(&1.0f64.to_le_bytes());

        // Write as b"SUCK" instead of b"TUCK"
        write_chunk(&mut buf_all, *b"SUCK", &tuck_payload).unwrap();
        let mut cursor = Cursor::new(&buf_all);
        // The reader skips SUCK as an unknown chunk, but returns error because TUCK was promised
        assert!(matches!(
            CalyxContainer::read(&mut cursor),
            Err(DspError::InvalidParameter {
                name: "container",
                ..
            })
        ));
    }

    #[test]
    fn test_overflow_dimension_rejection_tuck_and_cpmd() {
        // 1. CPMD rows * cols overflow (3_000_000_000 * 3_000_000_000)
        let mut buf = create_minimal_valid_container(FLAG_HAS_CP);
        let mut cpmd = Vec::new();
        cpmd.extend_from_slice(&1u32.to_le_bytes()); // rank = 1
        cpmd.extend_from_slice(&1u32.to_le_bytes()); // num_weights = 1
        cpmd.extend_from_slice(&1.0f64.to_le_bytes());
        cpmd.extend_from_slice(&1u32.to_le_bytes()); // num_factors = 1
        cpmd.extend_from_slice(&3_000_000_000u32.to_le_bytes()); // rows
        cpmd.extend_from_slice(&3_000_000_000u32.to_le_bytes()); // cols
        write_chunk(&mut buf, CHUNK_CPMD, &cpmd).unwrap();

        let mut cursor = Cursor::new(&buf);
        let res = CalyxContainer::read(&mut cursor);
        assert!(matches!(
            res,
            Err(DspError::InvalidParameter {
                name: "cpmd_chunk",
                ..
            })
        ));

        // 2. CPMD total * 8 overflow (1_000_000_000 * 3_000_000_000)
        let mut buf = create_minimal_valid_container(FLAG_HAS_CP);
        let mut cpmd = Vec::new();
        cpmd.extend_from_slice(&1u32.to_le_bytes());
        cpmd.extend_from_slice(&1u32.to_le_bytes());
        cpmd.extend_from_slice(&1.0f64.to_le_bytes());
        cpmd.extend_from_slice(&1u32.to_le_bytes());
        cpmd.extend_from_slice(&1_000_000_000u32.to_le_bytes());
        cpmd.extend_from_slice(&3_000_000_000u32.to_le_bytes());
        write_chunk(&mut buf, CHUNK_CPMD, &cpmd).unwrap();

        let mut cursor = Cursor::new(&buf);
        let res = CalyxContainer::read(&mut cursor);
        assert!(matches!(
            res,
            Err(DspError::InvalidParameter {
                name: "cpmd_chunk",
                ..
            })
        ));

        // 3. TUCK factor rows * cols overflow (3_000_000_000 * 3_000_000_000)
        let mut buf = create_minimal_valid_container(FLAG_HAS_TUCKER);
        let mut tuck = Vec::new();
        tuck.extend_from_slice(&1u32.to_le_bytes()); // ndim = 1
        tuck.extend_from_slice(&1u32.to_le_bytes()); // core_shape = [1]
        tuck.extend_from_slice(&1u32.to_le_bytes()); // core_len = 1
        tuck.extend_from_slice(&1.0f64.to_le_bytes());
        tuck.extend_from_slice(&1u32.to_le_bytes()); // num_factors = 1
        tuck.extend_from_slice(&3_000_000_000u32.to_le_bytes()); // rows
        tuck.extend_from_slice(&3_000_000_000u32.to_le_bytes()); // cols
        write_chunk(&mut buf, CHUNK_TUCK, &tuck).unwrap();

        let mut cursor = Cursor::new(&buf);
        let res = CalyxContainer::read(&mut cursor);
        assert!(matches!(
            res,
            Err(DspError::InvalidParameter {
                name: "tuck_chunk",
                ..
            })
        ));

        // 4. TUCK factor total * 8 overflow (1_000_000_000 * 3_000_000_000)
        let mut buf = create_minimal_valid_container(FLAG_HAS_TUCKER);
        let mut tuck = Vec::new();
        tuck.extend_from_slice(&1u32.to_le_bytes());
        tuck.extend_from_slice(&1u32.to_le_bytes());
        tuck.extend_from_slice(&1u32.to_le_bytes());
        tuck.extend_from_slice(&1.0f64.to_le_bytes());
        tuck.extend_from_slice(&1u32.to_le_bytes());
        tuck.extend_from_slice(&1_000_000_000u32.to_le_bytes());
        tuck.extend_from_slice(&3_000_000_000u32.to_le_bytes());
        write_chunk(&mut buf, CHUNK_TUCK, &tuck).unwrap();

        let mut cursor = Cursor::new(&buf);
        let res = CalyxContainer::read(&mut cursor);
        assert!(matches!(
            res,
            Err(DspError::InvalidParameter {
                name: "tuck_chunk",
                ..
            })
        ));

        // 5. MatrixData::new overflow
        let m_res = MatrixData::new(usize::MAX, 2, Vec::new());
        assert!(matches!(
            m_res,
            Err(DspError::InvalidParameter {
                name: "dimensions",
                ..
            })
        ));
    }

    #[test]
    fn test_capacity_allocation_safeguards() {
        // META chunk with crafted num_entries = 2_000_000_000
        let mut buf = create_minimal_valid_container(0);
        let mut meta = Vec::new();
        meta.extend_from_slice(&2_000_000_000u32.to_le_bytes());
        write_chunk(&mut buf, CHUNK_META, &meta).unwrap();
        let mut cursor = Cursor::new(&buf);
        assert!(matches!(
            CalyxContainer::read(&mut cursor),
            Err(DspError::InvalidParameter {
                name: "meta_chunk",
                ..
            })
        ));

        // AXES chunk with crafted num_axes = 2_000_000_000
        let header = CalyxContainerHeader::new(QualityTier::Production, 48000.0, 256, 16);
        let mut buf_axes = Vec::new();
        buf_axes.extend_from_slice(&header.to_bytes());
        let mut axes_payload = Vec::new();
        axes_payload.extend_from_slice(&2_000_000_000u32.to_le_bytes());
        write_chunk(&mut buf_axes, CHUNK_AXES, &axes_payload).unwrap();
        let mut cursor = Cursor::new(&buf_axes);
        assert!(matches!(
            CalyxContainer::read(&mut cursor),
            Err(DspError::InvalidParameter {
                name: "axes_chunk",
                ..
            })
        ));

        // TUCK chunk with crafted ndim = 2_000_000_000
        let mut buf = create_minimal_valid_container(FLAG_HAS_TUCKER);
        let mut tuck = Vec::new();
        tuck.extend_from_slice(&2_000_000_000u32.to_le_bytes());
        write_chunk(&mut buf, CHUNK_TUCK, &tuck).unwrap();
        let mut cursor = Cursor::new(&buf);
        assert!(matches!(
            CalyxContainer::read(&mut cursor),
            Err(DspError::InvalidParameter {
                name: "tuck_chunk",
                ..
            })
        ));

        // CPMD chunk with crafted num_weights = 2_000_000_000
        let mut buf = create_minimal_valid_container(FLAG_HAS_CP);
        let mut cpmd = Vec::new();
        cpmd.extend_from_slice(&1u32.to_le_bytes()); // rank
        cpmd.extend_from_slice(&2_000_000_000u32.to_le_bytes()); // num_weights
        write_chunk(&mut buf, CHUNK_CPMD, &cpmd).unwrap();
        let mut cursor = Cursor::new(&buf);
        assert!(matches!(
            CalyxContainer::read(&mut cursor),
            Err(DspError::InvalidParameter {
                name: "cpmd_chunk",
                ..
            })
        ));

        // NOIS chunk with crafted num_freqs = 2_000_000_000
        let mut buf = create_minimal_valid_container(FLAG_HAS_RESIDUAL);
        let mut nois = Vec::new();
        nois.extend_from_slice(&1u32.to_le_bytes()); // num_bands
        nois.extend_from_slice(&2_000_000_000u32.to_le_bytes()); // num_freqs
        write_chunk(&mut buf, CHUNK_NOIS, &nois).unwrap();
        let mut cursor = Cursor::new(&buf);
        assert!(matches!(
            CalyxContainer::read(&mut cursor),
            Err(DspError::InvalidParameter {
                name: "nois_chunk",
                ..
            })
        ));
    }
}
