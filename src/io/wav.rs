//! Pure Rust RIFF/WAVE audio writer and reader.
//!
//! Implements standard 16-bit linear PCM and 32-bit IEEE float RIFF/WAVE file
//! generation and inspection from first principles with strictly zero external crate
//! dependencies.
//!
//! Conforms to the standard RIFF/WAVE format:
//! - 44-byte canonical RIFF header
//! - Linear 16-bit signed integer PCM (`WAVE_FORMAT_PCM = 1`)
//! - 32-bit IEEE 754 floating point (`WAVE_FORMAT_IEEE_FLOAT = 3`)
//! - Little-endian sample serialization
//! - Arbitrary channel count and sample rate support

use crate::error::DspError;
use crate::synthetic::buffer::SignalBuffer;
use std::fmt;
use std::io::{Read, Write};
use std::path::Path;

/// Supported WAV audio sample encoding formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavFormat {
    /// Linear PCM (16-bit signed integer, `WAVE_FORMAT_PCM = 1`).
    Pcm16,
    /// IEEE 754 32-bit floating point (`WAVE_FORMAT_IEEE_FLOAT = 3`).
    Float32,
}

impl WavFormat {
    /// Returns the standard two-byte RIFF format tag.
    #[inline]
    pub fn format_code(&self) -> u16 {
        match self {
            WavFormat::Pcm16 => 1,
            WavFormat::Float32 => 3,
        }
    }

    /// Returns the bit depth per audio sample for this format.
    #[inline]
    pub fn bits_per_sample(&self) -> u16 {
        match self {
            WavFormat::Pcm16 => 16,
            WavFormat::Float32 => 32,
        }
    }

    /// Resolves a `WavFormat` from a RIFF format code and bit depth.
    pub fn from_code(audio_format: u16, bits_per_sample: u16) -> Result<Self, DspError> {
        match (audio_format, bits_per_sample) {
            (1, 16) => Ok(WavFormat::Pcm16),
            (3, 32) => Ok(WavFormat::Float32),
            _ => Err(DspError::invalid_parameter(
                "audio_format",
                format!(
                    "Unsupported audio format tag {} with {} bits per sample",
                    audio_format, bits_per_sample
                ),
            )),
        }
    }
}

impl fmt::Display for WavFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WavFormat::Pcm16 => write!(f, "16-bit PCM"),
            WavFormat::Float32 => write!(f, "32-bit IEEE Float"),
        }
    }
}

/// Specifications for a WAV audio stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavSpec {
    /// Sampling frequency in Hertz (e.g. 44100, 48000).
    pub sample_rate: u32,
    /// Number of audio channels (e.g. 1 for mono, 2 for stereo).
    pub channels: u16,
    /// Bit depth per sample (16 for PCM, 32 for float).
    pub bits_per_sample: u16,
}

impl WavSpec {
    /// Creates a new `WavSpec` after validating the arguments.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] if `sample_rate` is 0, `channels` is 0,
    /// or `bits_per_sample` is not 16 or 32.
    pub fn new(sample_rate: u32, channels: u16, bits_per_sample: u16) -> Result<Self, DspError> {
        if sample_rate == 0 {
            return Err(DspError::invalid_parameter(
                "sample_rate",
                "Sample rate must be greater than 0",
            ));
        }
        if channels == 0 {
            return Err(DspError::invalid_parameter(
                "channels",
                "Channel count must be greater than 0",
            ));
        }
        if bits_per_sample != 16 && bits_per_sample != 32 {
            return Err(DspError::invalid_parameter(
                "bits_per_sample",
                format!(
                    "Unsupported bit depth: {} (must be 16 or 32)",
                    bits_per_sample
                ),
            ));
        }
        Ok(Self {
            sample_rate,
            channels,
            bits_per_sample,
        })
    }

    /// Standard CD-quality specification: 44.1 kHz, stereo, 16-bit PCM.
    #[inline]
    pub fn cd_quality() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            bits_per_sample: 16,
        }
    }

    /// Standard studio-quality mono specification: 48.0 kHz, mono, 16-bit PCM.
    #[inline]
    pub fn studio_mono() -> Self {
        Self {
            sample_rate: 48000,
            channels: 1,
            bits_per_sample: 16,
        }
    }
}

/// Standard 44-byte RIFF/WAVE file header representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavHeader {
    /// RIFF chunk identifier (always `*b"RIFF"`).
    pub riff_id: [u8; 4],
    /// Overall file size minus 8 bytes (`36 + data_size`).
    pub chunk_size: u32,
    /// WAVE format identifier (always `*b"WAVE"`).
    pub wave_id: [u8; 4],
    /// Format subchunk identifier (always `*b"fmt "`).
    pub fmt_id: [u8; 4],
    /// Format subchunk size (16 for PCM / float).
    pub fmt_size: u32,
    /// Audio format code (1 = PCM, 3 = IEEE Float).
    pub audio_format: u16,
    /// Number of interleaved audio channels.
    pub channels: u16,
    /// Sampling frequency in Hertz.
    pub sample_rate: u32,
    /// Data transmission rate in bytes/sec (`sample_rate * block_align`).
    pub byte_rate: u32,
    /// Block alignment in bytes (`channels * bits_per_sample / 8`).
    pub block_align: u16,
    /// Quantization bit depth per sample.
    pub bits_per_sample: u16,
    /// Data subchunk identifier (always `*b"data"`).
    pub data_id: [u8; 4],
    /// Size of audio sample payload in bytes.
    pub data_size: u32,
}

impl WavHeader {
    /// Constructs a standard 44-byte RIFF/WAVE header for a given specification and sample count.
    ///
    /// # Arguments
    /// * `spec` - Audio stream specification (sample rate, channels, bits per sample).
    /// * `format` - Sample format encoding.
    /// * `num_samples` - Total number of samples across all channels.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] if parameters are invalid or total data size
    /// exceeds the 4GB RIFF limit.
    pub fn new(spec: WavSpec, format: WavFormat, num_samples: usize) -> Result<Self, DspError> {
        let bytes_per_sample = (format.bits_per_sample() / 8) as usize;
        let data_bytes = num_samples.checked_mul(bytes_per_sample).ok_or_else(|| {
            DspError::invalid_parameter("num_samples", "Sample count causes byte size overflow")
        })?;

        if data_bytes > (u32::MAX - 36) as usize {
            return Err(DspError::invalid_parameter(
                "num_samples",
                "Total WAV payload exceeds 4GB standard RIFF size limit",
            ));
        }

        let data_size = data_bytes as u32;
        let chunk_size = 36 + data_size;
        let block_align = spec.channels * (format.bits_per_sample() / 8);
        let byte_rate = spec.sample_rate * (block_align as u32);

        Ok(Self {
            riff_id: *b"RIFF",
            chunk_size,
            wave_id: *b"WAVE",
            fmt_id: *b"fmt ",
            fmt_size: 16,
            audio_format: format.format_code(),
            channels: spec.channels,
            sample_rate: spec.sample_rate,
            byte_rate,
            block_align,
            bits_per_sample: format.bits_per_sample(),
            data_id: *b"data",
            data_size,
        })
    }

    /// Serializes this 44-byte header to a fixed-size byte array.
    pub fn to_bytes(&self) -> [u8; 44] {
        let mut bytes = [0u8; 44];
        bytes[0..4].copy_from_slice(&self.riff_id);
        bytes[4..8].copy_from_slice(&self.chunk_size.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.wave_id);
        bytes[12..16].copy_from_slice(&self.fmt_id);
        bytes[16..20].copy_from_slice(&self.fmt_size.to_le_bytes());
        bytes[20..22].copy_from_slice(&self.audio_format.to_le_bytes());
        bytes[22..24].copy_from_slice(&self.channels.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.sample_rate.to_le_bytes());
        bytes[28..32].copy_from_slice(&self.byte_rate.to_le_bytes());
        bytes[32..34].copy_from_slice(&self.block_align.to_le_bytes());
        bytes[34..36].copy_from_slice(&self.bits_per_sample.to_le_bytes());
        bytes[36..40].copy_from_slice(&self.data_id);
        bytes[40..44].copy_from_slice(&self.data_size.to_le_bytes());
        bytes
    }

    /// Parses a standard 44-byte WAV header from a byte slice.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, DspError> {
        if bytes.len() < 44 {
            return Err(DspError::invalid_parameter(
                "wav_header",
                format!(
                    "Header byte slice too short: {} bytes (minimum 44 required)",
                    bytes.len()
                ),
            ));
        }
        let mut cursor = std::io::Cursor::new(bytes);
        read_wav_header(&mut cursor)
    }

    /// Returns the [`WavSpec`] corresponding to this header.
    #[inline]
    pub fn spec(&self) -> WavSpec {
        WavSpec {
            sample_rate: self.sample_rate,
            channels: self.channels,
            bits_per_sample: self.bits_per_sample,
        }
    }

    /// Returns the parsed [`WavFormat`] if supported.
    #[inline]
    pub fn format(&self) -> Result<WavFormat, DspError> {
        WavFormat::from_code(self.audio_format, self.bits_per_sample)
    }

    /// Total number of individual audio samples across all channels.
    #[inline]
    pub fn total_samples(&self) -> usize {
        let bytes_per_sample = (self.bits_per_sample / 8).max(1) as usize;
        (self.data_size as usize) / bytes_per_sample
    }

    /// Number of audio frames (time steps). Equal to `total_samples / channels`.
    #[inline]
    pub fn num_frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.total_samples() / (self.channels as usize)
        }
    }

    /// Duration of the audio file in seconds.
    #[inline]
    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.num_frames() as f64 / self.sample_rate as f64
        }
    }
}

/// Converts an `f64` audio sample in `[-1.0, 1.0]` to a 16-bit signed PCM integer (`i16`).
///
/// Handles clipping: samples exceeding `1.0` clamp to `32767` (`i16::MAX`),
/// and samples below `-1.0` clamp to `-32768` (`i16::MIN`). NaNs map to 0.
#[inline]
pub fn f64_to_pcm16(sample: f64) -> i16 {
    if sample.is_nan() {
        return 0;
    }
    let clamped = sample.clamp(-1.0, 1.0);
    let scaled = if clamped < 0.0 {
        clamped * 32768.0
    } else {
        clamped * 32767.0
    };
    scaled.round().clamp(-32768.0, 32767.0) as i16
}

/// Converts a 16-bit signed PCM integer (`i16`) to a normalized `f64` in `[-1.0, 1.0]`.
#[inline]
pub fn pcm16_to_f64(sample: i16) -> f64 {
    if sample < 0 {
        sample as f64 / 32768.0
    } else {
        sample as f64 / 32767.0
    }
}

/// Writes a standard 16-bit PCM RIFF WAVE file to the provided byte stream.
///
/// Samples outside `[-1.0, 1.0]` are clamped cleanly to avoid digital overflow distortion.
///
/// # Arguments
/// * `writer` - Target byte stream implementing [`Write`].
/// * `samples` - Interleaved sample buffer.
/// * `sample_rate` - Sampling frequency in Hz (must be > 0).
/// * `channels` - Channel count (must be > 0).
///
/// # Errors
/// Returns [`DspError::InvalidParameter`] if `sample_rate` is 0, `channels` is 0,
/// `samples.len()` is not a multiple of `channels`, or the payload exceeds the 4GB RIFF limit.
pub fn write_wav_pcm16<W: Write>(
    writer: &mut W,
    samples: &[f64],
    sample_rate: u32,
    channels: u16,
) -> Result<(), DspError> {
    if sample_rate == 0 {
        return Err(DspError::invalid_parameter(
            "sample_rate",
            "Sample rate must be greater than 0",
        ));
    }
    if channels == 0 {
        return Err(DspError::invalid_parameter(
            "channels",
            "Channel count must be greater than 0",
        ));
    }
    if samples.len() % (channels as usize) != 0 {
        return Err(DspError::invalid_parameter(
            "samples",
            format!(
                "Sample count {} is not a multiple of channel count {}",
                samples.len(),
                channels
            ),
        ));
    }

    let spec = WavSpec::new(sample_rate, channels, 16)?;
    let header = WavHeader::new(spec, WavFormat::Pcm16, samples.len())?;

    writer
        .write_all(&header.to_bytes())
        .map_err(|e| DspError::invalid_parameter("io", format!("Failed writing WAV header: {}", e)))?;

    // Write samples in chunks of 2048 samples (4096 bytes)
    let mut chunk = [0u8; 4096];
    let mut offset = 0;
    for &sample in samples {
        let pcm = f64_to_pcm16(sample);
        let bytes = pcm.to_le_bytes();
        chunk[offset] = bytes[0];
        chunk[offset + 1] = bytes[1];
        offset += 2;
        if offset == chunk.len() {
            writer
                .write_all(&chunk)
                .map_err(|e| DspError::invalid_parameter("io", format!("Failed writing PCM samples: {}", e)))?;
            offset = 0;
        }
    }
    if offset > 0 {
        writer
            .write_all(&chunk[..offset])
            .map_err(|e| DspError::invalid_parameter("io", format!("Failed writing PCM samples: {}", e)))?;
    }

    writer
        .flush()
        .map_err(|e| DspError::invalid_parameter("io", format!("Failed flushing WAV writer: {}", e)))?;

    Ok(())
}

/// Writes a standard 32-bit IEEE float RIFF WAVE file to the provided byte stream.
///
/// # Arguments
/// * `writer` - Target byte stream implementing [`Write`].
/// * `samples` - Interleaved sample buffer.
/// * `sample_rate` - Sampling frequency in Hz (must be > 0).
/// * `channels` - Channel count (must be > 0).
///
/// # Errors
/// Returns [`DspError::InvalidParameter`] if `sample_rate` is 0, `channels` is 0,
/// `samples.len()` is not a multiple of `channels`, or the payload exceeds the 4GB RIFF limit.
pub fn write_wav_float32<W: Write>(
    writer: &mut W,
    samples: &[f64],
    sample_rate: u32,
    channels: u16,
) -> Result<(), DspError> {
    if sample_rate == 0 {
        return Err(DspError::invalid_parameter(
            "sample_rate",
            "Sample rate must be greater than 0",
        ));
    }
    if channels == 0 {
        return Err(DspError::invalid_parameter(
            "channels",
            "Channel count must be greater than 0",
        ));
    }
    if samples.len() % (channels as usize) != 0 {
        return Err(DspError::invalid_parameter(
            "samples",
            format!(
                "Sample count {} is not a multiple of channel count {}",
                samples.len(),
                channels
            ),
        ));
    }

    let spec = WavSpec::new(sample_rate, channels, 32)?;
    let header = WavHeader::new(spec, WavFormat::Float32, samples.len())?;

    writer
        .write_all(&header.to_bytes())
        .map_err(|e| DspError::invalid_parameter("io", format!("Failed writing WAV header: {}", e)))?;

    // Write samples in chunks of 1024 samples (4096 bytes)
    let mut chunk = [0u8; 4096];
    let mut offset = 0;
    for &sample in samples {
        let f = if sample.is_nan() { 0.0f32 } else { sample as f32 };
        let bytes = f.to_le_bytes();
        chunk[offset..offset + 4].copy_from_slice(&bytes);
        offset += 4;
        if offset == chunk.len() {
            writer
                .write_all(&chunk)
                .map_err(|e| DspError::invalid_parameter("io", format!("Failed writing float samples: {}", e)))?;
            offset = 0;
        }
    }
    if offset > 0 {
        writer
            .write_all(&chunk[..offset])
            .map_err(|e| DspError::invalid_parameter("io", format!("Failed writing float samples: {}", e)))?;
    }

    writer
        .flush()
        .map_err(|e| DspError::invalid_parameter("io", format!("Failed flushing WAV writer: {}", e)))?;

    Ok(())
}

/// Exports mono audio samples to a 16-bit PCM WAV file at the specified path.
///
/// # Arguments
/// * `path` - Destination filesystem path.
/// * `samples` - Audio samples.
/// * `sample_rate` - Sampling frequency in Hz.
///
/// # Errors
/// Returns [`DspError::InvalidParameter`] on I/O error or invalid parameter.
pub fn export_wav_file(
    path: impl AsRef<Path>,
    samples: &[f64],
    sample_rate: u32,
) -> Result<(), DspError> {
    let p = path.as_ref();
    let file = std::fs::File::create(p).map_err(|e| {
        DspError::invalid_parameter(
            "path",
            format!("Failed to create WAV file '{}': {}", p.display(), e),
        )
    })?;
    let mut writer = std::io::BufWriter::new(file);
    write_wav_pcm16(&mut writer, samples, sample_rate, 1)?;
    writer.flush().map_err(|e| {
        DspError::invalid_parameter(
            "io",
            format!("Failed to flush WAV file '{}': {}", p.display(), e),
        )
    })?;
    Ok(())
}

/// Exports mono audio samples to a 32-bit IEEE float WAV file at the specified path.
///
/// # Arguments
/// * `path` - Destination filesystem path.
/// * `samples` - Audio samples.
/// * `sample_rate` - Sampling frequency in Hz.
///
/// # Errors
/// Returns [`DspError::InvalidParameter`] on I/O error or invalid parameter.
pub fn export_wav_file_float32(
    path: impl AsRef<Path>,
    samples: &[f64],
    sample_rate: u32,
) -> Result<(), DspError> {
    let p = path.as_ref();
    let file = std::fs::File::create(p).map_err(|e| {
        DspError::invalid_parameter(
            "path",
            format!("Failed to create WAV file '{}': {}", p.display(), e),
        )
    })?;
    let mut writer = std::io::BufWriter::new(file);
    write_wav_float32(&mut writer, samples, sample_rate, 1)?;
    writer.flush().map_err(|e| {
        DspError::invalid_parameter(
            "io",
            format!("Failed to flush WAV file '{}': {}", p.display(), e),
        )
    })?;
    Ok(())
}

/// Exports a [`SignalBuffer`] directly to a 16-bit PCM WAV file.
///
/// # Arguments
/// * `path` - Destination filesystem path.
/// * `buffer` - The signal buffer containing audio samples and sampling rate.
///
/// # Errors
/// Returns [`DspError::InvalidParameter`] if sampling rate is invalid or writing fails.
pub fn export_signal_buffer(
    path: impl AsRef<Path>,
    buffer: &SignalBuffer,
) -> Result<(), DspError> {
    if buffer.sample_rate <= 0.0 || buffer.sample_rate.is_nan() {
        return Err(DspError::invalid_parameter(
            "sample_rate",
            format!("Invalid SignalBuffer sample rate: {}", buffer.sample_rate),
        ));
    }
    let sample_rate = buffer.sample_rate.round() as u32;
    export_wav_file(path, &buffer.samples, sample_rate)
}

/// Reads and parses a RIFF/WAVE header from any byte stream.
///
/// Upon successful return, `reader` is positioned at the start of the audio data chunk.
///
/// Handles standard 44-byte headers as well as files containing extra metadata chunks
/// or extended format subchunks before the `data` chunk.
///
/// # Errors
/// Returns [`DspError::InvalidParameter`] if the header is malformed, truncated, or lacks
/// essential RIFF, WAVE, fmt, or data chunks.
pub fn read_wav_header<R: Read>(reader: &mut R) -> Result<WavHeader, DspError> {
    let mut riff_head = [0u8; 12];
    reader.read_exact(&mut riff_head).map_err(|e| {
        DspError::invalid_parameter("wav_header", format!("Failed to read RIFF header: {}", e))
    })?;

    let riff_id: [u8; 4] = riff_head[0..4].try_into().unwrap();
    if &riff_id != b"RIFF" {
        return Err(DspError::invalid_parameter(
            "wav_header",
            format!("Expected RIFF header magic, found {:?}", riff_id),
        ));
    }

    let chunk_size = u32::from_le_bytes(riff_head[4..8].try_into().unwrap());
    let wave_id: [u8; 4] = riff_head[8..12].try_into().unwrap();
    if &wave_id != b"WAVE" {
        return Err(DspError::invalid_parameter(
            "wav_header",
            format!("Expected WAVE format ID, found {:?}", wave_id),
        ));
    }

    let mut fmt_fields: Option<([u8; 4], u32, u16, u16, u32, u32, u16, u16)> = None;

    loop {
        let mut chunk_id = [0u8; 4];
        if let Err(e) = reader.read_exact(&mut chunk_id) {
            return Err(DspError::invalid_parameter(
                "wav_header",
                format!("Failed reading subchunk ID: {}", e),
            ));
        }

        let mut size_bytes = [0u8; 4];
        if let Err(e) = reader.read_exact(&mut size_bytes) {
            return Err(DspError::invalid_parameter(
                "wav_header",
                format!("Failed reading subchunk size: {}", e),
            ));
        }
        let subchunk_size = u32::from_le_bytes(size_bytes);

        if &chunk_id == b"fmt " {
            if subchunk_size < 16 {
                return Err(DspError::invalid_parameter(
                    "wav_header",
                    format!(
                        "Format subchunk size too small: {} (expected >= 16)",
                        subchunk_size
                    ),
                ));
            }

            let mut fmt_buf = [0u8; 16];
            reader.read_exact(&mut fmt_buf).map_err(|e| {
                DspError::invalid_parameter("wav_header", format!("Failed reading fmt chunk: {}", e))
            })?;

            let audio_format = u16::from_le_bytes(fmt_buf[0..2].try_into().unwrap());
            let channels = u16::from_le_bytes(fmt_buf[2..4].try_into().unwrap());
            let sample_rate = u32::from_le_bytes(fmt_buf[4..8].try_into().unwrap());
            let byte_rate = u32::from_le_bytes(fmt_buf[8..12].try_into().unwrap());
            let block_align = u16::from_le_bytes(fmt_buf[12..14].try_into().unwrap());
            let bits_per_sample = u16::from_le_bytes(fmt_buf[14..16].try_into().unwrap());

            if channels == 0 {
                return Err(DspError::invalid_parameter(
                    "channels",
                    "Channel count in WAV header is 0",
                ));
            }
            if sample_rate == 0 {
                return Err(DspError::invalid_parameter(
                    "sample_rate",
                    "Sample rate in WAV header is 0",
                ));
            }

            // Skip any extended format bytes (e.g., cbSize)
            let remaining = subchunk_size - 16;
            if remaining > 0 {
                let mut discard = vec![0u8; remaining as usize];
                reader.read_exact(&mut discard).map_err(|e| {
                    DspError::invalid_parameter(
                        "wav_header",
                        format!("Failed skipping extra fmt bytes: {}", e),
                    )
                })?;
            }

            fmt_fields = Some((
                chunk_id,
                subchunk_size,
                audio_format,
                channels,
                sample_rate,
                byte_rate,
                block_align,
                bits_per_sample,
            ));
        } else if &chunk_id == b"data" {
            let (
                fmt_id,
                fmt_size,
                audio_format,
                channels,
                sample_rate,
                byte_rate,
                block_align,
                bits_per_sample,
            ) = fmt_fields.ok_or_else(|| {
                DspError::invalid_parameter("wav_header", "Found data chunk before fmt chunk")
            })?;

            return Ok(WavHeader {
                riff_id,
                chunk_size,
                wave_id,
                fmt_id,
                fmt_size,
                audio_format,
                channels,
                sample_rate,
                byte_rate,
                block_align,
                bits_per_sample,
                data_id: chunk_id,
                data_size: subchunk_size,
            });
        } else {
            // Skip unrecognized chunks (e.g. JUNK, LIST, ID3)
            let mut discard = vec![0u8; subchunk_size as usize];
            reader.read_exact(&mut discard).map_err(|e| {
                DspError::invalid_parameter(
                    "wav_header",
                    format!("Failed skipping unrecognized chunk {:?}: {}", chunk_id, e),
                )
            })?;
        }
    }
}

/// Reads a 16-bit PCM RIFF WAVE stream, returning the header and reconstructed samples.
///
/// # Arguments
/// * `reader` - Byte stream containing a valid 16-bit PCM WAV.
///
/// # Errors
/// Returns [`DspError::InvalidParameter`] if the file is not 16-bit PCM, has corrupt headers,
/// or terminates prematurely.
pub fn read_wav_pcm16<R: Read>(reader: &mut R) -> Result<(WavHeader, Vec<f64>), DspError> {
    let header = read_wav_header(reader)?;
    if header.audio_format != 1 {
        return Err(DspError::invalid_parameter(
            "audio_format",
            format!("Expected PCM format (1), got {}", header.audio_format),
        ));
    }
    if header.bits_per_sample != 16 {
        return Err(DspError::invalid_parameter(
            "bits_per_sample",
            format!("Expected 16 bits per sample, got {}", header.bits_per_sample),
        ));
    }
    if header.data_size % 2 != 0 {
        return Err(DspError::invalid_parameter(
            "data_size",
            format!("PCM 16-bit data size must be even, got {}", header.data_size),
        ));
    }

    let num_samples = (header.data_size / 2) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    let mut buf = [0u8; 4096];
    let mut remaining = header.data_size as usize;
    while remaining > 0 {
        let to_read = remaining.min(buf.len());
        reader.read_exact(&mut buf[..to_read]).map_err(|e| {
            DspError::invalid_parameter("io", format!("Failed reading PCM sample data: {}", e))
        })?;
        for pair in buf[..to_read].chunks_exact(2) {
            let pcm = i16::from_le_bytes([pair[0], pair[1]]);
            samples.push(pcm16_to_f64(pcm));
        }
        remaining -= to_read;
    }

    Ok((header, samples))
}

/// Reads a 32-bit IEEE float RIFF WAVE stream, returning the header and reconstructed samples.
///
/// # Arguments
/// * `reader` - Byte stream containing a valid 32-bit float WAV.
///
/// # Errors
/// Returns [`DspError::InvalidParameter`] if the file is not 32-bit IEEE float, has corrupt
/// headers, or terminates prematurely.
pub fn read_wav_float32<R: Read>(reader: &mut R) -> Result<(WavHeader, Vec<f64>), DspError> {
    let header = read_wav_header(reader)?;
    if header.audio_format != 3 {
        return Err(DspError::invalid_parameter(
            "audio_format",
            format!("Expected IEEE Float format (3), got {}", header.audio_format),
        ));
    }
    if header.bits_per_sample != 32 {
        return Err(DspError::invalid_parameter(
            "bits_per_sample",
            format!("Expected 32 bits per sample, got {}", header.bits_per_sample),
        ));
    }
    if header.data_size % 4 != 0 {
        return Err(DspError::invalid_parameter(
            "data_size",
            format!(
                "IEEE Float 32-bit data size must be multiple of 4, got {}",
                header.data_size
            ),
        ));
    }

    let num_samples = (header.data_size / 4) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    let mut buf = [0u8; 4096];
    let mut remaining = header.data_size as usize;
    while remaining > 0 {
        let to_read = remaining.min(buf.len());
        reader.read_exact(&mut buf[..to_read]).map_err(|e| {
            DspError::invalid_parameter("io", format!("Failed reading float sample data: {}", e))
        })?;
        for quartet in buf[..to_read].chunks_exact(4) {
            let f = f32::from_le_bytes([quartet[0], quartet[1], quartet[2], quartet[3]]);
            samples.push(f as f64);
        }
        remaining -= to_read;
    }

    Ok((header, samples))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_wav_header_byte_encoding() {
        let spec = WavSpec::new(48000, 2, 16).unwrap();
        let num_samples = 96000; // 1 second of stereo audio
        let header = WavHeader::new(spec, WavFormat::Pcm16, num_samples).unwrap();
        let bytes = header.to_bytes();

        assert_eq!(bytes.len(), 44);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(&bytes[36..40], b"data");

        let chunk_size = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let data_size = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        assert_eq!(data_size, (num_samples * 2) as u32);
        assert_eq!(chunk_size, 36 + data_size);

        let parsed = WavHeader::from_slice(&bytes).unwrap();
        assert_eq!(parsed, header);
        assert_eq!(parsed.sample_rate, 48000);
        assert_eq!(parsed.channels, 2);
        assert_eq!(parsed.bits_per_sample, 16);
        assert_eq!(parsed.audio_format, 1);
        assert_eq!(parsed.byte_rate, 48000 * 2 * 2);
        assert_eq!(parsed.block_align, 4);
        assert_eq!(parsed.num_frames(), 48000);
        assert!((parsed.duration_seconds() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pcm16_scaling_and_clamping() {
        // Zero / silence
        assert_eq!(f64_to_pcm16(0.0), 0);
        assert_eq!(pcm16_to_f64(0), 0.0);

        // Maximum amplitudes
        assert_eq!(f64_to_pcm16(1.0), 32767);
        assert_eq!(pcm16_to_f64(32767), 1.0);

        assert_eq!(f64_to_pcm16(-1.0), -32768);
        assert_eq!(pcm16_to_f64(-32768), -1.0);

        // Over-range clipping
        assert_eq!(f64_to_pcm16(1.5), 32767);
        assert_eq!(f64_to_pcm16(100.0), 32767);
        assert_eq!(f64_to_pcm16(-1.5), -32768);
        assert_eq!(f64_to_pcm16(-100.0), -32768);

        // NaN handling
        assert_eq!(f64_to_pcm16(f64::NAN), 0);

        // Intermediate values within quantization step
        let test_vals = [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75];
        for &val in &test_vals {
            let pcm = f64_to_pcm16(val);
            let reconstructed = pcm16_to_f64(pcm);
            assert!(
                (reconstructed - val).abs() <= 1.0 / 32767.0,
                "Val: {}, Reconstructed: {}",
                val,
                reconstructed
            );
        }
    }

    #[test]
    fn test_pcm16_little_endian_serialization() {
        let samples = [0.0, 1.0, -1.0];
        let mut buffer = Vec::new();
        write_wav_pcm16(&mut buffer, &samples, 44100, 1).unwrap();

        assert_eq!(buffer.len(), 44 + 6);
        // Header verification
        assert_eq!(&buffer[0..4], b"RIFF");
        // Sample bytes (little endian)
        // 0.0 -> 0x0000
        assert_eq!(buffer[44], 0x00);
        assert_eq!(buffer[45], 0x00);
        // 1.0 -> 32767 (0x7FFF) -> [0xFF, 0x7F]
        assert_eq!(buffer[46], 0xFF);
        assert_eq!(buffer[47], 0x7F);
        // -1.0 -> -32768 (0x8000) -> [0x00, 0x80]
        assert_eq!(buffer[48], 0x00);
        assert_eq!(buffer[49], 0x80);
    }

    #[test]
    fn test_float32_serialization() {
        let samples = [0.0, 1.0, -1.0, 0.5];
        let mut buffer = Vec::new();
        write_wav_float32(&mut buffer, &samples, 48000, 1).unwrap();

        assert_eq!(buffer.len(), 44 + 16);
        assert_eq!(&buffer[0..4], b"RIFF");
        assert_eq!(&buffer[8..12], b"WAVE");
        assert_eq!(&buffer[36..40], b"data");

        // Verify float values match bitwise
        for (i, &s) in samples.iter().enumerate() {
            let expected_bytes = (s as f32).to_le_bytes();
            let actual_bytes = &buffer[44 + i * 4..44 + (i + 1) * 4];
            assert_eq!(actual_bytes, &expected_bytes);
        }
    }

    #[test]
    fn test_roundtrip_pcm16_mono() {
        let sample_rate = 44100;
        let num_samples = 4410; // 0.1s
        let mut original = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let t = i as f64 / sample_rate as f64;
            // 440 Hz sine wave
            original.push((2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.9);
        }

        let mut buffer = Vec::new();
        write_wav_pcm16(&mut buffer, &original, sample_rate, 1).unwrap();

        let mut cursor = Cursor::new(buffer);
        let (header, decoded) = read_wav_pcm16(&mut cursor).unwrap();

        assert_eq!(header.sample_rate, sample_rate);
        assert_eq!(header.channels, 1);
        assert_eq!(header.bits_per_sample, 16);
        assert_eq!(header.audio_format, 1);
        assert_eq!(decoded.len(), original.len());

        for (orig, dec) in original.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 1e-4,
                "orig: {}, dec: {}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_roundtrip_pcm16_stereo() {
        let sample_rate = 48000;
        let num_frames = 2400;
        let mut original = Vec::with_capacity(num_frames * 2);
        for i in 0..num_frames {
            let t = i as f64 / sample_rate as f64;
            let left = (2.0 * std::f64::consts::PI * 220.0 * t).sin() * 0.5;
            let right = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.8;
            original.push(left);
            original.push(right);
        }

        let mut buffer = Vec::new();
        write_wav_pcm16(&mut buffer, &original, sample_rate, 2).unwrap();

        let mut cursor = Cursor::new(buffer);
        let (header, decoded) = read_wav_pcm16(&mut cursor).unwrap();

        assert_eq!(header.sample_rate, sample_rate);
        assert_eq!(header.channels, 2);
        assert_eq!(header.bits_per_sample, 16);
        assert_eq!(decoded.len(), original.len());

        for (orig, dec) in original.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 1e-4);
        }
    }

    #[test]
    fn test_roundtrip_float32() {
        let sample_rate = 48000;
        let num_samples = 4800;
        let mut original = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let t = i as f64 / sample_rate as f64;
            original.push((2.0 * std::f64::consts::PI * 880.0 * t).sin() * 0.95);
        }

        let mut buffer = Vec::new();
        write_wav_float32(&mut buffer, &original, sample_rate, 1).unwrap();

        let mut cursor = Cursor::new(buffer);
        let (header, decoded) = read_wav_float32(&mut cursor).unwrap();

        assert_eq!(header.sample_rate, sample_rate);
        assert_eq!(header.channels, 1);
        assert_eq!(header.bits_per_sample, 32);
        assert_eq!(header.audio_format, 3);
        assert_eq!(decoded.len(), original.len());

        for (orig, dec) in original.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 1e-6,
                "orig: {}, dec: {}",
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_empty_slice_handling() {
        // PCM16 empty
        let mut pcm_buf = Vec::new();
        write_wav_pcm16(&mut pcm_buf, &[], 44100, 1).unwrap();
        assert_eq!(pcm_buf.len(), 44);

        let mut cursor = Cursor::new(&pcm_buf);
        let (header, samples) = read_wav_pcm16(&mut cursor).unwrap();
        assert_eq!(header.data_size, 0);
        assert_eq!(header.chunk_size, 36);
        assert!(samples.is_empty());

        // Float32 empty
        let mut float_buf = Vec::new();
        write_wav_float32(&mut float_buf, &[], 48000, 2).unwrap();
        assert_eq!(float_buf.len(), 44);

        let mut cursor = Cursor::new(&float_buf);
        let (header, samples) = read_wav_float32(&mut cursor).unwrap();
        assert_eq!(header.data_size, 0);
        assert_eq!(header.chunk_size, 36);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_edge_cases_silence_dc_clipping() {
        let samples = [0.0, 0.0, 0.0, 0.5, -0.5, 1.0, -1.0, 2.5, -3.2];
        let mut buffer = Vec::new();
        write_wav_pcm16(&mut buffer, &samples, 44100, 1).unwrap();

        let mut cursor = Cursor::new(buffer);
        let (_, decoded) = read_wav_pcm16(&mut cursor).unwrap();

        assert_eq!(decoded[0], 0.0);
        assert_eq!(decoded[1], 0.0);
        assert_eq!(decoded[2], 0.0);
        assert!((decoded[3] - 0.5).abs() < 1e-4);
        assert!((decoded[4] - (-0.5)).abs() < 1e-4);
        assert_eq!(decoded[5], 1.0);
        assert_eq!(decoded[6], -1.0);
        // Clipped values must saturate to 1.0 and -1.0
        assert_eq!(decoded[7], 1.0);
        assert_eq!(decoded[8], -1.0);
    }

    #[test]
    fn test_invalid_parameters() {
        let mut buf = Vec::new();

        // Sample rate = 0
        let res = write_wav_pcm16(&mut buf, &[0.0], 0, 1);
        assert!(matches!(res, Err(DspError::InvalidParameter { name: "sample_rate", .. })));

        // Channel count = 0
        let res = write_wav_pcm16(&mut buf, &[0.0], 44100, 0);
        assert!(matches!(res, Err(DspError::InvalidParameter { name: "channels", .. })));

        // Incomplete stereo frame
        let res = write_wav_pcm16(&mut buf, &[0.0, 0.1, 0.2], 44100, 2);
        assert!(matches!(res, Err(DspError::InvalidParameter { name: "samples", .. })));

        // Float32 sample rate = 0
        let res = write_wav_float32(&mut buf, &[0.0], 0, 1);
        assert!(matches!(res, Err(DspError::InvalidParameter { name: "sample_rate", .. })));

        // Float32 channel count = 0
        let res = write_wav_float32(&mut buf, &[0.0], 48000, 0);
        assert!(matches!(res, Err(DspError::InvalidParameter { name: "channels", .. })));

        // Spec validation
        assert!(WavSpec::new(0, 1, 16).is_err());
        assert!(WavSpec::new(44100, 0, 16).is_err());
        assert!(WavSpec::new(44100, 1, 24).is_err());
    }

    #[test]
    fn test_corrupt_headers() {
        // Less than 44 bytes
        let short = [0u8; 30];
        let mut cursor = Cursor::new(&short);
        assert!(read_wav_header(&mut cursor).is_err());

        // Bad RIFF magic
        let mut bad_riff = [0u8; 44];
        bad_riff[0..4].copy_from_slice(b"RIFX");
        bad_riff[8..12].copy_from_slice(b"WAVE");
        bad_riff[12..16].copy_from_slice(b"fmt ");
        bad_riff[36..40].copy_from_slice(b"data");
        let mut cursor = Cursor::new(&bad_riff);
        assert!(read_wav_header(&mut cursor).is_err());

        // Bad WAVE format ID
        let mut bad_wave = [0u8; 44];
        bad_wave[0..4].copy_from_slice(b"RIFF");
        bad_wave[8..12].copy_from_slice(b"AVI ");
        bad_wave[12..16].copy_from_slice(b"fmt ");
        bad_wave[36..40].copy_from_slice(b"data");
        let mut cursor = Cursor::new(&bad_wave);
        assert!(read_wav_header(&mut cursor).is_err());
    }

    #[test]
    fn test_export_wav_file_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("calyx_test_export.wav");

        let samples = [0.0, 0.5, -0.5, 0.9, -0.9];
        export_wav_file(&path, &samples, 48000).unwrap();

        let mut file = std::fs::File::open(&path).unwrap();
        let (header, read_samples) = read_wav_pcm16(&mut file).unwrap();

        assert_eq!(header.sample_rate, 48000);
        assert_eq!(header.channels, 1);
        assert_eq!(read_samples.len(), samples.len());
        for (orig, dec) in samples.iter().zip(read_samples.iter()) {
            assert!((orig - dec).abs() < 1e-4);
        }

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_signal_buffer() {
        let dir = std::env::temp_dir();
        let path = dir.join("calyx_test_signal_buffer.wav");

        let buf = SignalBuffer::new(
            44100.0,
            vec![0.1, 0.2, 0.3, -0.1, -0.2],
            Default::default(),
        );
        export_signal_buffer(&path, &buf).unwrap();

        let mut file = std::fs::File::open(&path).unwrap();
        let (header, read_samples) = read_wav_pcm16(&mut file).unwrap();

        assert_eq!(header.sample_rate, 44100);
        assert_eq!(read_samples.len(), buf.samples.len());
        for (orig, dec) in buf.samples.iter().zip(read_samples.iter()) {
            assert!((orig - dec).abs() < 1e-4);
        }

        let _ = std::fs::remove_file(&path);
    }
}
