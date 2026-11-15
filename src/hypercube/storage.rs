//! Dense, sparse, and compressed spectral hypercube storage containers.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library.

use std::collections::HashMap;
use crate::error::{DspError, DspResult};
use crate::math::Matrix;
use crate::spectral::{HarmonicPartial, NoiseBandEnvelope, SpectralFrame};
use crate::hypercube::coordinates::HypercubeLayout;
use crate::hypercube::tucker::TuckerDecomposition;

/// Dense hypercube storage container with contiguous frame buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseHypercube {
    /// Hypercube axis layout and striding.
    pub layout: HypercubeLayout,
    /// Flat vector of optional spectral frames.
    pub frames: Vec<Option<SpectralFrame>>,
}

impl DenseHypercube {
    /// Creates an empty dense hypercube with the given layout.
    pub fn new(layout: HypercubeLayout) -> Self {
        let total = layout.total_cells();
        Self {
            layout,
            frames: vec![None; total],
        }
    }

    /// Inserts a frame at the specified discrete coordinates.
    pub fn insert(&mut self, coords: &[usize], frame: SpectralFrame) -> DspResult<()> {
        let idx = self.layout.flat_index(coords)?;
        self.frames[idx] = Some(frame);
        Ok(())
    }

    /// Retrieves a frame reference at the specified discrete coordinates.
    pub fn get(&self, coords: &[usize]) -> DspResult<&SpectralFrame> {
        let idx = self.layout.flat_index(coords)?;
        match &self.frames[idx] {
            Some(f) => Ok(f),
            None => Err(DspError::invalid_parameter(
                "coords",
                "Frame at requested coordinate is unpopulated",
            )),
        }
    }
}

/// Sparse hypercube storage container for sparsely sampled coordinate spaces.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseHypercube {
    /// Hypercube axis layout.
    pub layout: HypercubeLayout,
    /// Map from discrete coordinate vectors to spectral frames.
    pub frames: HashMap<Vec<usize>, SpectralFrame>,
}

impl SparseHypercube {
    /// Creates an empty sparse hypercube with the given layout.
    pub fn new(layout: HypercubeLayout) -> Self {
        Self {
            layout,
            frames: HashMap::new(),
        }
    }

    /// Inserts a frame at the specified discrete coordinates.
    pub fn insert(&mut self, coords: &[usize], frame: SpectralFrame) -> DspResult<()> {
        let _ = self.layout.flat_index(coords)?;
        self.frames.insert(coords.to_vec(), frame);
        Ok(())
    }

    /// Retrieves a frame reference at the specified discrete coordinates.
    pub fn get(&self, coords: &[usize]) -> DspResult<&SpectralFrame> {
        let _ = self.layout.flat_index(coords)?;
        self.frames.get(coords).ok_or_else(|| {
            DspError::invalid_parameter("coords", "Frame at requested coordinate not present in sparse storage")
        })
    }
}

/// Compressed parametric hypercube container holding decomposed factors and frames.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressedHypercube {
    /// Hypercube layout.
    pub layout: HypercubeLayout,
    /// Concrete or reconstructed spectral frames.
    pub frames: Vec<SpectralFrame>,
    /// Compression kind ("PCA", "NMF", "Tucker", "Unit").
    pub kind: String,
    /// Effective decomposition rank.
    pub rank: usize,
    /// Audio sampling rate in Hz.
    pub sample_rate: f64,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Frame timestamps.
    pub timestamps: Vec<f64>,
    /// Frame fundamental frequencies.
    pub fundamental_freqs: Vec<f64>,
    /// Projection weights matrix (e.g. PCA weights or NMF W).
    pub weights: Option<Matrix<f64>>,
    /// Basis matrix (e.g. PCA basis or NMF H).
    pub basis: Option<Matrix<f64>>,
    /// Higher-Order Tucker decomposition if applicable.
    pub tucker: Option<TuckerDecomposition>,
    /// Residual noise envelopes.
    pub residuals: Vec<NoiseBandEnvelope>,
}

impl CompressedHypercube {
    /// Constructs a dummy unit 5D cube spanning [2, 2, 2, 2, 2] for boundary testing.
    pub fn dummy_unit_cube() -> Self {
        let layout = HypercubeLayout::new(vec![2, 2, 2, 2, 2]).expect("5D unit layout creation");
        let total = layout.total_cells(); // 32
        let mut frames = Vec::with_capacity(total);

        for idx in 0..total {
            let t = idx as f64 * 0.01;
            frames.push(SpectralFrame {
                timestamp: t,
                fundamental_freq: 440.0,
                partials: vec![
                    HarmonicPartial {
                        harmonic_index: 1,
                        frequency: 440.0,
                        amplitude: 0.8,
                        phase: 0.0,
                    },
                    HarmonicPartial {
                        harmonic_index: 2,
                        frequency: 880.0,
                        amplitude: 0.4,
                        phase: 0.0,
                    },
                ],
                residual: NoiseBandEnvelope {
                    band_energies: vec![0.01; 16],
                    noise_ratio: 0.1,
                },
            });
        }

        Self {
            layout,
            frames,
            kind: "Unit".to_string(),
            rank: 1,
            sample_rate: 48000.0,
            hop_size: 256,
            timestamps: vec![0.0; total],
            fundamental_freqs: vec![440.0; total],
            weights: None,
            basis: None,
            tucker: None,
            residuals: vec![NoiseBandEnvelope::default(); total],
        }
    }

    /// Discrete frame retrieval with boundary checks.
    pub fn get_frame(&self, indices: &[usize]) -> DspResult<&SpectralFrame> {
        let idx = self.layout.flat_index(indices)?;
        self.frames.get(idx).ok_or_else(|| {
            DspError::invalid_parameter("indices", "Frame index out of bounds in compressed hypercube")
        })
    }

    /// Continuous multi-linear parameter interpolation over normalized coordinates [0.0, 1.0]^D.
    ///
    /// # Errors
    /// Returns `DspError::InvalidParameter` if any coordinate is outside [0.0, 1.0].
    /// Returns `DspError::DimensionMismatch` if coordinate length does not match hypercube dimension.
    pub fn interpolate(&self, coords: &[f64]) -> DspResult<SpectralFrame> {
        let num_axes = self.layout.axes().len();
        if coords.len() != num_axes {
            return Err(DspError::dimension_mismatch(
                self.layout.axes().to_vec(),
                coords.iter().map(|&c| c as usize).collect::<Vec<_>>(),
            ));
        }

        // Validate normalized coordinate bounds [0.0, 1.0]
        for (d, &c) in coords.iter().enumerate() {
            if !c.is_finite() || c < 0.0 || c > 1.0 {
                return Err(DspError::invalid_parameter(
                    "coords",
                    format!(
                        "Coordinate {} at axis {} out of normalized bounds [0.0, 1.0]",
                        c, d
                    ),
                ));
            }
        }

        // Fast path for exact origin vertex [0.0, 0.0, ...]
        if coords.iter().all(|&c| c.abs() < 1e-12) {
            let origin_indices = vec![0; num_axes];
            if let Ok(frame) = self.get_frame(&origin_indices) {
                return Ok(frame.clone());
            }
        }

        // Map normalized coordinates into grid index floats
        let mut i_floor = Vec::with_capacity(num_axes);
        let mut i_ceil = Vec::with_capacity(num_axes);
        let mut weights_per_axis = Vec::with_capacity(num_axes);

        for (_d, (&c, &dim)) in coords.iter().zip(self.layout.axes().iter()).enumerate() {
            if dim <= 1 {
                i_floor.push(0);
                i_ceil.push(0);
                weights_per_axis.push(0.0);
            } else {
                let g = c * (dim - 1) as f64;
                let low = (g.floor() as usize).min(dim - 1);
                let high = (low + 1).min(dim - 1);
                let frac = g - low as f64;
                i_floor.push(low);
                i_ceil.push(high);
                weights_per_axis.push(frac);
            }
        }

        // 2^D vertex blending
        let num_vertices = 1usize << num_axes;
        let mut interp_timestamp = 0.0;
        let mut interp_f0 = 0.0;
        let mut partial_amps = Vec::new();
        let mut partial_freqs = Vec::new();
        let mut partial_phases = Vec::new();
        let mut residual_bands = Vec::new();
        let mut residual_ratio = 0.0;
        let mut weight_sum = 0.0;

        for v in 0..num_vertices {
            let mut v_coords = Vec::with_capacity(num_axes);
            let mut v_weight = 1.0;

            for d in 0..num_axes {
                let bit = (v >> d) & 1;
                let w = weights_per_axis[d];
                if bit == 1 {
                    v_coords.push(i_ceil[d]);
                    v_weight *= w;
                } else {
                    v_coords.push(i_floor[d]);
                    v_weight *= 1.0 - w;
                }
            }

            if v_weight <= 0.0 {
                continue;
            }

            if let Ok(frame) = self.get_frame(&v_coords) {
                weight_sum += v_weight;
                interp_timestamp += v_weight * frame.timestamp;
                interp_f0 += v_weight * frame.fundamental_freq;
                residual_ratio += v_weight * frame.residual.noise_ratio;

                // Initialize partial accumulators
                if partial_amps.is_empty() && !frame.partials.is_empty() {
                    partial_amps = vec![0.0; frame.partials.len()];
                    partial_freqs = vec![0.0; frame.partials.len()];
                    partial_phases = vec![0.0; frame.partials.len()];
                    for (k, p) in frame.partials.iter().enumerate() {
                        partial_freqs[k] = p.frequency;
                        partial_phases[k] = p.phase;
                    }
                }

                for (k, p) in frame.partials.iter().enumerate() {
                    if k < partial_amps.len() {
                        partial_amps[k] += v_weight * p.amplitude;
                    }
                }

                if residual_bands.is_empty() && !frame.residual.band_energies.is_empty() {
                    residual_bands = vec![0.0; frame.residual.band_energies.len()];
                }
                for (b, &e) in frame.residual.band_energies.iter().enumerate() {
                    if b < residual_bands.len() {
                        residual_bands[b] += v_weight * e;
                    }
                }
            }
        }

        if weight_sum > 1e-12 {
            interp_timestamp /= weight_sum;
            interp_f0 /= weight_sum;
            residual_ratio /= weight_sum;
            for a in &mut partial_amps {
                *a /= weight_sum;
            }
            for e in &mut residual_bands {
                *e /= weight_sum;
            }
        }

        let partials = partial_amps
            .into_iter()
            .enumerate()
            .map(|(idx, amp)| HarmonicPartial {
                harmonic_index: idx + 1,
                frequency: partial_freqs.get(idx).copied().unwrap_or(440.0 * (idx + 1) as f64),
                amplitude: amp,
                phase: partial_phases.get(idx).copied().unwrap_or(0.0),
            })
            .collect();

        Ok(SpectralFrame {
            timestamp: interp_timestamp,
            fundamental_freq: interp_f0,
            partials,
            residual: NoiseBandEnvelope {
                band_energies: residual_bands,
                noise_ratio: residual_ratio,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_unit_cube_interpolation() {
        let cube = CompressedHypercube::dummy_unit_cube();
        let frame = cube.interpolate(&[0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(frame.timestamp, 0.0);
        assert!((frame.fundamental_freq - 440.0).abs() < 1e-6);
        assert!(!frame.partials.is_empty());
    }

    #[test]
    fn test_interpolate_out_of_bounds() {
        let cube = CompressedHypercube::dummy_unit_cube();
        assert!(cube.interpolate(&[-0.1, 0.5, 0.5, 0.5, 0.5]).is_err());
        assert!(cube.interpolate(&[1.1, 0.5, 0.5, 0.5, 0.5]).is_err());
        assert!(cube.interpolate(&[0.5, 0.5]).is_err());
    }

    #[test]
    fn test_missing_frame_retrieval() {
        let cube = CompressedHypercube::dummy_unit_cube();
        assert!(cube.get_frame(&[99, 99, 99, 99, 99]).is_err());
    }
}
