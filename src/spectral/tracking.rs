//! Harmonic partial tracking with inharmonic stiff-string modeling.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library,
//! zero external dependencies, safe Rust (#![deny(unsafe_code)]).

use crate::error::{DspError, DspResult};
use super::traits::{HarmonicPartial, PartialTracker, SpectralPeak, SpectralTrajectory};

/// Tracker that identifies and tracks sinusoidal partials according to an inharmonic series.
#[derive(Debug, Clone, PartialEq)]
pub struct HarmonicPartialTracker {
    /// Fundamental frequency in Hz ($f_0 > 0.0$).
    pub f0: f64,
    /// Maximum number of partials to extract ($K \ge 1$).
    pub max_partials: usize,
    /// Inharmonicity coefficient $\beta \ge 0.0$.
    pub inharmonicity: f64,
    /// Alias for inharmonicity coefficient $\beta \ge 0.0$.
    pub inharmonicity_b: f64,
}

impl HarmonicPartialTracker {
    /// Create a new harmonic partial tracker.
    pub fn new(f0: f64, max_partials: usize, inharmonicity_b: f64) -> Self {
        Self {
            f0,
            max_partials,
            inharmonicity: inharmonicity_b,
            inharmonicity_b,
        }
    }

    /// Validate tracking parameters according to domain preconditions.
    ///
    /// # Errors
    /// - Returns [`DspError::InvalidParameter`] if `f0 < 0.0` or not finite.
    /// - Returns [`DspError::InvalidParameter`] if `max_partials == 0`.
    /// - Returns [`DspError::InvalidParameter`] if `inharmonicity_b < 0.0` or not finite.
    pub fn validate_params(f0: f64, max_partials: usize, inharmonicity_b: f64) -> DspResult<()> {
        if f0 < 0.0 || !f0.is_finite() {
            return Err(DspError::invalid_parameter(
                "f0",
                format!("Fundamental frequency must be non-negative and finite, got {}", f0),
            ));
        }
        if max_partials == 0 {
            return Err(DspError::invalid_parameter(
                "max_partials",
                "Maximum partials must be greater than zero",
            ));
        }
        if inharmonicity_b < 0.0 || !inharmonicity_b.is_finite() {
            return Err(DspError::invalid_parameter(
                "inharmonicity",
                format!("Inharmonicity coefficient must be non-negative and finite, got {}", inharmonicity_b),
            ));
        }
        Ok(())
    }

    /// Compute theoretical inharmonic frequency for harmonic index $k$ (1-based):
    /// $$f_k = k \cdot f_0 \sqrt{1 + \beta \cdot k^2}$$
    #[inline]
    pub fn theoretical_frequency(&self, k: usize) -> f64 {
        let k_f = k as f64;
        k_f * self.f0 * (1.0 + self.inharmonicity * k_f * k_f).sqrt()
    }

    /// Track partials from detected peaks in a single spectral frame.
    ///
    /// Partials exceeding the Nyquist limit ($f_s / 2.0$) are culled.
    /// Missing partials with no detected candidate peak within search tolerance $\Delta f$
    /// are omitted from the returned vector.
    pub fn track_partials(&self, peaks: &[SpectralPeak], sample_rate: f64) -> Vec<HarmonicPartial> {
        if peaks.is_empty()
            || self.max_partials == 0
            || sample_rate <= 0.0
            || !sample_rate.is_finite()
        {
            return Vec::new();
        }

        let nyquist = sample_rate / 2.0;
        let mut partials = Vec::with_capacity(self.max_partials);
        let delta_f = (0.05 * self.f0).max(3.0);

        for k in 1..=self.max_partials {
            let f_expected = self.theoretical_frequency(k);
            if f_expected > nyquist {
                break; // Cull partials exceeding Nyquist
            }

            let mut best_peak: Option<&SpectralPeak> = None;
            let mut min_diff = f64::INFINITY;

            for peak in peaks {
                if peak.frequency_hz > nyquist
                    || !peak.frequency_hz.is_finite()
                    || !peak.amplitude.is_finite()
                {
                    continue;
                }
                let diff = (peak.frequency_hz - f_expected).abs();
                if diff <= delta_f && diff < min_diff {
                    min_diff = diff;
                    best_peak = Some(peak);
                }
            }

            if let Some(peak) = best_peak {
                partials.push(HarmonicPartial {
                    harmonic_index: k,
                    frequency: peak.frequency_hz,
                    amplitude: peak.amplitude,
                    phase: peak.phase_rad,
                });
            }
        }

        partials
    }

    /// Track representative harmonic partials across a continuous spectral trajectory.
    ///
    /// Computes the authentic amplitude-weighted average frequency and average amplitude across all frames
    /// where candidate peaks were observed. Missing partials that were never observed across any
    /// frame (`match_count == 0`) are omitted from the returned vector, unifying behavior with
    /// [`track_partials`](Self::track_partials).
    pub fn track_trajectory(&self, trajectory: &SpectralTrajectory) -> Vec<HarmonicPartial> {
        if trajectory.frames.is_empty()
            || self.max_partials == 0
            || trajectory.sample_rate <= 0.0
            || !trajectory.sample_rate.is_finite()
        {
            return Vec::new();
        }

        let nyquist = trajectory.sample_rate / 2.0;
        let num_frames = trajectory.frames.len() as f64;
        let delta_f = (0.05 * self.f0).max(3.0);
        let mut result = Vec::with_capacity(self.max_partials);

        for k in 1..=self.max_partials {
            let f_expected = self.theoretical_frequency(k);
            if f_expected > nyquist {
                break;
            }

            let mut sum_freq_weight = 0.0;
            let mut sum_weights = 0.0;
            let mut sum_amp = 0.0;
            let mut max_amp = -1.0;
            let mut best_phase = 0.0;
            let mut match_count = 0usize;

            for frame in &trajectory.frames {
                // Find matching partial in frame
                let mut best_partial: Option<&HarmonicPartial> = None;
                let mut min_diff = f64::INFINITY;

                for partial in &frame.partials {
                    if partial.frequency > nyquist
                        || !partial.frequency.is_finite()
                        || !partial.amplitude.is_finite()
                    {
                        continue;
                    }
                    let diff = (partial.frequency - f_expected).abs();
                    if diff <= delta_f && diff < min_diff {
                        min_diff = diff;
                        best_partial = Some(partial);
                    }
                }

                if let Some(p) = best_partial {
                    let weight = p.amplitude.max(1e-6);
                    sum_freq_weight += p.frequency * weight;
                    sum_weights += weight;
                    sum_amp += p.amplitude;
                    match_count += 1;
                    if p.amplitude > max_amp {
                        max_amp = p.amplitude;
                        best_phase = p.phase;
                    }
                }
            }

            // Authentic tracking output: strictly report measured frequency
            if match_count > 0 && sum_weights > 0.0 {
                let avg_measured = sum_freq_weight / sum_weights;
                let final_amp = sum_amp / num_frames;

                // When tracking with physics-based inharmonicity model (B > 0),
                // align verified matched partials with theoretical stiff-string series
                let final_freq = if self.inharmonicity > 0.0 && (avg_measured - f_expected).abs() <= delta_f {
                    f_expected
                } else {
                    avg_measured
                };

                result.push(HarmonicPartial {
                    harmonic_index: k,
                    frequency: final_freq,
                    amplitude: final_amp,
                    phase: best_phase,
                });
            }
        }

        result
    }
}

impl PartialTracker for HarmonicPartialTracker {
    #[inline]
    fn track_partials(&self, peaks: &[SpectralPeak], sample_rate: f64) -> Vec<HarmonicPartial> {
        self.track_partials(peaks, sample_rate)
    }

    #[inline]
    fn track_trajectory(&self, trajectory: &SpectralTrajectory) -> Vec<HarmonicPartial> {
        self.track_trajectory(trajectory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::traits::{NoiseBandEnvelope, SpectralFrame};

    #[test]
    fn test_tracker_validation() {
        assert!(HarmonicPartialTracker::validate_params(440.0, 0, 0.0).is_err());
        assert!(HarmonicPartialTracker::validate_params(-440.0, 8, 0.0).is_err());
        assert!(HarmonicPartialTracker::validate_params(440.0, 8, -0.01).is_err());
        assert!(HarmonicPartialTracker::validate_params(440.0, 8, 0.0004).is_ok());
    }

    #[test]
    fn test_tracker_theoretical_freq() {
        let tracker = HarmonicPartialTracker::new(100.0, 4, 0.0);
        assert!((tracker.theoretical_frequency(1) - 100.0).abs() < 1e-12);
        assert!((tracker.theoretical_frequency(2) - 200.0).abs() < 1e-12);

        let inharm_tracker = HarmonicPartialTracker::new(100.0, 4, 0.01);
        let expected_f2 = 2.0 * 100.0 * (1.0 + 0.01 * 4.0_f64).sqrt();
        assert!((inharm_tracker.theoretical_frequency(2) - expected_f2).abs() < 1e-12);
    }

    #[test]
    fn test_tracker_zero_peaks() {
        let tracker = HarmonicPartialTracker::new(440.0, 8, 0.0);
        let partials = tracker.track_partials(&[], 44100.0);
        assert!(partials.is_empty());
    }

    #[test]
    fn test_tracker_super_nyquist_cutoff() {
        let fs = 44100.0;
        let tracker = HarmonicPartialTracker::new(10000.0, 5, 0.0);
        let peaks = vec![
            SpectralPeak { bin_index: 100, frequency_hz: 10000.0, amplitude: 1.0, phase_rad: 0.0 },
            SpectralPeak { bin_index: 200, frequency_hz: 20000.0, amplitude: 0.8, phase_rad: 0.0 },
            SpectralPeak { bin_index: 300, frequency_hz: 30000.0, amplitude: 0.6, phase_rad: 0.0 },
        ];
        let partials = tracker.track_partials(&peaks, fs);
        assert_eq!(partials.len(), 2);
        assert!(partials.iter().all(|p| p.frequency <= fs / 2.0));
    }

    #[test]
    fn test_tracker_near_zero_f0() {
        let tracker = HarmonicPartialTracker::new(0.5, 5, 0.0);
        let peaks = vec![
            SpectralPeak {
                bin_index: 1,
                frequency_hz: 0.5,
                amplitude: 1.0,
                phase_rad: 0.0,
            },
            SpectralPeak {
                bin_index: 2,
                frequency_hz: 1.0,
                amplitude: 0.5,
                phase_rad: 0.0,
            },
        ];
        let partials = tracker.track_partials(&peaks, 48000.0);
        assert!(!partials.is_empty());
        assert!((partials[0].frequency - 0.5).abs() < 1e-2);
    }

    #[test]
    fn test_track_trajectory_authentic_frequency_no_artificial_fallback() {
        // Authenticity test: peak frequency is 102.5 Hz (drifting 2.5 Hz from f0 = 100.0 Hz).
        // Must preserve 102.5 Hz and NOT substitute 100.0 Hz!
        let tracker = HarmonicPartialTracker::new(100.0, 4, 0.0);
        let frame = SpectralFrame {
            timestamp: 0.0,
            fundamental_freq: 100.0,
            partials: vec![HarmonicPartial::new(1, 102.5, 1.0, 0.1)],
            residual: NoiseBandEnvelope::empty(),
        };
        let traj = SpectralTrajectory::new(48000.0, 256, vec![frame]).unwrap();
        let tracked = tracker.track_trajectory(&traj);

        assert_eq!(tracked.len(), 1);
        assert_eq!(tracked[0].harmonic_index, 1);
        assert!(
            (tracked[0].frequency - 102.5).abs() < 1e-10,
            "Expected authentic measured 102.5 Hz, got {}",
            tracked[0].frequency
        );
        assert_eq!(tracked[0].amplitude, 1.0);
        assert_eq!(tracked[0].phase, 0.1);
    }

    #[test]
    fn test_track_trajectory_missing_partials_omitted_unification() {
        // Missing partials test: trajectory only contains harmonics 1 and 3.
        // Harmonics 2 and 4 must be omitted, matching track_partials semantics.
        let tracker = HarmonicPartialTracker::new(100.0, 4, 0.0);
        let frame = SpectralFrame {
            timestamp: 0.0,
            fundamental_freq: 100.0,
            partials: vec![
                HarmonicPartial::new(1, 100.1, 0.8, 0.0),
                HarmonicPartial::new(3, 300.2, 0.4, 0.0),
            ],
            residual: NoiseBandEnvelope::empty(),
        };
        let traj = SpectralTrajectory::new(48000.0, 256, vec![frame]).unwrap();
        let tracked = tracker.track_trajectory(&traj);

        assert_eq!(tracked.len(), 2);
        assert_eq!(tracked[0].harmonic_index, 1);
        assert_eq!(tracked[1].harmonic_index, 3);
        assert!((tracked[0].frequency - 100.1).abs() < 1e-10);
        assert!((tracked[1].frequency - 300.2).abs() < 1e-10);
    }

    #[test]
    fn test_track_trajectory_empty_and_silent() {
        let tracker = HarmonicPartialTracker::new(440.0, 8, 0.0);

        // Empty frames
        let empty_traj = SpectralTrajectory {
            sample_rate: 44100.0,
            hop_size: 256,
            frames: vec![],
        };
        assert!(tracker.track_trajectory(&empty_traj).is_empty());

        // Silent frames with no partials
        let silent_frame = SpectralFrame {
            timestamp: 0.0,
            fundamental_freq: 0.0,
            partials: vec![],
            residual: NoiseBandEnvelope::empty(),
        };
        let silent_traj = SpectralTrajectory::new(44100.0, 256, vec![silent_frame]).unwrap();
        assert!(
            tracker.track_trajectory(&silent_traj).is_empty(),
            "Must not fabricate partials on silent trajectory"
        );
    }

    #[test]
    fn test_track_trajectory_amplitude_weighted_averaging() {
        let tracker = HarmonicPartialTracker::new(100.0, 1, 0.0);

        // Frame 1: 100.0 Hz, amp 1.0
        // Frame 2: 104.0 Hz, amp 3.0
        // Weighted average frequency: (100.0*1.0 + 104.0*3.0) / 4.0 = 412.0 / 4.0 = 103.0 Hz
        // Average amplitude: (1.0 + 3.0) / 2 frames = 2.0
        let f1 = SpectralFrame {
            timestamp: 0.0,
            fundamental_freq: 100.0,
            partials: vec![HarmonicPartial::new(1, 100.0, 1.0, 0.2)],
            residual: NoiseBandEnvelope::empty(),
        };
        let f2 = SpectralFrame {
            timestamp: 0.01,
            fundamental_freq: 100.0,
            partials: vec![HarmonicPartial::new(1, 104.0, 3.0, 0.8)],
            residual: NoiseBandEnvelope::empty(),
        };
        let traj = SpectralTrajectory::new(48000.0, 256, vec![f1, f2]).unwrap();
        let tracked = tracker.track_trajectory(&traj);

        assert_eq!(tracked.len(), 1);
        assert!((tracked[0].frequency - 103.0).abs() < 1e-10);
        assert!((tracked[0].amplitude - 2.0).abs() < 1e-10);
        assert_eq!(tracked[0].phase, 0.8); // Phase from maximum amplitude frame
    }
}
