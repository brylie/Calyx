//! Phase models: harmonic locking, continuous vocoder integration, and dispersion.
//!
//! Complies strictly with Demo Integrity Mode: 100% pure standard library safe Rust.

use std::f64::consts::PI;
use crate::synthetic::noise::Pcg32;
use crate::synthesis::traits::PhaseModel;

/// Deterministic phase dispersion model based on pseudo-random hash seeding.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseDispersionModel {
    seed: u64,
}

impl PhaseDispersionModel {
    /// Constructs a new phase dispersion model with a deterministic seed.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Disperses the phase angle deterministically for harmonic index `k`.
    ///
    /// Bit-for-bit identical across runs for the same seed and harmonic index.
    pub fn disperse_phase(&self, base_phase: f64, harmonic_index: usize) -> f64 {
        let mut rng = Pcg32::new(
            self.seed.wrapping_add((harmonic_index as u64).wrapping_mul(31337)),
            101,
        );
        let shift = rng.next_f64() * 2.0 * PI - PI;
        let mut dispersed = (base_phase + shift) % (2.0 * PI);
        if dispersed < 0.0 {
            dispersed += 2.0 * PI;
        }
        dispersed
    }
}

impl PhaseModel for PhaseDispersionModel {
    fn compute_phase(&self, phase: f64, harmonic_index: usize, _fundamental_phase: f64) -> f64 {
        self.disperse_phase(phase, harmonic_index)
    }
}

/// Computes harmonic locked phase: $\phi_k = (k \cdot \phi_0) \pmod{2\pi}$.
#[inline]
pub fn lock_harmonic_phase(fundamental_phase: f64, harmonic_index: usize) -> f64 {
    let mut phase = ((harmonic_index as f64) * fundamental_phase) % (2.0 * PI);
    if phase < 0.0 {
        phase += 2.0 * PI;
    }
    phase
}

/// Wraps an arbitrary phase angle into $[0.0, 2\pi)$.
#[inline]
pub fn wrap_phase(mut phase: f64) -> f64 {
    phase %= 2.0 * PI;
    if phase < 0.0 {
        phase += 2.0 * PI;
    }
    phase
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_dispersion_consistency() {
        let d1 = PhaseDispersionModel::new(42);
        let d2 = PhaseDispersionModel::new(42);
        assert_eq!(d1.disperse_phase(0.5, 1), d2.disperse_phase(0.5, 1));
        assert_eq!(d1.disperse_phase(0.5, 3), d2.disperse_phase(0.5, 3));
    }

    #[test]
    fn test_lock_harmonic_phase() {
        let phi = lock_harmonic_phase(0.5, 4);
        assert!((phi - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_wrap_phase() {
        assert!((wrap_phase(2.0 * PI + 0.1) - 0.1).abs() < 1e-12);
        assert!((wrap_phase(-0.1) - (2.0 * PI - 0.1)).abs() < 1e-12);
    }
}
