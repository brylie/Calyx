//! Numerical stability and DSP invariant verification checks.

use crate::error::{DspError, DspResult};

/// Detailed diagnostics of signal numerical stability and invariant compliance.
#[derive(Debug, Clone, PartialEq)]
pub struct StabilityReport {
    /// True if signal meets all stability tolerances with zero non-finite values and zero clipping.
    pub is_clean: bool,
    /// Maximum absolute sample amplitude observed in the reconstructed signal.
    pub peak_amplitude: f64,
    /// Energy conservation ratio $E_{\text{recon}} / E_{\text{orig}}$.
    pub energy_ratio: f64,
    /// Absolute DC offset deviation $(\mu_{\text{recon}} - \mu_{\text{orig}})$.
    pub dc_offset: f64,
    /// Whether any NaN values were detected.
    pub has_nan: bool,
    /// Whether any Infinite values were detected.
    pub has_inf: bool,
    /// Total count of NaN samples detected.
    pub nan_count: usize,
    /// Total count of Infinite samples detected.
    pub inf_count: usize,
    /// Total count of subnormal (denormal) samples detected.
    pub subnormal_count: usize,
    /// Total count of clipped samples ($|s| > 1.0001$).
    pub clipping_count: usize,
    /// Specific invariant violations if `is_clean` is false.
    pub violations: Vec<String>,
}

/// Verifies that an audio signal satisfies essential numerical invariants.
///
/// # Invariants Checked:
/// - Zero NaN values.
/// - Zero Infinite ($+\infty, -\infty$) values.
/// - Peak amplitude bounded within standard headroom ($|s[n]| \le 1.0001$).
///
/// # Errors
/// Returns [`DspError::NumericalInstability`] detailing the first invariant violation encountered.
pub fn verify_numerical_stability(signal: &[f64]) -> DspResult<()> {
    if signal.is_empty() {
        return Err(DspError::EmptyInput);
    }
    let mut clip_count = 0;
    for (i, &s) in signal.iter().enumerate() {
        if s.is_nan() {
            return Err(DspError::numerical_instability(format!(
                "NaN detected at sample index {}",
                i
            )));
        }
        if s.is_infinite() {
            return Err(DspError::numerical_instability(format!(
                "Infinity detected at sample index {}",
                i
            )));
        }
        if s.abs() > 1.0001 {
            clip_count += 1;
        }
    }
    if clip_count > 0 {
        return Err(DspError::numerical_instability(format!(
            "Peak clipping detected: {} samples exceed 1.0001",
            clip_count
        )));
    }
    Ok(())
}

/// Evaluates comparative numerical stability between reference and reconstructed audio signals.
pub fn check_numerical_stability(reference: &[f64], reconstructed: &[f64]) -> StabilityReport {
    let mut nan_count = 0_usize;
    let mut inf_count = 0_usize;
    let mut subnormal_count = 0_usize;
    let mut clipping_count = 0_usize;
    let mut peak_amplitude = 0.0_f64;
    let mut sum_recon = 0.0_f64;
    let mut energy_recon = 0.0_f64;
    let mut violations = Vec::new();

    for &s in reconstructed {
        if s.is_nan() {
            nan_count += 1;
        } else if s.is_infinite() {
            inf_count += 1;
        } else {
            if s.is_subnormal() {
                subnormal_count += 1;
            }
            if s.abs() > peak_amplitude {
                peak_amplitude = s.abs();
            }
            if s.abs() > 1.0001 {
                clipping_count += 1;
            }
            sum_recon += s;
            energy_recon += s * s;
        }
    }

    let mut sum_ref = 0.0_f64;
    let mut energy_ref = 0.0_f64;
    for &r in reference {
        if !r.is_nan() && !r.is_infinite() {
            sum_ref += r;
            energy_ref += r * r;
        }
    }

    let n_recon = reconstructed.len().max(1) as f64;
    let n_ref = reference.len().max(1) as f64;
    let mean_recon = sum_recon / n_recon;
    let mean_ref = sum_ref / n_ref;
    let dc_offset = (mean_recon - mean_ref).abs();

    let energy_ratio = if energy_ref == 0.0 {
        if energy_recon == 0.0 {
            1.0
        } else {
            f64::INFINITY
        }
    } else {
        energy_recon / energy_ref
    };

    if nan_count > 0 {
        violations.push(format!("Detected {} NaN samples", nan_count));
    }
    if inf_count > 0 {
        violations.push(format!("Detected {} Infinite samples", inf_count));
    }
    if clipping_count > 0 {
        violations.push(format!("Detected {} clipped samples (|s| > 1.0001)", clipping_count));
    }
    if dc_offset > 0.01 {
        violations.push(format!("Severe DC offset detected: {:.6}", dc_offset));
    }
    if energy_ref > 0.0 && (energy_ratio < 0.85 || energy_ratio > 1.15) {
        violations.push(format!("Energy conservation ratio deviation: {:.4}", energy_ratio));
    }

    let is_clean = violations.is_empty();

    StabilityReport {
        is_clean,
        peak_amplitude,
        energy_ratio,
        dc_offset,
        has_nan: nan_count > 0,
        has_inf: inf_count > 0,
        nan_count,
        inf_count,
        subnormal_count,
        clipping_count,
        violations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stability_clean_signal() {
        let clean = vec![0.0, 0.5, -0.5, 0.25];
        assert!(verify_numerical_stability(&clean).is_ok());
        let report = check_numerical_stability(&clean, &clean);
        assert!(report.is_clean);
        assert_eq!(report.energy_ratio, 1.0);
    }
}
