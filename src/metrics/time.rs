//! Time-domain waveform fidelity metrics (MSE, RMSE, MAE, SNR, PSNR, Normalized Cross-Correlation).

use crate::error::{DspError, DspResult};

/// Time-domain fidelity metrics comparing a reference signal against a reconstructed signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeFidelityMetrics {
    /// Mean Squared Error: $\frac{1}{N} \sum_{n=0}^{N-1} (x[n] - \hat{x}[n])^2$.
    pub mse: f64,
    /// Root Mean Squared Error: $\sqrt{\text{MSE}}$.
    pub rmse: f64,
    /// Mean Absolute Error: $\frac{1}{N} \sum_{n=0}^{N-1} |x[n] - \hat{x}[n]|$.
    pub mae: f64,
    /// Signal-to-Noise Ratio in decibels (dB). Returns `+inf` for identical signals.
    pub snr_db: f64,
    /// Peak Signal-to-Noise Ratio in decibels (dB) referenced to $\max(x_{\text{peak}}, 1.0)$.
    pub psnr_db: f64,
    /// Zero-lag Normalized Cross-Correlation (cosine similarity) in $[-1.0, 1.0]$.
    pub correlation_r: f64,
    /// Alias for [`correlation_r`].
    pub ncc: f64,
    /// Centered Pearson correlation coefficient in $[-1.0, 1.0]$.
    pub pearson_r: f64,
}

/// Computes all time-domain fidelity metrics between reference and reconstructed audio signals.
///
/// # Errors
/// Returns [`DspError::EmptyInput`] if either slice is empty, or [`DspError::BufferLengthMismatch`]
/// if lengths do not match.
pub fn compute_time_metrics(reference: &[f64], reconstructed: &[f64]) -> DspResult<TimeFidelityMetrics> {
    if reference.is_empty() || reconstructed.is_empty() {
        return Err(DspError::EmptyInput);
    }
    if reference.len() != reconstructed.len() {
        return Err(DspError::BufferLengthMismatch {
            expected: reference.len(),
            actual: reconstructed.len(),
        });
    }

    let n = reference.len() as f64;
    let mut sum_sq_err = 0.0_f64;
    let mut sum_abs_err = 0.0_f64;
    let mut energy_ref = 0.0_f64;
    let mut energy_recon = 0.0_f64;
    let mut dot_product = 0.0_f64;
    let mut sum_ref = 0.0_f64;
    let mut sum_recon = 0.0_f64;
    let mut peak_ref = 0.0_f64;

    for (&r, &h) in reference.iter().zip(reconstructed.iter()) {
        let diff = r - h;
        sum_sq_err += diff * diff;
        sum_abs_err += diff.abs();
        energy_ref += r * r;
        energy_recon += h * h;
        dot_product += r * h;
        sum_ref += r;
        sum_recon += h;
        if r.abs() > peak_ref {
            peak_ref = r.abs();
        }
    }

    let mse = sum_sq_err / n;
    let rmse = mse.sqrt();
    let mae = sum_abs_err / n;

    // SNR calculation with strict edge-case handling
    let snr_db = if sum_sq_err == 0.0 {
        // Zero noise error -> perfect reconstruction (+inf)
        f64::INFINITY
    } else if energy_ref == 0.0 {
        // Silent reference with non-zero noise -> -inf
        f64::NEG_INFINITY
    } else {
        10.0 * (energy_ref / sum_sq_err).log10()
    };

    // PSNR calculation
    let full_scale_peak = peak_ref.max(1.0);
    let psnr_db = if rmse == 0.0 {
        f64::INFINITY
    } else {
        20.0 * (full_scale_peak / rmse).log10()
    };

    // Zero-lag normalized cross-correlation
    let norm_prod = (energy_ref * energy_recon).sqrt();
    let ncc = if norm_prod == 0.0 {
        if energy_ref == 0.0 && energy_recon == 0.0 {
            1.0 // Both pure silence -> perfect correlation
        } else {
            0.0
        }
    } else {
        (dot_product / norm_prod).clamp(-1.0, 1.0)
    };

    // Centered Pearson correlation coefficient
    let mean_ref = sum_ref / n;
    let mean_recon = sum_recon / n;
    let mut cov = 0.0_f64;
    let mut var_ref = 0.0_f64;
    let mut var_recon = 0.0_f64;
    for (&r, &h) in reference.iter().zip(reconstructed.iter()) {
        let dr = r - mean_ref;
        let dh = h - mean_recon;
        cov += dr * dh;
        var_ref += dr * dr;
        var_recon += dh * dh;
    }
    let pearson_denom = (var_ref * var_recon).sqrt();
    let pearson_r = if pearson_denom == 0.0 {
        if var_ref == 0.0 && var_recon == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        (cov / pearson_denom).clamp(-1.0, 1.0)
    };

    Ok(TimeFidelityMetrics {
        mse,
        rmse,
        mae,
        snr_db,
        psnr_db,
        correlation_r: ncc,
        ncc,
        pearson_r,
    })
}

/// Computes Mean Squared Error (MSE).
#[inline]
pub fn compute_mse(reference: &[f64], reconstructed: &[f64]) -> DspResult<f64> {
    compute_time_metrics(reference, reconstructed).map(|m| m.mse)
}

/// Computes Root Mean Squared Error (RMSE).
#[inline]
pub fn compute_rmse(reference: &[f64], reconstructed: &[f64]) -> DspResult<f64> {
    compute_time_metrics(reference, reconstructed).map(|m| m.rmse)
}

/// Computes Mean Absolute Error (MAE).
#[inline]
pub fn compute_mae(reference: &[f64], reconstructed: &[f64]) -> DspResult<f64> {
    compute_time_metrics(reference, reconstructed).map(|m| m.mae)
}

/// Computes Signal-to-Noise Ratio (SNR in dB).
#[inline]
pub fn compute_snr_db(reference: &[f64], reconstructed: &[f64]) -> DspResult<f64> {
    compute_time_metrics(reference, reconstructed).map(|m| m.snr_db)
}

/// Computes SNR in dB capped at `max_db` for display and reporting stability.
#[inline]
pub fn compute_snr_db_capped(reference: &[f64], reconstructed: &[f64], max_db: f64) -> DspResult<f64> {
    let snr = compute_snr_db(reference, reconstructed)?;
    if snr > max_db || snr.is_infinite() {
        Ok(max_db)
    } else {
        Ok(snr)
    }
}

/// Computes Peak Signal-to-Noise Ratio (PSNR in dB).
#[inline]
pub fn compute_psnr_db(reference: &[f64], reconstructed: &[f64]) -> DspResult<f64> {
    compute_time_metrics(reference, reconstructed).map(|m| m.psnr_db)
}

/// Computes PSNR in dB relative to an explicit peak full-scale amplitude.
pub fn compute_psnr_db_with_peak(
    reference: &[f64],
    reconstructed: &[f64],
    peak: f64,
) -> DspResult<f64> {
    let rmse = compute_rmse(reference, reconstructed)?;
    if rmse == 0.0 {
        Ok(f64::INFINITY)
    } else {
        Ok(20.0 * (peak.abs() / rmse).log10())
    }
}

/// Computes zero-lag Normalized Cross-Correlation (cosine similarity).
#[inline]
pub fn compute_ncc(reference: &[f64], reconstructed: &[f64]) -> DspResult<f64> {
    compute_time_metrics(reference, reconstructed).map(|m| m.ncc)
}

/// Alias for [`compute_ncc`].
#[inline]
pub fn compute_cross_correlation(reference: &[f64], reconstructed: &[f64]) -> DspResult<f64> {
    compute_ncc(reference, reconstructed)
}

/// Computes centered Pearson correlation coefficient.
#[inline]
pub fn compute_pearson_r(reference: &[f64], reconstructed: &[f64]) -> DspResult<f64> {
    compute_time_metrics(reference, reconstructed).map(|m| m.pearson_r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_metrics_identities() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let m = compute_time_metrics(&x, &x).unwrap();
        assert_eq!(m.mse, 0.0);
        assert_eq!(m.rmse, 0.0);
        assert!(m.snr_db.is_infinite());
        assert_eq!(m.correlation_r, 1.0);
    }
}
