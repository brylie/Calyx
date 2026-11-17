//! Quantitative fidelity, stability, and automated verification metrics.

pub mod report;
pub mod spectral;
pub mod stability;
pub mod time;

// Re-exports
pub use report::{
    generate_verification_report, MetricEvaluation, QualityTier, QualityTierConfig,
    VerificationReport,
};
pub use spectral::{
    compute_harmonic_peak_error, compute_log_spectral_distance, compute_magnitude_dft,
    compute_spectral_convergence, compute_spectral_metrics, compute_spectral_metrics_stft,
    evaluate_harmonic_extraction, HarmonicExtractionReport, SpectralFidelityMetrics, DELTA_FLOOR,
};
pub use stability::{check_numerical_stability, verify_numerical_stability, StabilityReport};
pub use time::{
    compute_cross_correlation, compute_mae, compute_mse, compute_ncc, compute_pearson_r,
    compute_psnr_db, compute_psnr_db_with_peak, compute_rmse, compute_snr_db,
    compute_snr_db_capped, compute_time_metrics, TimeFidelityMetrics,
};
