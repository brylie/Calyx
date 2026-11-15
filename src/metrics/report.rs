//! Quality tiers, verification reporting, and formatted table generators.

use crate::error::{DspError, DspResult};
use crate::metrics::spectral::{compute_spectral_metrics, SpectralFidelityMetrics};
use crate::metrics::stability::{check_numerical_stability, StabilityReport};
use crate::metrics::time::{compute_time_metrics, TimeFidelityMetrics};

/// Configurable quality tiers defined in Calyx specification Section 4.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QualityTier {
    /// Strict/Exact verification tier (lossless, unit tests).
    Strict,
    /// Reference tier (solo exposure, mastering, near-indistinguishable).
    #[default]
    Reference,
    /// Production tier (music production, mixing, perceptually transparent).
    Production,
    /// Preview tier (auditioning, browsing, high compression).
    Preview,
}

/// Configuration specifying DSP rank and fidelity acceptance thresholds for a quality tier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityTierConfig {
    /// Quality tier classification.
    pub tier: QualityTier,
    /// Minimum acceptable Signal-to-Noise Ratio (dB).
    pub min_snr_db: f64,
    /// Maximum acceptable Log-Spectral Distance (dB).
    pub max_lsd_db: f64,
    /// Decomposition rank parameter.
    pub rank: usize,
    /// Number of subband noise residual filterbank bands.
    pub num_noise_bands: usize,
}

impl QualityTierConfig {
    /// Pre-configured Reference Quality Tier (mastering / high fidelity).
    pub fn reference() -> Self {
        Self {
            tier: QualityTier::Reference,
            min_snr_db: 35.0,
            max_lsd_db: 0.8,
            rank: 8,
            num_noise_bands: 16,
        }
    }

    /// Pre-configured Production Quality Tier (standard production fidelity).
    pub fn production() -> Self {
        Self {
            tier: QualityTier::Production,
            min_snr_db: 28.0,
            max_lsd_db: 1.5,
            rank: 4,
            num_noise_bands: 8,
        }
    }

    /// Pre-configured Preview Quality Tier (low latency / high compression).
    pub fn preview() -> Self {
        Self {
            tier: QualityTier::Preview,
            min_snr_db: 18.0,
            max_lsd_db: 2.8,
            rank: 2,
            num_noise_bands: 4,
        }
    }

    /// Constructs a custom quality tier configuration with custom rank and noise bands.
    ///
    /// # Errors
    /// Returns [`DspError::InvalidParameter`] if `rank == 0` or `num_noise_bands == 0`.
    pub fn custom(rank: usize, num_noise_bands: usize) -> DspResult<Self> {
        if rank == 0 {
            return Err(DspError::invalid_parameter(
                "rank",
                "rank must be greater than zero",
            ));
        }
        if num_noise_bands == 0 {
            return Err(DspError::invalid_parameter(
                "num_noise_bands",
                "num_noise_bands must be greater than zero",
            ));
        }

        Ok(Self {
            tier: QualityTier::Production,
            min_snr_db: 25.0,
            max_lsd_db: 1.8,
            rank,
            num_noise_bands,
        })
    }

    /// Validates measured fidelity metrics against tier thresholds.
    ///
    /// # Errors
    /// Returns [`DspError::NumericalInstability`] if any metric fails tier requirements.
    pub fn validate_fidelity(
        &self,
        time_metrics: &TimeFidelityMetrics,
        spec_metrics: &SpectralFidelityMetrics,
    ) -> DspResult<()> {
        if time_metrics.snr_db < self.min_snr_db {
            return Err(DspError::numerical_instability(format!(
                "SNR {:.2} dB below quality tier minimum {:.2} dB",
                time_metrics.snr_db, self.min_snr_db
            )));
        }
        if spec_metrics.log_spectral_distance_db > self.max_lsd_db {
            return Err(DspError::numerical_instability(format!(
                "LSD {:.2} dB exceeds quality tier maximum {:.2} dB",
                spec_metrics.log_spectral_distance_db, self.max_lsd_db
            )));
        }
        Ok(())
    }
}

/// Evaluation record for a single verification metric.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricEvaluation {
    /// Metric human-readable identifier.
    pub name: &'static str,
    /// Formatted observed value string.
    pub value_str: String,
    /// Formatted acceptance threshold string.
    pub threshold_str: String,
    /// Unit of measurement (e.g. "dB", "").
    pub unit: &'static str,
    /// Whether the observed value satisfies the threshold.
    pub passed: bool,
}

/// Unified verification report aggregating time, spectral, and stability metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct VerificationReport {
    /// Test or benchmark description.
    pub test_name: String,
    /// Processed sample count.
    pub sample_count: usize,
    /// Audio sampling rate in Hertz.
    pub sample_rate: f64,
    /// Attached quality tier configuration.
    pub tier_config: QualityTierConfig,
    /// Evaluated time-domain metrics.
    pub time_metrics: TimeFidelityMetrics,
    /// Evaluated spectral-domain metrics.
    pub spectral_metrics: SpectralFidelityMetrics,
    /// Evaluated stability invariants report.
    pub stability_report: StabilityReport,
    /// Individual metric pass/fail evaluations.
    pub evaluations: Vec<MetricEvaluation>,
    /// Global pass/fail status across all evaluations.
    pub overall_passed: bool,
}

impl VerificationReport {
    /// Formats the report as a Unicode box-drawing table.
    pub fn to_table_string(&self) -> String {
        let mut out = String::new();
        out.push_str("┌────────────────────────────────────────────────────────────────────────┐\n");
        out.push_str(&format!("│ CALYX DSP VERIFICATION REPORT: {:<40}│\n", self.test_name));
        out.push_str("├──────────────────────┬─────────────┬──────────────┬──────────┬────────┤\n");
        out.push_str("│ Metric               │ Value       │ Threshold    │ Unit     │ Status │\n");
        out.push_str("├──────────────────────┼─────────────┼──────────────┼──────────┼────────┤\n");

        for eval in &self.evaluations {
            let status = if eval.passed { "PASS" } else { "FAIL" };
            out.push_str(&format!(
                "│ {:<20} │ {:<11} │ {:<12} │ {:<8} │ {:<6} │\n",
                eval.name, eval.value_str, eval.threshold_str, eval.unit, status
            ));
        }

        out.push_str("├──────────────────────┴─────────────┴──────────────┴──────────┴────────┤\n");
        let passed_count = self.evaluations.iter().filter(|e| e.passed).count();
        let total_count = self.evaluations.len();
        let global_status = if self.overall_passed { "PASS" } else { "FAIL" };
        out.push_str(&format!(
            "│ Quality Tier: {:<10} | Overall Result: {:<4} ({}/{} Passed)           │\n",
            format!("{:?}", self.tier_config.tier),
            global_status,
            passed_count,
            total_count
        ));
        out.push_str("└────────────────────────────────────────────────────────────────────────┘\n");
        out
    }

    /// Formats the report as an ASCII table fallback.
    pub fn to_ascii_string(&self) -> String {
        let mut out = String::new();
        out.push_str("+------------------------------------------------------------------------+\n");
        out.push_str(&format!("| CALYX DSP VERIFICATION REPORT: {:<40}|\n", self.test_name));
        out.push_str("+----------------------+-------------+--------------+----------+--------+\n");
        out.push_str("| Metric               | Value       | Threshold    | Unit     | Status |\n");
        out.push_str("+----------------------+-------------+--------------+----------+--------+\n");

        for eval in &self.evaluations {
            let status = if eval.passed { "PASS" } else { "FAIL" };
            out.push_str(&format!(
                "| {:<20} | {:<11} | {:<12} | {:<8} | {:<6} |\n",
                eval.name, eval.value_str, eval.threshold_str, eval.unit, status
            ));
        }

        out.push_str("+------------------------------------------------------------------------+\n");
        out
    }

    /// Formats the report as a Markdown table.
    pub fn to_markdown_string(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("### Calyx Verification Report: {}\n\n", self.test_name));
        out.push_str("| Metric | Value | Threshold | Unit | Status |\n");
        out.push_str("|---|---|---|---|---|\n");
        for eval in &self.evaluations {
            let status = if eval.passed { "PASS" } else { "**FAIL**" };
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                eval.name, eval.value_str, eval.threshold_str, eval.unit, status
            ));
        }
        out
    }

    /// Returns the list of failed metric descriptions.
    pub fn failures(&self) -> Vec<String> {
        self.evaluations
            .iter()
            .filter(|e| !e.passed)
            .map(|e| format!("{}: {} did not meet {}", e.name, e.value_str, e.threshold_str))
            .collect()
    }
}

/// Generates a comprehensive verification report comparing reference and reconstructed signals.
pub fn generate_verification_report(
    test_name: impl Into<String>,
    reference: &[f64],
    reconstructed: &[f64],
    sample_rate: f64,
    tier_config: QualityTierConfig,
) -> DspResult<VerificationReport> {
    let time_metrics = compute_time_metrics(reference, reconstructed)?;
    let spectral_metrics = compute_spectral_metrics(reference, reconstructed)?;
    let stability_report = check_numerical_stability(reference, reconstructed);

    let mut evaluations = Vec::new();

    // SNR check
    let snr_passed = time_metrics.snr_db >= tier_config.min_snr_db;
    evaluations.push(MetricEvaluation {
        name: "SNR",
        value_str: if time_metrics.snr_db.is_infinite() {
            "+inf".to_string()
        } else {
            format!("{:.2}", time_metrics.snr_db)
        },
        threshold_str: format!(">= {:.2}", tier_config.min_snr_db),
        unit: "dB",
        passed: snr_passed,
    });

    // LSD check
    let lsd_passed = spectral_metrics.log_spectral_distance_db <= tier_config.max_lsd_db;
    evaluations.push(MetricEvaluation {
        name: "Log-Spectral Dist",
        value_str: format!("{:.4}", spectral_metrics.log_spectral_distance_db),
        threshold_str: format!("<= {:.2}", tier_config.max_lsd_db),
        unit: "dB",
        passed: lsd_passed,
    });

    // MSE check
    evaluations.push(MetricEvaluation {
        name: "MSE",
        value_str: format!("{:.3e}", time_metrics.mse),
        threshold_str: "<= 1.0e-2".to_string(),
        unit: "",
        passed: time_metrics.mse <= 1.0e-2,
    });

    // Correlation check
    evaluations.push(MetricEvaluation {
        name: "Normalized Corr",
        value_str: format!("{:.6}", time_metrics.correlation_r),
        threshold_str: ">= 0.95".to_string(),
        unit: "",
        passed: time_metrics.correlation_r >= 0.95,
    });

    // Stability check
    evaluations.push(MetricEvaluation {
        name: "Numerical Stability",
        value_str: if stability_report.is_clean {
            "Clean".to_string()
        } else {
            "Violations".to_string()
        },
        threshold_str: "Clean".to_string(),
        unit: "",
        passed: stability_report.is_clean,
    });

    let overall_passed = evaluations.iter().all(|e| e.passed);

    Ok(VerificationReport {
        test_name: test_name.into(),
        sample_count: reference.len(),
        sample_rate,
        tier_config,
        time_metrics,
        spectral_metrics,
        stability_report,
        evaluations,
        overall_passed,
    })
}
