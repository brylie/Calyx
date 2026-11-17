//! CLI runner for the Calyx DSP empirical research & benchmark suite.
//!
//! Executes all 6 empirical experiment sweeps and prints the formatted Markdown report.

#![deny(unsafe_code)]

use calyx_dsp::experiments::EmpiricalStudySuite;

fn main() {
    match EmpiricalStudySuite::run_all() {
        Ok(suite) => {
            println!("{}", suite.to_markdown());
        }
        Err(err) => {
            eprintln!("Empirical study execution failed: {}", err);
            std::process::exit(1);
        }
    }
}
