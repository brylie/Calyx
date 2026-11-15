//! Mathematical foundations and data containers for the Calyx DSP framework.

pub mod complex;
pub mod matrix;
pub mod tensor;
pub mod window;

pub use complex::Complex;
pub use matrix::Matrix;
pub use tensor::Tensor;
pub use window::{
    coherent_gain, energy_gain, equivalent_noise_bandwidth, generate_window,
    generate_window_symmetric, verify_cola, WindowType,
};
