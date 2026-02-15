use pyo3::prelude::*;

mod complexity;
mod dynamics;
mod embedding;

/// Rust-accelerated primitives for signal analysis.
/// Both Prime and Manifold import from this package.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Complexity
    m.add_function(wrap_pyfunction!(complexity::hurst::hurst_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(complexity::entropy::permutation_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(complexity::entropy::sample_entropy, m)?)?;

    // Dynamics
    m.add_function(wrap_pyfunction!(dynamics::lyapunov::lyapunov_rosenstein, m)?)?;
    m.add_function(wrap_pyfunction!(dynamics::lyapunov::lyapunov_kantz, m)?)?;
    m.add_function(wrap_pyfunction!(dynamics::ftle::ftle_local_linearization, m)?)?;
    m.add_function(wrap_pyfunction!(dynamics::ftle::ftle_direct_perturbation, m)?)?;

    // Embedding
    m.add_function(wrap_pyfunction!(embedding::delay::optimal_delay, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::delay::time_delay_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(embedding::delay::optimal_dimension, m)?)?;

    Ok(())
}
