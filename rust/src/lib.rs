//! Python extension module `faser._core`: a thin pyo3 wrapper around the
//! `faser-core` simulator (see ./core). The Python side derives the
//! integration quantities itself (`PSFConfig` properties) and fills a
//! `PsfConfig`; `generate_psf_json` additionally accepts the user-facing
//! `psf_config.json` format and derives everything in Rust, which is the
//! code path the WebAssembly playground uses.

use numpy::{PyArray, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub use faser_core::{Mode, Normalize, Params, PsfConfig, Slm, ThetaSampling};

/// Returns the PSF intensity as a float64 array of shape (Nz, Ny, Nx).
#[pyfunction]
fn generate_psf<'py>(py: Python<'py>, config: PsfConfig) -> PyResult<Bound<'py, PyArray3<f64>>> {
    config.validate().map_err(PyValueError::new_err)?;
    // Release the GIL: the computation is pure Rust and uses all cores via rayon.
    let result = py.detach(move || faser_core::psf_volume(config));
    Ok(PyArray::from_owned_array(py, result))
}

/// Same as `generate_psf`, but from the user-facing parameters serialized as
/// JSON (the `psf_config.json` format); all derived quantities are computed
/// in Rust. Mainly exists to test that derivation against the Python model.
/// With `scalar=True` the scalar Debye integral is used instead (same pupil,
/// phase mask, aberrations and coverslip/depth phase, no polarization).
#[pyfunction]
#[pyo3(signature = (json, scalar = false))]
fn generate_psf_json<'py>(py: Python<'py>, json: &str, scalar: bool) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let params: Params = serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let config = params.to_config().map_err(PyValueError::new_err)?;
    let result = py.detach(move || faser_core::psf_volume_model(config, scalar));
    Ok(PyArray::from_owned_array(py, result))
}

/// Native backend of faser, compiled to `faser._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PsfConfig>()?;
    m.add_class::<Mode>()?;
    m.add_class::<Normalize>()?;
    m.add_class::<Slm>()?;
    m.add_class::<ThetaSampling>()?;
    m.add_function(wrap_pyfunction!(generate_psf, m)?)?;
    m.add_function(wrap_pyfunction!(generate_psf_json, m)?)?;
    Ok(())
}
