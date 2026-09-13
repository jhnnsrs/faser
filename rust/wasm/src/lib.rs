//! WebAssembly bindings of the faser simulator for the documentation
//! playground. Parameters are passed as JSON in the `psf_config.json` format
//! (the same file the napari plugin and CLI write).

use wasm_bindgen::prelude::*;

use faser_core::{Params, SampleSpec};
use ndarray::Array3;

fn parse(json: &str) -> Result<Params, JsError> {
    serde_json::from_str(json).map_err(|e| JsError::new(&format!("invalid config: {e}")))
}

/// A computed PSF volume: intensities in (z, y, x) order plus the grid shape
/// and the derived quantities the parameters imply.
#[wasm_bindgen]
pub struct PsfVolume {
    data: Vec<f32>,
    nz: u32,
    ny: u32,
    nx: u32,
    derived: String,
    max: f64,
}

#[wasm_bindgen]
impl PsfVolume {
    /// Intensities, C-order (z, y, x), single precision. Copies into JS.
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f32> {
        self.data.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn nz(&self) -> u32 {
        self.nz
    }
    #[wasm_bindgen(getter)]
    pub fn ny(&self) -> u32 {
        self.ny
    }
    #[wasm_bindgen(getter)]
    pub fn nx(&self) -> u32 {
        self.nx
    }
    /// Maximum intensity before normalization (1.0 if Normalize is YES).
    #[wasm_bindgen(getter)]
    pub fn max(&self) -> f64 {
        self.max
    }
    /// JSON of the derived quantities (`k0`, `alpha`, `r0`, `dfoc`, ...).
    #[wasm_bindgen(getter)]
    pub fn derived(&self) -> String {
        self.derived.clone()
    }
}

/// Compute the PSF for the JSON-serialized parameters. `scalar` selects the
/// scalar Debye integral (no polarization / Fresnel terms) instead of the
/// full vectorial model.
#[wasm_bindgen]
pub fn generate_psf(json: &str, scalar: bool) -> Result<PsfVolume, JsError> {
    let params = parse(json)?;
    let derived = serde_json::to_string(&params.derived()).map_err(|e| JsError::new(&e.to_string()))?;
    let volume = faser_core::generate_model(&params, scalar).map_err(|e| JsError::new(&e))?;
    let shape = volume.shape();
    let (nz, ny, nx) = (shape[0] as u32, shape[1] as u32, shape[2] as u32);
    let max = volume.iter().copied().fold(0.0_f64, f64::max);
    let data = volume.into_raw_vec().into_iter().map(|v| v as f32).collect();
    Ok(PsfVolume { data, nz, ny, nx, derived, max })
}

/// Derived quantities only (cheap; used to draw the microscope model while
/// the user edits parameters). Returns JSON.
#[wasm_bindgen]
pub fn derive(json: &str) -> Result<String, JsError> {
    let params = parse(json)?;
    params.validate().map_err(|e| JsError::new(&e))?;
    serde_json::to_string(&params.derived()).map_err(|e| JsError::new(&e.to_string()))
}

/// The default parameters as JSON (the Python `PSFConfig` defaults).
#[wasm_bindgen]
pub fn default_params() -> String {
    serde_json::to_string(&Params::default()).expect("serializable")
}

/// A plain float volume in (z, y, x) order.
#[wasm_bindgen]
pub struct Volume {
    data: Vec<f32>,
    nz: u32,
    ny: u32,
    nx: u32,
    max: f64,
}

#[wasm_bindgen]
impl Volume {
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f32> {
        self.data.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn nz(&self) -> u32 {
        self.nz
    }
    #[wasm_bindgen(getter)]
    pub fn ny(&self) -> u32 {
        self.ny
    }
    #[wasm_bindgen(getter)]
    pub fn nx(&self) -> u32 {
        self.nx
    }
    #[wasm_bindgen(getter)]
    pub fn max(&self) -> f64 {
        self.max
    }
}

fn volume_from(arr: Array3<f32>) -> Volume {
    let shape = arr.shape();
    let (nz, ny, nx) = (shape[0] as u32, shape[1] as u32, shape[2] as u32);
    let max = arr.iter().copied().fold(0.0_f32, f32::max) as f64;
    Volume { data: arr.into_raw_vec(), nz, ny, nx, max }
}

/// Generate a synthetic sample volume from a JSON `SampleSpec`
/// (`{kind, seed, nx, ny, nz, dx, dz, count, radius, spacing}`).
#[wasm_bindgen]
pub fn generate_sample(json: &str) -> Result<Volume, JsError> {
    let spec: SampleSpec = serde_json::from_str(json).map_err(|e| JsError::new(&format!("invalid sample spec: {e}")))?;
    let arr = faser_core::generate_sample(&spec).map_err(|e| JsError::new(&e))?;
    Ok(volume_from(arr))
}

/// Image a sample through a PSF: linear 3-D convolution (same size as the
/// sample, PSF normalized to unit sum) and, if `photons > 0`, Poisson shot
/// noise with that many expected counts in the brightest voxel.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn convolve(
    sample: &[f32],
    snz: u32,
    sny: u32,
    snx: u32,
    psf: &[f32],
    pnz: u32,
    pny: u32,
    pnx: u32,
    photons: f64,
    seed: u64,
) -> Result<Volume, JsError> {
    let s = Array3::from_shape_vec((snz as usize, sny as usize, snx as usize), sample.to_vec())
        .map_err(|e| JsError::new(&format!("sample shape: {e}")))?;
    let k = Array3::from_shape_vec((pnz as usize, pny as usize, pnx as usize), psf.to_vec())
        .map_err(|e| JsError::new(&format!("psf shape: {e}")))?;
    let mut out = faser_core::convolve3d(&s, &k).map_err(|e| JsError::new(&e))?;
    faser_core::add_shot_noise(&mut out, photons, seed);
    Ok(volume_from(out))
}
