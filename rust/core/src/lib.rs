//! Vectorial PSF simulator (Richards–Wolf / Török stratified medium).
//!
//! This crate is backend-agnostic: it is compiled into the Python extension
//! `faser._core` (with the `python` + `parallel` features) and into the
//! WebAssembly module used by the documentation playground (single-threaded).
//!
//! Two layers of configuration exist:
//! - [`Params`] is the user-facing model, identical to the Python `PSFConfig`
//!   (NA, wavelength, refractive indices, ...), and derives the integration
//!   quantities (`k0`, `alpha`, `r0`, `Dfoc`, ...).
//! - [`PsfConfig`] is the flat, fully derived struct the integrator consumes.

use ndarray::{Array2, Array3, ArrayViewMut2, Axis};
use num_complex::Complex64;
use std::f64::consts::PI;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "python")]
use pyo3::prelude::*;

mod params;
pub use params::{Params, Derived, Window};
pub mod sample;
pub use sample::{SampleKind, SampleSpec, generate_sample};
pub mod convolve;
pub use convolve::{add_shot_noise, convolve3d};

/// `into_par_iter()` with the `parallel` feature, a plain iterator otherwise.
macro_rules! maybe_par {
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        {
            $e.into_par_iter()
        }
        #[cfg(not(feature = "parallel"))]
        {
            $e.into_iter()
        }
    }};
}

// --- Enums ---

#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    #[cfg_attr(feature = "serde", serde(rename = "GAUSSIAN"))]
    Gaussian,
    #[cfg_attr(feature = "serde", serde(rename = "DONUT"))]
    Donut,
    #[cfg_attr(feature = "serde", serde(rename = "BOTTLE"))]
    Bottle,
    #[cfg_attr(feature = "serde", serde(rename = "DONUT BOTTLE"))]
    DonutBottle,
    #[cfg_attr(feature = "serde", serde(rename = "LOADED"))]
    Loaded,
}

#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Normalize {
    #[cfg_attr(feature = "serde", serde(rename = "YES"))]
    Yes,
    #[cfg_attr(feature = "serde", serde(rename = "NO"))]
    No,
}

// --- Flat, fully derived config consumed by the integrator ---

#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug)]
pub struct PsfConfig {
    pub n1: f64,
    pub n2: f64,
    pub n3: f64,
    pub r0: f64,
    pub r0_eff: f64,
    pub n_xy: usize,
    pub n_z: usize,
    pub n_theta: usize,
    pub n_phi: usize,
    pub deltatheta: f64,
    pub deltaphi: f64,
    pub k0: f64,
    pub waist: f64,
    pub wd: f64,
    pub l_obs_xy: f64,
    pub l_obs_z: f64,
    pub dfoc: f64,
    pub thickness: f64,
    pub depth: f64,
    pub collar: f64,
    pub mode: Mode,
    pub normalize: Normalize,
    pub polarization: usize,
    pub psi: f64,
    pub eps: f64,
    pub alpha_eff: f64,

    // Geometry offsets (coverslip tilt)
    pub cg: f64,
    pub sg: f64,

    // Zernike coefficients
    pub a0: f64,
    pub a1: f64,
    pub a2: f64,
    pub a3: f64,
    pub a4: f64,
    pub a5: f64,
    pub a6: f64,
    pub a7: f64,
    pub a8: f64,
    pub a9: f64,
    pub a12: f64,
    pub a24: f64,

    // Offsets
    pub ampl_offset_x: f64,
    pub ampl_offset_y: f64,
    pub mask_offset_x: f64,
    pub mask_offset_y: f64,
    pub aberration_offset_x: f64,
    pub aberration_offset_y: f64,

    // Mask parameters
    pub vc: f64,
    pub rc: f64,
    pub ring_radius: f64,
    pub p: f64,
}

impl Default for PsfConfig {
    fn default() -> Self {
        PsfConfig {
            n1: 1.518, n2: 1.518, n3: 1.33, r0: 1.0, r0_eff: 1.0,
            n_xy: 64, n_z: 32, n_theta: 50, n_phi: 50,
            deltatheta: 0.01, deltaphi: 0.01, k0: 10.0,
            waist: 1.0, wd: 300.0, l_obs_xy: 5.0, l_obs_z: 5.0,
            dfoc: 0.0, thickness: 170.0, depth: 0.0, collar: 0.0,
            mode: Mode::Gaussian, normalize: Normalize::Yes,
            polarization: 0, psi: 0.0, eps: 0.0, alpha_eff: 1.0,
            cg: 1.0, sg: 0.0,
            a0: 0., a1: 0., a2: 0., a3: 0., a4: 0., a5: 0., a6: 0.,
            a7: 0., a8: 0., a9: 0., a12: 0., a24: 0.,
            ampl_offset_x: 0., ampl_offset_y: 0.,
            mask_offset_x: 0., mask_offset_y: 0.,
            aberration_offset_x: 0., aberration_offset_y: 0.,
            vc: 0., rc: 0., ring_radius: 0., p: 0.5,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PsfConfig {
    #[new]
    fn py_new() -> Self {
        Self::default()
    }
}

impl PsfConfig {
    /// Validate the values the integrator relies on.
    pub fn validate(&self) -> Result<(), String> {
        if self.polarization > 2 {
            return Err("polarization must be 0 (elliptical), 1 (radial) or 2 (azimuthal)".into());
        }
        if self.n_xy == 0 || self.n_z == 0 || self.n_theta == 0 || self.n_phi == 0 {
            return Err("grid sizes must be positive".into());
        }
        if self.mode == Mode::Loaded {
            return Err("Mode LOADED (custom phase mask) is not supported by the native backend".into());
        }
        Ok(())
    }
}

// --- Internal calculation logic ---

fn cart_to_polar(x: f64, y: f64) -> (f64, f64) {
    ((x.powi(2) + y.powi(2)).sqrt(), y.atan2(x))
}

fn amplitude(x: f64, y: f64, s: &PsfConfig) -> f64 {
    (-(x.powi(2) + y.powi(2)) / s.waist.powi(2)).exp()
}

fn zernike(x: f64, y: f64, s: &PsfConfig) -> f64 {
    let (temp, phi) = cart_to_polar(x, y);
    let rho = temp / s.r0;

    let rho2 = rho.powi(2);
    let rho3 = rho.powi(3);
    let rho4 = rho.powi(4);
    let rho6 = rho.powi(6);
    let sin_phi = phi.sin();
    let cos_phi = phi.cos();

    let z0 = 1.0;
    let z1 = 2.0 * rho * sin_phi;
    let z2 = 2.0 * rho * cos_phi;
    let z3 = 6.0f64.sqrt() * rho2 * (2.0 * phi).sin();
    let z4 = 3.0f64.sqrt() * (2.0 * rho2 - 1.0);
    let z5 = 6.0f64.sqrt() * rho2 * (2.0 * phi).cos();
    let z6 = 8.0f64.sqrt() * rho3 * (3.0 * phi).sin();
    let z7 = 8.0f64.sqrt() * (3.0 * rho3 - 2.0 * rho) * sin_phi;
    let z8 = 8.0f64.sqrt() * (3.0 * rho3 - 2.0 * rho) * cos_phi;
    let z9 = 8.0f64.sqrt() * rho3 * (3.0 * phi).cos();
    let z12 = 5.0f64.sqrt() * (6.0 * rho4 - 6.0 * rho2 + 1.0);
    let z24 = 7.0f64.sqrt() * (20.0 * rho6 - 30.0 * rho4 + 12.0 * rho2 - 1.0);

    s.a0 * z0 + s.a1 * z1 + s.a2 * z2 + s.a3 * z3 + s.a4 * z4 + s.a5 * z5
        + s.a6 * z6 + s.a7 * z7 + s.a8 * z8 + s.a9 * z9 + s.a12 * z12 + s.a24 * z24
}

fn fresnel_coeff(s: &PsfConfig, ca: f64, c2a: Complex64, c2at: Complex64, c3a: Complex64) -> (Complex64, Complex64) {
    let t1p = 2.0 * s.n1 * ca / (s.n2 * ca + s.n1 * c2a);
    let t2p = 2.0 * s.n2 * c2a / (s.n3 * c2a + s.n2 * c3a);
    let r1p = (s.n2 * ca - s.n1 * c2a) / (s.n2 * ca + s.n1 * c2a);
    let r2p = (s.n3 * c2a - s.n2 * c3a) / (s.n3 * c2a + s.n2 * c3a);

    let t1s = 2.0 * s.n1 * ca / (s.n1 * ca + s.n2 * c2a);
    let t2s = 2.0 * s.n2 * c2a / (s.n2 * c2a + s.n3 * c3a);
    let r1s = (s.n1 * ca - s.n2 * c2a) / (s.n1 * ca + s.n2 * c2a);
    let r2s = (s.n2 * c2a - s.n3 * c3a) / (s.n2 * c2a + s.n3 * c3a);

    // Single-pass phase through the coverslip, minus the design (collar) phase
    let beta = Complex64::i() * s.k0 * s.n2 * (s.thickness * c2a - s.collar * c2at);
    // Round-trip phase for the multiple-reflection (Airy) term: physical thickness only
    let beta_phys = Complex64::i() * s.k0 * s.n2 * s.thickness * c2a;
    let exp_beta = beta.exp();
    let exp_2beta = (2.0 * beta_phys).exp();

    let tp = t2p * t1p * exp_beta / (1.0 + r1p * r2p * exp_2beta);
    let ts = t2s * t1s * exp_beta / (1.0 + r1s * r2s * exp_2beta);
    (tp, ts)
}

fn phase_mask(x: f64, y: f64, s: &PsfConfig) -> Complex64 {
    let (rho, phi) = cart_to_polar(x, y);
    match s.mode {
        Mode::Gaussian => Complex64::new(1.0, 0.0),
        Mode::Donut => Complex64::new(0.0, s.vc * phi).exp(),
        Mode::Bottle => {
            if rho <= s.ring_radius * s.r0 {
                Complex64::new(0.0, s.rc * PI).exp()
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        _ => Complex64::new(1.0, 0.0),
    }
}

/// Scalar coefficients of one plane-wave component of the Debye integral,
/// i.e. one (theta, phi) quadrature sample inside the aperture.
struct PlaneWave {
    kx: f64,
    ky: f64,
    cx: Complex64,
    cy: Complex64,
    cz: Complex64,
}

/// Everything that belongs to one theta ring: its in-aperture phi samples and
/// the axial phase vector exp(i k0 n3 cos(theta3) z), which is phi-independent.
struct ThetaRing {
    waves: Vec<PlaneWave>,
    phase_z: Vec<Complex64>,
}

/// Stage 0: per-(theta, phi) scalar work (pupil functions, Fresnel, polarization).
fn theta_ring(s: &PsfConfig, p: usize, z_coords: &[f64], scalar: bool) -> ThetaRing {
    let theta = (p as f64 + 0.5) * s.deltatheta; // midpoint rule
    let (sa, ca) = theta.sin_cos();

    // Snell; cosines via complex sqrt so supercritical angles are evanescent, not clamped
    let sin_theta2 = (s.n1 / s.n2) * sa;
    let c2a = Complex64::new(1.0 - sin_theta2.powi(2), 0.0).sqrt();
    let sin_theta3 = (s.n2 / s.n3) * sin_theta2;
    let c3a = Complex64::new(1.0 - sin_theta3.powi(2), 0.0).sqrt();
    let s3a = sin_theta3;

    let k_z = Complex64::i() * s.k0 * s.n3 * c3a;
    let phase_z: Vec<Complex64> = z_coords.iter().map(|&z| (k_z * z).exp()).collect();

    // phi-independent constants
    let off = s.r0 / (s.n_xy as f64);
    let (sp, cp) = s.psi.sin_cos();
    let (se, ce) = s.eps.sin_cos();
    let ell_x = Complex64::new(cp * ce, -sp * se);
    let ell_y = Complex64::new(sp * ce, cp * se);
    let dw = s.deltaphi * s.deltatheta;
    let k_t = s.k0 * s.n1 * sa;

    let mut waves = Vec::with_capacity(s.n_phi);
    for q in 0..s.n_phi {
        let phi = q as f64 * s.deltaphi;
        let (si, ci) = phi.sin_cos();

        // Pupil coordinates, rotated by the coverslip tilt (about y)
        let x_pup_t = s.cg * s.wd * sa * ci - s.sg * s.wd * ca;
        let y_pup_t = s.wd * sa * si;

        // Polar angle in the objective frame
        let cat = (sa * ci * s.sg + ca * s.cg).clamp(-1.0, 1.0);
        let theta_t = cat.acos();
        if theta_t > s.alpha_eff {
            continue;
        }
        let sin_theta2_t = (s.n1 / s.n2) * theta_t.sin();
        let c2at = Complex64::new(1.0 - sin_theta2_t.powi(2), 0.0).sqrt();

        // Pupil functions
        let amp = amplitude(x_pup_t - off * s.ampl_offset_x, y_pup_t - off * s.ampl_offset_y, s);
        let pm = phase_mask(x_pup_t - off * s.mask_offset_x, y_pup_t - off * s.mask_offset_y, s);
        let zval = zernike(x_pup_t - off * s.aberration_offset_x, y_pup_t - off * s.aberration_offset_y, s);
        let w = Complex64::new(0.0, zval).exp();

        // Incident polarization: 0 = elliptical, 1 = radial, 2 = azimuthal
        let (p_inc_0, p_inc_1) = match s.polarization {
            0 => (ell_x, ell_y),
            1 => (Complex64::new(ci, 0.0), Complex64::new(si, 0.0)),
            _ => (Complex64::new(-si, 0.0), Complex64::new(ci, 0.0)),
        };

        // Scalar model: same pupil, phase mask, aberrations and stratified-medium
        // phase, but a scalar field (no polarization coupling, no Fresnel
        // transmission). The comparison against it isolates the vectorial effects.
        let (p_foc_0, p_foc_1, p_foc_2) = if scalar {
            (Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0))
        } else {
            let (tp, ts) = fresnel_coeff(s, ca, c2a, c2at, c3a);

            // Polarization matrix T (third column unused: no longitudinal input)
            let t00 = tp * c3a * ci * ci + ts * si * si;
            let t01 = (tp * c3a - ts) * si * ci;
            let t10 = t01;
            let t11 = tp * c3a * si * si + ts * ci * ci;
            let t20 = -tp * s3a * ci;
            let t21 = -tp * s3a * si;

            (
                t00 * p_inc_0 + t01 * p_inc_1,
                t10 * p_inc_0 + t11 * p_inc_1,
                t20 * p_inc_0 + t21 * p_inc_1,
            )
        };

        // Coverslip / depth aberration and apodization
        let psi_w = s.n3 * s.depth * c3a - s.n1 * (s.thickness + s.depth) * ca + s.n1 * s.collar * cat;
        let ab_wind = (Complex64::i() * s.k0 * psi_w).exp();
        let prefactor = Complex64::from(sa * cat.sqrt() * amp * dw) * pm * w * ab_wind;

        waves.push(PlaneWave {
            kx: k_t * ci,
            ky: k_t * si,
            cx: prefactor * p_foc_0,
            cy: prefactor * p_foc_1,
            cz: prefactor * p_foc_2,
        });
    }

    ThetaRing { waves, phase_z }
}

/// Fill `out[j] = exp(i k x[j])` using a phase recurrence, re-seeded with an
/// exact exp every RESEED elements to keep rounding error at the 1e-15 level.
fn phase_vector(k: f64, x: &[f64], out: &mut [Complex64]) {
    const RESEED: usize = 32;
    let n = x.len();
    if n == 0 {
        return;
    }
    let dx = if n > 1 { x[1] - x[0] } else { 0.0 };
    let step = Complex64::from_polar(1.0, k * dx);
    let mut cur = Complex64::from_polar(1.0, k * x[0]);
    for (j, o) in out.iter_mut().enumerate() {
        if j % RESEED == 0 {
            cur = Complex64::from_polar(1.0, k * x[j]);
        }
        *o = cur;
        cur *= step;
    }
}

/// Stage 1 kernel: accumulate the phi-sum of one theta ring into a chunk of
/// rows [i0, i0 + rows) of the three 2-D (y, x) field accumulators.
fn accumulate_rows(
    ring: &ThetaRing,
    i0: usize,
    x: &[f64],
    y: &[f64],
    mut ax: ArrayViewMut2<Complex64>,
    mut ay: ArrayViewMut2<Complex64>,
    mut az: ArrayViewMut2<Complex64>,
) {
    let n = x.len();
    let rows = ax.nrows();
    let mut px = vec![Complex64::new(0.0, 0.0); n];

    for wv in &ring.waves {
        phase_vector(wv.kx, x, &mut px);
        for r in 0..rows {
            let py = Complex64::from_polar(1.0, wv.ky * y[i0 + r]);
            let (rx, ry, rz) = (wv.cx * py, wv.cy * py, wv.cz * py);
            let ra = ax.row_mut(r).into_slice().expect("contiguous row");
            let rb = ay.row_mut(r).into_slice().expect("contiguous row");
            let rc = az.row_mut(r).into_slice().expect("contiguous row");
            for (((a, b), c), &e) in ra.iter_mut().zip(rb).zip(rc).zip(&px) {
                *a += rx * e;
                *b += ry * e;
                *c += rz * e;
            }
        }
    }
}

/// Intensity |Ex|^2 + |Ey|^2 + |Ez|^2 on the (z, y, x) grid.
///
/// The Debye integral is separable in z: exp(i k0 n3 cos(theta3) z) depends on
/// theta only. So the phi-sum is accumulated on 2-D (y, x) arrays per theta
/// ring (stage 1, parallel over theta x row-chunks) and the z dependence is
/// applied afterwards as a rank-1 update per z-slice (stage 2, parallel over
/// z). Cost is O(Ntheta * (Nphi + Nz) * Nxy^2) instead of
/// O(Ntheta * Nphi * Nz * Nxy^2).
fn calculate_intensity(s: &PsfConfig, scalar: bool) -> Array3<f64> {
    const ROW_CHUNK: usize = 8;
    let (n_xy, n_z, n_theta) = (s.n_xy, s.n_z, s.n_theta);

    let x: Vec<f64> = linspace(-s.l_obs_xy, s.l_obs_xy, n_xy);
    let y = x.clone();
    let z: Vec<f64> = linspace(-s.l_obs_z, s.l_obs_z, n_z).into_iter().map(|v| v + s.dfoc).collect();

    // Stage 0
    let rings: Vec<ThetaRing> = maybe_par!(0..n_theta).map(|p| theta_ring(s, p, &z, scalar)).collect();

    // Stage 1: per-theta 2-D accumulators, filled by (theta, row-chunk) tasks
    let mut ax: Vec<Array2<Complex64>> = (0..n_theta).map(|_| Array2::zeros((n_xy, n_xy))).collect();
    let mut ay = ax.clone();
    let mut az = ax.clone();
    {
        let mut tasks: Vec<(usize, usize, ArrayViewMut2<Complex64>, ArrayViewMut2<Complex64>, ArrayViewMut2<Complex64>)> =
            Vec::with_capacity(n_theta * n_xy.div_ceil(ROW_CHUNK));
        for (p, ((a, b), c)) in ax.iter_mut().zip(ay.iter_mut()).zip(az.iter_mut()).enumerate() {
            let chunks = a
                .axis_chunks_iter_mut(Axis(0), ROW_CHUNK)
                .zip(b.axis_chunks_iter_mut(Axis(0), ROW_CHUNK))
                .zip(c.axis_chunks_iter_mut(Axis(0), ROW_CHUNK));
            for (ci, ((va, vb), vc)) in chunks.enumerate() {
                tasks.push((p, ci * ROW_CHUNK, va, vb, vc));
            }
        }
        maybe_par!(tasks).for_each(|(p, i0, va, vb, vc)| {
            accumulate_rows(&rings[p], i0, &x, &y, va, vb, vc);
        });
    }

    // Stage 2: z outer product and intensity, one z-slice per task
    let mut intensity = Array3::<f64>::zeros((n_z, n_xy, n_xy));
    maybe_par!(intensity.outer_iter_mut())
        .enumerate()
        .for_each(|(k, mut slice)| {
            let n2 = n_xy * n_xy;
            let mut ex = vec![Complex64::new(0.0, 0.0); n2];
            let mut ey = ex.clone();
            let mut ez = ex.clone();
            for p in 0..n_theta {
                let f = rings[p].phase_z[k];
                let (sa, sb, sc) = (
                    ax[p].as_slice().expect("contiguous"),
                    ay[p].as_slice().expect("contiguous"),
                    az[p].as_slice().expect("contiguous"),
                );
                for i in 0..n2 {
                    ex[i] += sa[i] * f;
                    ey[i] += sb[i] * f;
                    ez[i] += sc[i] * f;
                }
            }
            let out = slice.as_slice_mut().expect("contiguous slice");
            for i in 0..n2 {
                out[i] = ex[i].norm_sqr() + ey[i].norm_sqr() + ez[i].norm_sqr();
            }
        });

    intensity
}

pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![a],
        _ => (0..n).map(|i| a + (b - a) * (i as f64) / ((n - 1) as f64)).collect(),
    }
}

/// PSF intensity of shape (Nz, Ny, Nx) for a fully derived [`PsfConfig`].
///
/// Does not validate; call [`PsfConfig::validate`] first.
pub fn psf_volume(config: PsfConfig) -> Array3<f64> {
    psf_volume_model(config, false)
}

/// Like [`psf_volume`] with the scalar Debye integral (see `theta_ring`).
pub fn psf_volume_scalar(config: PsfConfig) -> Array3<f64> {
    psf_volume_model(config, true)
}

pub fn psf_volume_model(mut config: PsfConfig, scalar: bool) -> Array3<f64> {
    let mut result = if config.mode == Mode::DonutBottle {
        config.mode = Mode::Donut;
        let i_donut = calculate_intensity(&config, scalar);
        config.mode = Mode::Bottle;
        let i_bottle = calculate_intensity(&config, scalar);
        i_donut * config.p + i_bottle * (1.0 - config.p)
    } else {
        calculate_intensity(&config, scalar)
    };

    if config.normalize == Normalize::Yes {
        let max = result.iter().copied().fold(0.0_f64, f64::max);
        if max > 0.0 {
            #[cfg(feature = "parallel")]
            result.par_mapv_inplace(|v| v / max);
            #[cfg(not(feature = "parallel"))]
            result.mapv_inplace(|v| v / max);
        }
    }
    result
}

/// PSF intensity of shape (Nz, Nxy, Nxy) from user-facing [`Params`].
pub fn generate(params: &Params) -> Result<Array3<f64>, String> {
    generate_model(params, false)
}

/// Vectorial (`scalar = false`) or scalar (`scalar = true`) PSF from [`Params`].
pub fn generate_model(params: &Params, scalar: bool) -> Result<Array3<f64>, String> {
    let config = params.to_config()?;
    config.validate()?;
    Ok(psf_volume_model(config, scalar))
}

#[cfg(test)]
mod model_tests {
    use super::*;

    #[test]
    fn scalar_differs_from_vectorial_at_high_na() {
        let p = Params { NA: 1.4, n1: 1.518, n2: 1.518, n3: 1.518, Nxy: 15, Nz: 7, Ntheta: 20, Nphi: 16, Normalize: Normalize::Yes, ..Default::default() };
        let v = generate_model(&p, false).unwrap();
        let s = generate_model(&p, true).unwrap();
        assert_eq!(v.shape(), s.shape());
        // both peak at the centre
        assert!((v[[3, 7, 7]] - 1.0).abs() < 1e-12 && (s[[3, 7, 7]] - 1.0).abs() < 1e-12);
        // the scalar spot is narrower / rounder: the volumes are not identical
        let diff = v.iter().zip(s.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        assert!(diff > 0.01, "max diff {diff}");
    }
}
