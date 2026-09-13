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

/// How the focusing angle θ is sampled by the quadrature.
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ThetaSampling {
    /// Midpoint rule on `n_theta` equal steps of θ (the numpy reference model).
    #[default]
    #[cfg_attr(feature = "serde", serde(rename = "UNIFORM"))]
    Uniform,
    /// When the aperture reaches past the critical angle of the sample
    /// (n₁ sin α > n₃), the axial phase k₀ n₃ cos θ₃ z has a square-root
    /// singularity at the critical angle and the uniform rule converges
    /// poorly. This variant integrates the sub-critical part over
    /// w = cos θ₃ instead, where that phase is linear, and keeps uniform
    /// steps elsewhere; identical to `Uniform` when there is no critical
    /// angle inside the aperture. `n_theta` is the total number of nodes.
    #[cfg_attr(feature = "serde", serde(rename = "ADAPTIVE"))]
    Adaptive,
}

/// A pixelated phase pattern displayed on a spatial light modulator (SLM)
/// in a plane conjugate to the back pupil.
///
/// The pattern is `n × n` pixels, row-major with the first row at +y, and
/// covers the square `[-extent·r0, extent·r0]²` of the pupil (µm). Light
/// outside the panel passes unmodulated. Phases are wrapped to [0, 2π) and,
/// if `levels` > 1, quantized to that many equally spaced levels (an 8-bit
/// SLM has 256), which is how a real device displays a pattern. The dead
/// space between pixels (`fill_factor` < 1) is far below the quadrature
/// resolution, so it is modelled by its area average: each pixel transmits
/// `fill_factor · exp(iφ) + (1 − fill_factor)`, i.e. a fraction of the light
/// is left unmodulated (the zeroth order of a real SLM). It multiplies the
/// pupil on top of the analytic phase mask (`Mode`) and the Zernike
/// aberrations, so it can either replace the phase plate (`Mode::Gaussian`
/// or `Mode::Loaded` plus a vortex on the SLM) or add to it (aberration
/// correction).
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[derive(Clone, Debug, PartialEq)]
pub struct Slm {
    /// Pixels per side.
    pub n: usize,
    /// Phase per pixel (radians), `n * n` values, row-major, first row at +y.
    pub phase: Vec<f64>,
    /// Half-width of the panel in units of the pupil radius r0 (1: the pupil inscribes the panel).
    pub extent: f64,
    /// Number of phase levels; 0 or 1 = continuous.
    pub levels: u32,
    /// Fraction of each pixel's area that modulates the phase, in (0, 1].
    pub fill_factor: f64,
    /// Lateral misalignment of the panel, in units of r0.
    pub offset_x: f64,
    pub offset_y: f64,
}

impl Default for Slm {
    fn default() -> Self {
        Slm { n: 0, phase: Vec::new(), extent: 1.0, levels: 0, fill_factor: 1.0, offset_x: 0.0, offset_y: 0.0 }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Slm {
    #[new]
    #[pyo3(signature = (n, phase, extent = 1.0, levels = 0, fill_factor = 1.0, offset_x = 0.0, offset_y = 0.0))]
    fn py_new(n: usize, phase: Vec<f64>, extent: f64, levels: u32, fill_factor: f64, offset_x: f64, offset_y: f64) -> Self {
        Slm { n, phase, extent, levels, fill_factor, offset_x, offset_y }
    }
}

impl Slm {
    pub fn validate(&self) -> Result<(), String> {
        if self.n == 0 {
            return Err("SLM: n must be positive".into());
        }
        if self.phase.len() != self.n * self.n {
            return Err(format!("SLM: expected {} phase values ({}²), got {}", self.n * self.n, self.n, self.phase.len()));
        }
        if !(self.extent > 0.0) || !self.extent.is_finite() {
            return Err("SLM: extent must be positive".into());
        }
        if !(self.fill_factor > 0.0 && self.fill_factor <= 1.0) {
            return Err("SLM: fill_factor must be in (0, 1]".into());
        }
        if self.phase.iter().any(|v| !v.is_finite()) {
            return Err("SLM: phase values must be finite".into());
        }
        Ok(())
    }

    /// Displayed phase (rad) of the pixel under (u, v), in units of r0, the
    /// pupil radius: wrapped and quantized like the device does it. `None`
    /// outside the panel.
    pub fn phase_at(&self, u: f64, v: f64) -> Option<f64> {
        let fu = ((u - self.offset_x) / self.extent + 1.0) * 0.5; // 0..1 across the panel
        let fv = (1.0 - (v - self.offset_y) / self.extent) * 0.5; // row 0 at +y
        if !(0.0..1.0).contains(&fu) || !(0.0..1.0).contains(&fv) {
            return None;
        }
        let n = self.n as f64;
        let i = ((fu * n).floor() as usize).min(self.n - 1);
        let j = ((fv * n).floor() as usize).min(self.n - 1);
        let wrapped = self.phase[j * self.n + i].rem_euclid(2.0 * PI);
        Some(if self.levels > 1 {
            let l = self.levels as f64;
            (wrapped / (2.0 * PI) * l).floor().min(l - 1.0) * (2.0 * PI / l)
        } else {
            wrapped
        })
    }

    /// Complex transmission of the panel at (u, v): the pixel phase weighted
    /// by the fill factor plus the unmodulated dead-space fraction; 1 outside.
    pub fn factor_at(&self, u: f64, v: f64) -> Complex64 {
        match self.phase_at(u, v) {
            None => Complex64::new(1.0, 0.0),
            Some(phase) => Complex64::from_polar(self.fill_factor, phase) + (1.0 - self.fill_factor),
        }
    }
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

    /// Optional spatial light modulator conjugate to the pupil.
    pub slm: Option<Slm>,
    pub theta_sampling: ThetaSampling,
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
            slm: None,
            theta_sampling: ThetaSampling::Uniform,
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
        if let Some(slm) = &self.slm {
            slm.validate()?;
        }
        if self.mode == Mode::Loaded && self.slm.is_none() {
            return Err("Mode LOADED needs a phase pattern on the SLM".into());
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
        // Loaded: the pattern is displayed on the SLM; DonutBottle is split before getting here
        _ => Complex64::new(1.0, 0.0),
    }
}

/// Transmission of the SLM at pupil position (x, y) in µm; 1 without an SLM.
fn slm_factor(x: f64, y: f64, s: &PsfConfig) -> Complex64 {
    match &s.slm {
        Some(slm) => slm.factor_at(x / s.r0, y / s.r0),
        None => Complex64::new(1.0, 0.0),
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

/// Where the adaptive rule switches from θ to w = cos θ₃: this fraction of
/// the critical angle (the singularity only matters close to it).
pub const ADAPTIVE_SPLIT: f64 = 0.6;
/// Node density of the w segment relative to the uniform segments.
pub const ADAPTIVE_DENSITY: f64 = 2.0;

/// Quadrature nodes (θ, Δθ-weight) over the integration range [0, α_int]
/// (`n_theta · deltatheta`, the Python convention), see [`ThetaSampling`].
pub fn theta_nodes(s: &PsfConfig) -> Vec<(f64, f64)> {
    let n = s.n_theta;
    let alpha = n as f64 * s.deltatheta;
    let mut out = Vec::with_capacity(n + 8);
    let uniform = |a: f64, b: f64, m: usize, out: &mut Vec<(f64, f64)>| {
        let h = (b - a) / m as f64;
        for k in 0..m {
            out.push((a + (k as f64 + 0.5) * h, h));
        }
    };
    let ratio = s.n1 / s.n3; // sin θ₃ = ratio · sin θ
    let critical = if ratio > 1.0 { (1.0 / ratio).asin() } else { f64::INFINITY };
    if s.theta_sampling == ThetaSampling::Uniform || critical >= alpha {
        uniform(0.0, alpha, n, &mut out);
        return out;
    }
    let theta_a = ADAPTIVE_SPLIT * critical;
    let len = theta_a + ADAPTIVE_DENSITY * (critical - theta_a) + (alpha - critical);
    let n1 = ((n as f64) * theta_a / len).round().max(4.0) as usize;
    let n3 = ((n as f64) * (alpha - critical) / len).round().max(2.0) as usize;
    let n2 = n.saturating_sub(n1 + n3).max(8);
    uniform(0.0, theta_a, n1, &mut out);
    // w = cos θ₃ runs from w_a at θ_a down to 0 at the critical angle
    let w_a = (1.0 - (ratio * theta_a.sin()).powi(2)).max(0.0).sqrt();
    let h = w_a / n2 as f64;
    for k in 0..n2 {
        let w = w_a - (k as f64 + 0.5) * h; // descending w: ascending θ
        let sin_t = (1.0 - w * w).sqrt() / ratio;
        let theta = sin_t.asin();
        // |dθ/dw| = w / (ratio · sqrt(1 - w²) · cos θ)
        let jac = w / (ratio * (1.0 - w * w).sqrt() * theta.cos());
        out.push((theta, jac * h));
    }
    uniform(critical, alpha, n3, &mut out);
    out
}

/// Stage 0: per-(theta, phi) scalar work (pupil functions, Fresnel, polarization).
/// `dtheta` is the quadrature weight of this θ node.
fn theta_ring(s: &PsfConfig, theta: f64, dtheta: f64, z_coords: &[f64], scalar: bool) -> ThetaRing {
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
    let dw = s.deltaphi * dtheta;
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
        let w = Complex64::new(0.0, zval).exp() * slm_factor(x_pup_t, y_pup_t, s);

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
    let (n_xy, n_z) = (s.n_xy, s.n_z);

    let x: Vec<f64> = linspace(-s.l_obs_xy, s.l_obs_xy, n_xy);
    let y = x.clone();
    let z: Vec<f64> = linspace(-s.l_obs_z, s.l_obs_z, n_z).into_iter().map(|v| v + s.dfoc).collect();

    // Stage 0
    let nodes = theta_nodes(s);
    let n_theta = nodes.len();
    let rings: Vec<ThetaRing> = maybe_par!(nodes).map(|(theta, dtheta)| theta_ring(s, theta, dtheta, &z, scalar)).collect();

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

    /// A vortex displayed on a fine, continuous SLM must reproduce the analytic donut.
    #[test]
    fn slm_vortex_matches_donut_mode() {
        let n = 512;
        let mut phase = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                let u = ((i as f64 + 0.5) / n as f64) * 2.0 - 1.0;
                let v = 1.0 - ((j as f64 + 0.5) / n as f64) * 2.0;
                phase.push(v.atan2(u));
            }
        }
        let base = Params { Nxy: 21, Nz: 5, Ntheta: 24, Nphi: 32, Normalize: Normalize::Yes, ..Default::default() };
        let donut = generate(&Params { Mode: Mode::Donut, VC: 1.0, ..base.clone() }).unwrap();
        let slm = Slm { n, phase, ..Default::default() };
        let via_slm = generate(&Params { Mode: Mode::Loaded, SLM: Some(slm.clone()), ..base.clone() }).unwrap();
        let diff = donut.iter().zip(via_slm.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        assert!(diff < 2e-3, "max diff {diff}");
        // dark centre survives
        assert!(via_slm[[2, 10, 10]] < 1e-3);
        // 8-bit quantization is a small perturbation; 2 levels is not
        let q8 = generate(&Params { Mode: Mode::Loaded, SLM: Some(Slm { levels: 256, ..slm.clone() }), ..base.clone() }).unwrap();
        let d8 = donut.iter().zip(q8.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        assert!(d8 < 5e-2, "8-bit max diff {d8}");
        let q2 = generate(&Params { Mode: Mode::Loaded, SLM: Some(Slm { levels: 2, ..slm }), ..base }).unwrap();
        let d2 = donut.iter().zip(q2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        assert!(d2 > 0.1, "binary max diff {d2}");
    }

    fn max_diff(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
    }

    /// Oil objective into water, 20 µm deep: the aperture reaches past the
    /// critical angle. The adaptive rule must beat the uniform one at the
    /// same node count, and agree with a very fine uniform reference.
    #[test]
    fn adaptive_theta_converges_at_supercritical_angles() {
        let base = Params {
            NA: 1.4, n1: 1.518, n2: 1.518, n3: 1.33, Depth: 20.0, Wavelength: 0.488,
            L_obs_Z: 3.0, Nxy: 17, Nz: 7, Nphi: 64, Normalize: Normalize::Yes, ..Default::default()
        };
        let reference = generate(&Params { Ntheta: 4000, ..base.clone() }).unwrap();
        let uniform = generate(&Params { Ntheta: 300, ..base.clone() }).unwrap();
        let adaptive = generate(&Params { Ntheta: 300, Theta_sampling: ThetaSampling::Adaptive, ..base.clone() }).unwrap();
        let (eu, ea) = (max_diff(&reference, &uniform), max_diff(&reference, &adaptive));
        assert!(ea < eu / 3.0, "uniform {eu:.2e}, adaptive {ea:.2e}");
        assert!(ea < 5e-3, "adaptive {ea:.2e}");
    }

    /// Without a critical angle inside the aperture both rules are the same nodes.
    #[test]
    fn adaptive_theta_is_uniform_when_subcritical() {
        let base = Params { Nxy: 9, Nz: 3, Ntheta: 20, Nphi: 8, ..Default::default() };
        let a = generate(&base).unwrap();
        let b = generate(&Params { Theta_sampling: ThetaSampling::Adaptive, ..base }).unwrap();
        assert!(max_diff(&a, &b) == 0.0);
    }

    #[test]
    fn theta_nodes_integrate_to_the_aperture() {
        // sum of weights = alpha for both rules (the Jacobian is right)
        let p = Params { NA: 1.4, n1: 1.518, n2: 1.518, n3: 1.33, Ntheta: 200, Theta_sampling: ThetaSampling::Adaptive, ..Default::default() };
        let c = p.to_config().unwrap();
        let nodes = theta_nodes(&c);
        let sum: f64 = nodes.iter().map(|(_, w)| w).sum();
        let alpha = c.n_theta as f64 * c.deltatheta;
        assert!((sum - alpha).abs() < 1e-5 * alpha, "sum {sum} vs alpha {alpha}"); // midpoint rule on the Jacobian
        assert!(nodes.windows(2).all(|w| w[1].0 > w[0].0), "nodes ascend");
    }

    #[test]
    fn slm_is_validated() {
        let p = Params { Mode: Mode::Loaded, ..Default::default() };
        assert!(p.validate().is_err());
        let bad = Slm { n: 4, phase: vec![0.0; 15], ..Default::default() };
        assert!(Params { SLM: Some(bad), ..Default::default() }.validate().is_err());
        let ok = Slm { n: 4, phase: vec![0.0; 16], ..Default::default() };
        assert!(Params { Mode: Mode::Loaded, SLM: Some(ok), ..Default::default() }.validate().is_ok());
    }

    /// A 93 % fill factor leaves 7 % of the amplitude unmodulated: the donut
    /// centre lifts by about (0.07)² relative to a Gaussian focus.
    #[test]
    fn fill_factor_leaks_into_the_donut_centre() {
        let n = 128;
        let mut phase = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                let u = ((i as f64 + 0.5) / n as f64) * 2.0 - 1.0;
                let v = 1.0 - ((j as f64 + 0.5) / n as f64) * 2.0;
                phase.push(v.atan2(u));
            }
        }
        let base = Params { Mode: Mode::Loaded, Nxy: 11, Nz: 3, Ntheta: 24, Nphi: 32, Normalize: Normalize::No, ..Default::default() };
        let gauss = generate(&Params { Mode: Mode::Gaussian, SLM: None, ..base.clone() }).unwrap();
        let full = generate(&Params { SLM: Some(Slm { n, phase: phase.clone(), ..Default::default() }), ..base.clone() }).unwrap();
        let leaky = generate(&Params { SLM: Some(Slm { n, phase, fill_factor: 0.93, ..Default::default() }), ..base }).unwrap();
        let centre = |v: &Array3<f64>| v[[1, 5, 5]];
        assert!(centre(&full) < 1e-4 * centre(&gauss));
        let ratio = centre(&leaky) / centre(&gauss);
        assert!((ratio - 0.07 * 0.07).abs() < 0.002, "ratio {ratio}");
    }

    #[test]
    fn flat_slm_changes_nothing() {
        let base = Params { Nxy: 9, Nz: 3, Ntheta: 10, Nphi: 8, Normalize: Normalize::No, ..Default::default() };
        let a = generate(&base).unwrap();
        let slm = Slm { n: 8, phase: vec![0.0; 64], levels: 256, fill_factor: 0.9, ..Default::default() };
        let b = generate(&Params { SLM: Some(slm), ..base }).unwrap();
        let diff = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max);
        assert!(diff < 1e-12, "max diff {diff}");
    }

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
