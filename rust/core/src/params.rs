//! User-facing simulation parameters.
//!
//! [`Params`] mirrors the Python `faser.generators.base.PSFConfig` model
//! field-for-field (including the field names, so the `psf_config.json`
//! written by the napari plugin / CLI deserializes directly), and reproduces
//! its derived properties (`k0`, `alpha`, `r0`, `Dfoc`, ...). It is what the
//! WebAssembly playground consumes; Python keeps deriving these itself.

use std::f64::consts::PI;

use crate::{Mode, Normalize, PsfConfig, Slm, ThetaSampling};

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Window {
    #[cfg_attr(feature = "serde", serde(rename = "NO"))]
    No,
    #[cfg_attr(feature = "serde", serde(rename = "CUSTOM"))]
    Custom,
}

/// All user-facing parameters, with the Python defaults.
///
/// Unknown JSON fields (`loaded_phase_mask`, the noise settings, ...) are
/// ignored; missing fields take their default.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct Params {
    // Sampling
    /// Observation half-width in XY (µm)
    pub L_obs_XY: f64,
    /// Observation half-width in Z (µm)
    pub L_obs_Z: f64,
    pub Nxy: usize,
    pub Nz: usize,
    pub Ntheta: usize,
    pub Nphi: usize,
    pub Normalize: Normalize,

    // Geometry
    /// Numerical aperture of the objective
    pub NA: f64,
    /// Effective focal length of the objective (µm); pupil radius r0 = WD·sin(alpha)
    pub WD: f64,
    /// Refractive index of the immersion medium
    pub n1: f64,
    /// Refractive index of the coverslip
    pub n2: f64,
    /// Refractive index of the sample
    pub n3: f64,
    /// Coverslip thickness (µm)
    pub Thickness: f64,
    /// Correction collar setting (µm)
    pub Collar: f64,
    /// Distance from the coverslip to the nominal focus (µm)
    pub Depth: f64,
    /// Coverslip tilt (degrees)
    pub Tilt: f64,
    pub Window: Window,
    /// Cranial window radius (mm)
    pub Wind_Radius: f64,
    /// Cranial window depth (mm)
    pub Wind_Depth: f64,
    pub Wind_Offset_x: f64,
    pub Wind_Offset_y: f64,

    // Aberrations (Zernike coefficients, radians of phase)
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
    pub Aberration_offset_x: f64,
    pub Aberration_offset_y: f64,

    // Beam
    pub Mode: Mode,
    /// 1 = elliptical, 2 = radial, 3 = azimuthal
    pub Polarization: u8,
    /// Wavelength (µm)
    pub Wavelength: f64,
    /// 1/e field radius of the Gaussian input beam on the pupil (µm)
    pub Waist: f64,
    pub Ampl_offset_x: f64,
    pub Ampl_offset_y: f64,

    // Polarization
    /// Polarization direction (degrees)
    pub Psi: f64,
    /// Polarization ellipticity (degrees)
    pub Epsilon: f64,

    // STED phase masks
    pub VC: f64,
    pub RC: f64,
    pub Ring_Radius: f64,
    pub Mask_offset_x: f64,
    pub Mask_offset_y: f64,
    /// Donut (p) / bottle (1 - p) mix for DONUT BOTTLE
    pub p: f64,

    /// Optional spatial light modulator conjugate to the back pupil; `null` = none.
    pub SLM: Option<Slm>,
    /// θ quadrature: "UNIFORM" (the numpy reference) or "ADAPTIVE" (see [`ThetaSampling`]).
    pub Theta_sampling: ThetaSampling,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            L_obs_XY: 2.0,
            L_obs_Z: 2.0,
            Nxy: 31,
            Nz: 31,
            Ntheta: 31,
            Nphi: 31,
            Normalize: Normalize::No,
            NA: 1.0,
            WD: 2800.0,
            n1: 1.33,
            n2: 1.52,
            n3: 1.38,
            Thickness: 170.0,
            Collar: 170.0,
            Depth: 0.0,
            Tilt: 0.0,
            Window: Window::No,
            Wind_Radius: 2.3,
            Wind_Depth: 2.23,
            Wind_Offset_x: 0.0,
            Wind_Offset_y: 0.0,
            a0: 0.0, a1: 0.0, a2: 0.0, a3: 0.0, a4: 0.0, a5: 0.0,
            a6: 0.0, a7: 0.0, a8: 0.0, a9: 0.0, a12: 0.0, a24: 0.0,
            Aberration_offset_x: 0.0,
            Aberration_offset_y: 0.0,
            Mode: Mode::Gaussian,
            Polarization: 1,
            Wavelength: 0.592,
            Waist: 8000.0,
            Ampl_offset_x: 0.0,
            Ampl_offset_y: 0.0,
            Psi: 0.0,
            Epsilon: 45.0,
            VC: 1.0,
            RC: 1.0,
            Ring_Radius: 0.707,
            Mask_offset_x: 0.0,
            Mask_offset_y: 0.0,
            p: 0.5,
            SLM: None,
            Theta_sampling: ThetaSampling::Uniform,
        }
    }
}

/// Quantities derived from [`Params`], as computed by the Python model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Derived {
    /// Vacuum wavevector (µm^-1)
    pub k0: f64,
    /// Semi-aperture angle of the objective (rad)
    pub alpha: f64,
    /// Pupil radius (µm)
    pub r0: f64,
    /// Coverslip tilt (rad)
    pub gamma: f64,
    /// Effective semi-aperture after the cranial window (rad)
    pub alpha_eff: f64,
    /// Effective NA after the cranial window
    pub na_eff: f64,
    /// Effective pupil radius (µm)
    pub r0_eff: f64,
    /// Integration range in theta (rad)
    pub alpha_int: f64,
    /// Refracted cone half-angle in the coverslip (rad)
    pub alpha2_eff: f64,
    /// Refracted cone half-angle in the sample (rad)
    pub alpha3_eff: f64,
    /// Axial shift of the aberrated focus relative to the nominal focus (µm)
    pub dfoc: f64,
    pub deltatheta: f64,
    pub deltaphi: f64,
}

impl Params {
    pub fn validate(&self) -> Result<(), String> {
        if !(self.NA > 0.0) {
            return Err("NA must be positive".into());
        }
        if self.n1 < self.NA {
            return Err("NA must be smaller than the immersion refractive index n1".into());
        }
        if self.Nxy == 0 || self.Nz == 0 || self.Ntheta == 0 || self.Nphi == 0 {
            return Err("Nxy, Nz, Ntheta and Nphi must be positive".into());
        }
        if !(1..=3).contains(&self.Polarization) {
            return Err("Polarization must be 1 (elliptical), 2 (radial) or 3 (azimuthal)".into());
        }
        if let Some(slm) = &self.SLM {
            slm.validate()?;
        }
        if self.Mode == Mode::Loaded && self.SLM.is_none() {
            return Err("Mode LOADED needs a phase pattern on the SLM".into());
        }
        if !(self.Wavelength > 0.0) || !(self.WD > 0.0) || !(self.Waist > 0.0) {
            return Err("Wavelength, WD and Waist must be positive".into());
        }
        Ok(())
    }

    fn t_wind(&self) -> f64 {
        match self.Window {
            Window::Custom => self.Wind_Depth * 1e3,
            Window::No => 2.23e3,
        }
    }

    fn r_wind(&self) -> f64 {
        match self.Window {
            Window::No => 100.0 * self.t_wind(),
            Window::Custom => self.Wind_Radius * 1e3,
        }
    }

    pub fn k0(&self) -> f64 {
        2.0 * PI / self.Wavelength
    }

    pub fn alpha(&self) -> f64 {
        (self.NA / self.n1).asin()
    }

    pub fn r0(&self) -> f64 {
        self.WD * self.alpha().sin()
    }

    pub fn gamma(&self) -> f64 {
        self.Tilt * PI / 180.0
    }

    pub fn alpha_eff(&self) -> f64 {
        (self.r_wind() / self.t_wind()).atan().min(self.alpha())
    }

    pub fn na_eff(&self) -> f64 {
        (self.n1 * self.alpha_eff().sin()).min(self.NA)
    }

    pub fn r0_eff(&self) -> f64 {
        self.WD * self.alpha_eff().sin()
    }

    pub fn alpha_int(&self) -> f64 {
        self.alpha_eff() + self.gamma().abs()
    }

    pub fn alpha2_eff(&self) -> f64 {
        ((self.n1 / self.n2) * self.alpha_eff().sin()).min(1.0).asin()
    }

    pub fn alpha3_eff(&self) -> f64 {
        ((self.n1 / self.n3) * self.alpha_eff().sin()).min(1.0).asin()
    }

    /// Axial position of the aberrated focus relative to the nominal
    /// (index-matched) focus: the defocus that minimises the sin(theta)-weighted
    /// variance of the coverslip/depth aberration phase over the pupil
    /// (balanced defocus). Positive values are deeper into the sample.
    pub fn dfoc(&self) -> f64 {
        const N: usize = 2001;
        let alpha_eff = self.alpha_eff();
        let (d, d_design, z) = (self.Thickness, self.Collar, self.Depth);

        let (mut sw, mut sg, mut sgg, mut sp, mut spg) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..N {
            let theta = alpha_eff * (i as f64) / ((N - 1) as f64);
            let (s1, c1) = theta.sin_cos();
            let w = s1;
            let s2 = (self.n1 / self.n2) * s1;
            let c2 = (1.0 - s2 * s2).max(0.0).sqrt();
            let s3 = (self.n1 / self.n3) * s1;
            let c3 = (1.0 - s3 * s3).max(0.0).sqrt();
            // optical path aberration (µm) as used in the field calculation
            let psi = self.n3 * z * c3 + self.n2 * (d - d_design) * c2 - self.n1 * (d + z) * c1
                + self.n1 * d_design * c1;
            let g = self.n3 * c3; // axial propagation term per unit defocus
            sw += w;
            sg += w * g;
            sgg += w * g * g;
            sp += w * psi;
            spg += w * psi * g;
        }
        let mean_g = sg / sw;
        let var_g = sgg / sw - mean_g * mean_g;
        if var_g == 0.0 {
            return 0.0;
        }
        let cov = spg / sw - (sp / sw) * mean_g;
        -cov / var_g
    }

    pub fn deltatheta(&self) -> f64 {
        self.alpha_int() / (self.Ntheta as f64)
    }

    pub fn deltaphi(&self) -> f64 {
        2.0 * PI / (self.Nphi as f64)
    }

    pub fn derived(&self) -> Derived {
        Derived {
            k0: self.k0(),
            alpha: self.alpha(),
            r0: self.r0(),
            gamma: self.gamma(),
            alpha_eff: self.alpha_eff(),
            na_eff: self.na_eff(),
            r0_eff: self.r0_eff(),
            alpha_int: self.alpha_int(),
            alpha2_eff: self.alpha2_eff(),
            alpha3_eff: self.alpha3_eff(),
            dfoc: self.dfoc(),
            deltatheta: self.deltatheta(),
            deltaphi: self.deltaphi(),
        }
    }

    /// Flatten into the integrator's config (validated).
    pub fn to_config(&self) -> Result<PsfConfig, String> {
        self.validate()?;
        let gamma = self.gamma();
        Ok(PsfConfig {
            n1: self.n1,
            n2: self.n2,
            n3: self.n3,
            r0: self.r0(),
            r0_eff: self.r0_eff(),
            n_xy: self.Nxy,
            n_z: self.Nz,
            n_theta: self.Ntheta,
            n_phi: self.Nphi,
            deltatheta: self.deltatheta(),
            deltaphi: self.deltaphi(),
            k0: self.k0(),
            waist: self.Waist,
            wd: self.WD,
            l_obs_xy: self.L_obs_XY,
            l_obs_z: self.L_obs_Z,
            dfoc: self.dfoc(),
            thickness: self.Thickness,
            depth: self.Depth,
            collar: self.Collar,
            mode: self.Mode,
            normalize: self.Normalize,
            polarization: (self.Polarization - 1) as usize,
            psi: self.Psi * PI / 180.0,
            eps: self.Epsilon * PI / 180.0,
            alpha_eff: self.alpha_eff(),
            cg: gamma.cos(),
            sg: gamma.sin(),
            a0: self.a0, a1: self.a1, a2: self.a2, a3: self.a3, a4: self.a4, a5: self.a5,
            a6: self.a6, a7: self.a7, a8: self.a8, a9: self.a9, a12: self.a12, a24: self.a24,
            ampl_offset_x: self.Ampl_offset_x,
            ampl_offset_y: self.Ampl_offset_y,
            mask_offset_x: self.Mask_offset_x,
            mask_offset_y: self.Mask_offset_y,
            aberration_offset_x: self.Aberration_offset_x,
            aberration_offset_y: self.Aberration_offset_y,
            vc: self.VC,
            rc: self.RC,
            ring_radius: self.Ring_Radius,
            p: self.p,
            slm: self.SLM.clone(),
            theta_sampling: self.Theta_sampling,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * b.abs().max(1.0)
    }

    /// Reference values printed by the Python model (faser.generators.base.PSFConfig).
    #[test]
    fn derived_matches_python() {
        let cases: Vec<(Params, [f64; 7])> = vec![
            (
                Params::default(),
                // k0, alpha, r0, alpha_eff, alpha_int, deltatheta, dfoc
                [10.613488694560113, 0.8509085144778487, 2105.2631578947367, 0.8509085144778487, 0.8509085144778487, 0.027448661757349956, 0.0],
            ),
            (
                Params { Depth: 50.0, Tilt: 5.0, Thickness: 180.0, Collar: 170.0, ..Default::default() },
                [10.613488694560113, 0.8509085144778487, 2105.2631578947367, 0.8509085144778487, 0.9381749770775651, 0.030263708937985974, 4.566110159842906],
            ),
            (
                Params { NA: 1.2, n3: 1.0, ..Default::default() },
                [10.613488694560113, 1.124972274337552, 2526.315789473684, 1.124972274337552, 1.124972274337552, 0.03628942820443717, 0.0],
            ),
            (
                Params { Window: Window::Custom, Wind_Radius: 1.0, Wind_Depth: 2.0, ..Default::default() },
                [10.613488694560113, 0.8509085144778487, 2105.2631578947367, 0.4636476090008061, 0.4636476090008061, 0.01495637448389697, 0.0],
            ),
            (
                Params { NA: 1.4, n1: 1.518, n2: 1.518, n3: 1.33, Depth: 10.0, Wavelength: 0.488, ..Default::default() },
                [12.87537972782702, 1.1739024744345716, 2582.345191040843, 1.1739024744345716, 1.1739024744345716, 0.03786782175595392, -3.718455718818813],
            ),
        ];
        for (p, want) in cases {
            let d = p.derived();
            let got = [d.k0, d.alpha, d.r0, d.alpha_eff, d.alpha_int, d.deltatheta, d.dfoc];
            for (g, w) in got.iter().zip(want.iter()) {
                assert!(close(*g, *w), "{p:?}: got {got:?}, want {want:?}");
            }
        }
    }

    #[test]
    fn tilt_sets_cg_sg() {
        let c = Params { Tilt: 5.0, ..Default::default() }.to_config().unwrap();
        assert!(close(c.cg, 0.9961946980917455));
        assert!(close(c.sg, 0.08715574274765817));
        assert!(close(c.eps, 0.7853981633974483));
        assert_eq!(c.polarization, 0);
    }

    #[test]
    fn rejects_invalid() {
        assert!(Params { NA: 1.5, n1: 1.33, ..Default::default() }.validate().is_err());
        assert!(Params { Polarization: 0, ..Default::default() }.validate().is_err());
        assert!(Params { Mode: Mode::Loaded, ..Default::default() }.validate().is_err());
        assert!(Params { Nxy: 0, ..Default::default() }.validate().is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parses_psf_config_json() {
        // Exactly what the napari plugin / CLI write, including fields we ignore.
        let json = r#"{"L_obs_XY": 2.0, "Nxy": 15, "Mode": "DONUT BOTTLE", "Polarization": 2,
            "Window": "CUSTOM", "Normalize": "YES", "loaded_phase_mask": null, "Add_noise": "YES"}"#;
        let p: Params = serde_json::from_str(json).unwrap();
        assert_eq!(p.Nxy, 15);
        assert_eq!(p.Nz, 31);
        assert_eq!(p.Mode, Mode::DonutBottle);
        assert_eq!(p.Polarization, 2);
        assert_eq!(p.Window, Window::Custom);
        assert_eq!(p.Normalize, Normalize::Yes);
        let back = serde_json::to_value(&p).unwrap();
        assert_eq!(back["Mode"], "DONUT BOTTLE");
        assert_eq!(back["Window"], "CUSTOM");
    }

    #[test]
    fn generates_a_volume() {
        let p = Params { Nxy: 9, Nz: 5, Ntheta: 10, Nphi: 8, Normalize: Normalize::Yes, ..Default::default() };
        let v = crate::generate(&p).unwrap();
        assert_eq!(v.shape(), &[5, 9, 9]);
        let max = v.iter().copied().fold(0.0_f64, f64::max);
        assert!((max - 1.0).abs() < 1e-12);
        // Gaussian focus: brightest voxel at the centre of the grid
        assert!((v[[2, 4, 4]] - 1.0).abs() < 1e-12);
    }
}
