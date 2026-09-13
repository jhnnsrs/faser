//! Synthetic 3-D samples ("ground truth" volumes) to image through a PSF.
//!
//! Everything is deterministic for a given seed and uses a small xorshift
//! generator so the same sample appears on every platform, wasm included.

use ndarray::Array3;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleKind {
    /// Randomly placed spheres of varying radius and brightness.
    #[cfg_attr(feature = "serde", serde(rename = "beads"))]
    Beads,
    /// Smooth random 3-D curves painted as thin tubes (microtubule-like).
    #[cfg_attr(feature = "serde", serde(rename = "filaments"))]
    Filaments,
    /// Ellipsoidal membranes with a nucleus and a few vesicles inside.
    #[cfg_attr(feature = "serde", serde(rename = "cells"))]
    Cells,
    /// Regular lattice of points, a resolution target in xy and z.
    #[cfg_attr(feature = "serde", serde(rename = "lattice"))]
    Lattice,
    /// A Siemens star in the focal plane: spokes get finer towards the centre.
    #[cfg_attr(feature = "serde", serde(rename = "spokes"))]
    Spokes,
}

/// What to generate. Sizes are in voxels, spacings in µm.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[derive(Clone, Debug, PartialEq)]
pub struct SampleSpec {
    pub kind: SampleKind,
    pub seed: u64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Lateral voxel size (µm)
    pub dx: f64,
    /// Axial voxel size (µm)
    pub dz: f64,
    /// Number of objects (beads, filaments, cells) or spokes
    pub count: usize,
    /// Object radius (µm): bead radius, filament tube radius, membrane thickness, lattice point radius
    pub radius: f64,
    /// Lattice spacing (µm)
    pub spacing: f64,
}

impl Default for SampleSpec {
    fn default() -> Self {
        SampleSpec {
            kind: SampleKind::Beads,
            seed: 1,
            nx: 96,
            ny: 96,
            nz: 48,
            dx: 0.0635,
            dz: 0.129,
            count: 30,
            radius: 0.15,
            spacing: 0.6,
        }
    }
}

/// xorshift64*: tiny, seedable, good enough for placing objects.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [0, 1)
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f64()
    }
    /// Standard normal (Box–Muller)
    pub fn normal(&mut self) -> f64 {
        let u1 = self.f64().max(1e-12);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    /// Poisson sample with mean `lambda` (Knuth for small means, normal approximation above 50).
    pub fn poisson(&mut self, lambda: f64) -> f64 {
        if lambda <= 0.0 {
            return 0.0;
        }
        if lambda > 50.0 {
            return (lambda + lambda.sqrt() * self.normal()).round().max(0.0);
        }
        let l = (-lambda).exp();
        let mut k = 0.0;
        let mut p = 1.0;
        loop {
            p *= self.f64();
            if p <= l {
                return k;
            }
            k += 1.0;
        }
    }
}

struct Canvas<'a> {
    v: &'a mut Array3<f32>,
    spec: &'a SampleSpec,
}

impl Canvas<'_> {
    /// Paint a sphere (centre and radius in µm), anti-aliased over one voxel, additively with `max`.
    fn sphere(&mut self, cx: f64, cy: f64, cz: f64, r: f64, value: f32) {
        let s = self.spec;
        let (rx, rz) = (r / s.dx + 1.0, r / s.dz + 1.0);
        let (ix, iy, iz) = (cx / s.dx, cy / s.dx, cz / s.dz);
        let x0 = ((ix - rx).floor().max(0.0)) as usize;
        let x1 = ((ix + rx).ceil().min(s.nx as f64 - 1.0)) as usize;
        let y0 = ((iy - rx).floor().max(0.0)) as usize;
        let y1 = ((iy + rx).ceil().min(s.ny as f64 - 1.0)) as usize;
        let z0 = ((iz - rz).floor().max(0.0)) as usize;
        let z1 = ((iz + rz).ceil().min(s.nz as f64 - 1.0)) as usize;
        if x0 > x1 || y0 > y1 || z0 > z1 {
            return;
        }
        for z in z0..=z1 {
            let dz = (z as f64 - iz) * s.dz;
            for y in y0..=y1 {
                let dy = (y as f64 - iy) * s.dx;
                for x in x0..=x1 {
                    let dx = (x as f64 - ix) * s.dx;
                    let d = (dx * dx + dy * dy + dz * dz).sqrt();
                    let a = (1.0 - (d - r) / s.dx).clamp(0.0, 1.0) as f32;
                    if a > 0.0 {
                        let cell = &mut self.v[[z, y, x]];
                        *cell = cell.max(a * value);
                    }
                }
            }
        }
    }
}

/// Generate the sample volume, shape (nz, ny, nx), values in [0, 1].
pub fn generate_sample(spec: &SampleSpec) -> Result<Array3<f32>, String> {
    if spec.nx == 0 || spec.ny == 0 || spec.nz == 0 {
        return Err("sample size must be positive".into());
    }
    if spec.nx * spec.ny * spec.nz > 64 * 1024 * 1024 {
        return Err("sample is too large (max 64M voxels)".into());
    }
    if !(spec.dx > 0.0) || !(spec.dz > 0.0) {
        return Err("voxel size must be positive".into());
    }
    let mut v = Array3::<f32>::zeros((spec.nz, spec.ny, spec.nx));
    let mut rng = Rng::new(spec.seed);
    let (wx, wy, wz) = (spec.nx as f64 * spec.dx, spec.ny as f64 * spec.dx, spec.nz as f64 * spec.dz);
    let mut canvas = Canvas { v: &mut v, spec };
    let margin = 0.08;

    match spec.kind {
        SampleKind::Beads => {
            for _ in 0..spec.count.max(1) {
                let r = spec.radius * rng.range(0.4, 1.4);
                canvas.sphere(
                    rng.range(wx * margin, wx * (1.0 - margin)),
                    rng.range(wy * margin, wy * (1.0 - margin)),
                    rng.range(wz * margin, wz * (1.0 - margin)),
                    r,
                    rng.range(0.5, 1.0) as f32,
                );
            }
        }
        SampleKind::Filaments => {
            let r = spec.radius.max(spec.dx * 0.6);
            let step = (spec.dx * 0.5).min(r);
            for _ in 0..spec.count.max(1) {
                let mut p = [rng.range(0.0, wx), rng.range(0.0, wy), rng.range(wz * 0.2, wz * 0.8)];
                // random unit direction, mostly lateral
                let mut d = [rng.normal(), rng.normal(), rng.normal() * 0.3];
                let value = rng.range(0.6, 1.0) as f32;
                let steps = ((wx + wy) / step) as usize;
                for _ in 0..steps {
                    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-9);
                    for k in 0..3 {
                        d[k] /= n;
                        p[k] += d[k] * step;
                    }
                    // smooth wandering
                    d[0] += rng.normal() * 0.05;
                    d[1] += rng.normal() * 0.05;
                    d[2] += rng.normal() * 0.015;
                    if p[0] < -r || p[0] > wx + r || p[1] < -r || p[1] > wy + r || p[2] < 0.0 || p[2] > wz {
                        break;
                    }
                    canvas.sphere(p[0], p[1], p[2], r, value);
                }
            }
        }
        SampleKind::Cells => {
            let thick = spec.radius.max(spec.dx);
            for _ in 0..spec.count.clamp(1, 12) {
                let ra = rng.range(0.16, 0.28) * wx.min(wy);
                let rb = ra * rng.range(0.7, 1.0);
                let rc = (ra * rng.range(0.4, 0.8)).min(wz * 0.45);
                let c = [rng.range(ra, wx - ra), rng.range(rb, wy - rb), rng.range(wz * 0.35, wz * 0.65)];
                let ang = rng.range(0.0, std::f64::consts::PI);
                let (sa, ca) = ang.sin_cos();
                let nucleus = [c[0] + rng.range(-0.2, 0.2) * ra, c[1] + rng.range(-0.2, 0.2) * rb, c[2]];
                let nr = [ra * 0.35, rb * 0.35, rc * 0.5];
                paint_shell(canvas.v, spec, c, [ra, rb, rc], (sa, ca), thick, 1.0);
                paint_shell(canvas.v, spec, nucleus, nr, (0.0, 1.0), thick * 0.8, 0.8);
                for _ in 0..6 {
                    let t = rng.range(0.0, std::f64::consts::TAU);
                    let u = rng.range(0.45, 0.85);
                    let px = c[0] + u * ra * t.cos() * ca - u * rb * t.sin() * sa;
                    let py = c[1] + u * ra * t.cos() * sa + u * rb * t.sin() * ca;
                    canvas.sphere(px, py, c[2] + rng.range(-0.3, 0.3) * rc, thick * 1.2, 0.7);
                }
            }
        }
        SampleKind::Lattice => {
            let s = spec.spacing.max(2.0 * spec.dx);
            let r = spec.radius.max(spec.dx * 0.6).min(s * 0.4);
            let nzp = ((wz * 0.6) / s).floor().max(1.0) as usize;
            let mut z = wz * 0.5 - (nzp as f64 - 1.0) * s * 0.5;
            for _ in 0..nzp {
                let mut y = s;
                while y < wy - s * 0.5 {
                    let mut x = s;
                    while x < wx - s * 0.5 {
                        canvas.sphere(x, y, z, r, 1.0);
                        x += s;
                    }
                    y += s;
                }
                z += s;
            }
        }
        SampleKind::Spokes => {
            let spokes = spec.count.clamp(4, 64) as f64;
            let (cx, cy, cz) = (wx / 2.0, wy / 2.0, wz / 2.0);
            let rmax = wx.min(wy) * 0.45;
            let rmin = spec.radius.max(spec.dx);
            let half_t = spec.spacing.max(spec.dz * 0.5) / 2.0; // slab half thickness (µm)
            for z in 0..spec.nz {
                let dzc = (z as f64 * spec.dz - cz).abs();
                let az = (1.0 - (dzc - half_t) / spec.dz).clamp(0.0, 1.0) as f32;
                if az == 0.0 {
                    continue;
                }
                for y in 0..spec.ny {
                    let dy = y as f64 * spec.dx - cy;
                    for x in 0..spec.nx {
                        let dx = x as f64 * spec.dx - cx;
                        let rr = (dx * dx + dy * dy).sqrt();
                        if rr < rmin || rr > rmax {
                            continue;
                        }
                        let phi = dy.atan2(dx);
                        let s = (spokes * phi).sin();
                        // anti-aliased spoke edge: width of one voxel at this radius
                        let edge = (spec.dx / rr) * spokes;
                        let a = ((s / edge) + 0.5).clamp(0.0, 1.0) as f32;
                        if a > 0.0 {
                            canvas.v[[z, y, x]] = canvas.v[[z, y, x]].max(a * az);
                        }
                    }
                }
            }
        }
    }
    Ok(v)
}

/// Ellipsoidal shell centred at `c` with semi-axes `r`, rotated about z by (sin, cos).
fn paint_shell(v: &mut Array3<f32>, spec: &SampleSpec, c: [f64; 3], r: [f64; 3], rot: (f64, f64), thick: f64, value: f32) {
    let (sa, ca) = rot;
    let bx = (r[0].max(r[1]) + thick) / spec.dx + 1.0;
    let bz = (r[2] + thick) / spec.dz + 1.0;
    let (ix, iy, iz) = (c[0] / spec.dx, c[1] / spec.dx, c[2] / spec.dz);
    let x0 = (ix - bx).floor().max(0.0) as usize;
    let x1 = (ix + bx).ceil().min(spec.nx as f64 - 1.0) as usize;
    let y0 = (iy - bx).floor().max(0.0) as usize;
    let y1 = (iy + bx).ceil().min(spec.ny as f64 - 1.0) as usize;
    let z0 = (iz - bz).floor().max(0.0) as usize;
    let z1 = (iz + bz).ceil().min(spec.nz as f64 - 1.0) as usize;
    if x0 > x1 || y0 > y1 || z0 > z1 {
        return;
    }
    for z in z0..=z1 {
        let dz = (z as f64 - iz) * spec.dz;
        for y in y0..=y1 {
            let dy = (y as f64 - iy) * spec.dx;
            for x in x0..=x1 {
                let dx = (x as f64 - ix) * spec.dx;
                // rotate into the ellipsoid frame
                let u = dx * ca + dy * sa;
                let w = -dx * sa + dy * ca;
                let q = ((u / r[0]).powi(2) + (w / r[1]).powi(2) + (dz / r[2]).powi(2)).sqrt();
                // distance to the surface, approximated with the smallest semi-axis
                let d = (q - 1.0).abs() * r[0].min(r[1]).min(r[2]);
                let a = (1.0 - (d - thick / 2.0) / spec.dx).clamp(0.0, 1.0) as f32;
                if a > 0.0 {
                    let cell = &mut v[[z, y, x]];
                    *cell = cell.max(a * value);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generators_fill_the_volume() {
        for kind in [SampleKind::Beads, SampleKind::Filaments, SampleKind::Cells, SampleKind::Lattice, SampleKind::Spokes] {
            let spec = SampleSpec { kind, nx: 48, ny: 40, nz: 16, ..Default::default() };
            let v = generate_sample(&spec).unwrap();
            assert_eq!(v.shape(), &[16, 40, 48]);
            let max = v.iter().copied().fold(0.0_f32, f32::max);
            assert!(max > 0.5, "{kind:?} produced nothing");
            assert!(v.iter().all(|&x| (0.0..=1.0).contains(&x)));
        }
    }

    #[test]
    fn deterministic_for_a_seed() {
        let spec = SampleSpec { seed: 42, nx: 32, ny: 32, nz: 8, ..Default::default() };
        assert_eq!(generate_sample(&spec).unwrap(), generate_sample(&spec).unwrap());
        let other = SampleSpec { seed: 43, ..spec };
        assert_ne!(generate_sample(&spec).unwrap(), generate_sample(&other).unwrap());
    }

    #[test]
    fn poisson_mean() {
        let mut rng = Rng::new(7);
        for lambda in [0.5, 5.0, 200.0] {
            let n = 4000;
            let mean: f64 = (0..n).map(|_| rng.poisson(lambda)).sum::<f64>() / n as f64;
            assert!((mean - lambda).abs() < 0.1 * lambda + 0.1, "lambda {lambda}: mean {mean}");
        }
    }
}
