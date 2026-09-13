//! Image formation: 3-D convolution of a sample with a PSF (FFT based, linear
//! convolution with zero padding, "same" output size), plus optional Poisson
//! shot noise.

use ndarray::Array3;
use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

use crate::sample::Rng;

/// Smallest n >= v whose prime factors are all in {2, 3, 5} (fast FFT sizes).
fn next_fast(v: usize) -> usize {
    let mut n = v.max(1);
    loop {
        let mut m = n;
        for p in [2, 3, 5] {
            while m % p == 0 {
                m /= p;
            }
        }
        if m == 1 {
            return n;
        }
        n += 1;
    }
}

/// In-place complex 3-D FFT of a (nz, ny, nx) C-order buffer along all axes.
fn fft3(buf: &mut [Complex32], dims: (usize, usize, usize), inverse: bool) {
    let (nz, ny, nx) = dims;
    let mut planner = FftPlanner::<f32>::new();
    let mut plan = |n: usize| -> Arc<dyn Fft<f32>> {
        if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) }
    };
    // x lines are contiguous
    let fx = plan(nx);
    let mut scratch = vec![Complex32::default(); fx.get_inplace_scratch_len()];
    for line in buf.chunks_exact_mut(nx) {
        fx.process_with_scratch(line, &mut scratch);
    }
    // y and z are strided: gather, transform, scatter
    let fy = plan(ny);
    let mut scratch = vec![Complex32::default(); fy.get_inplace_scratch_len()];
    let mut line = vec![Complex32::default(); ny];
    for z in 0..nz {
        for x in 0..nx {
            for y in 0..ny {
                line[y] = buf[(z * ny + y) * nx + x];
            }
            fy.process_with_scratch(&mut line, &mut scratch);
            for y in 0..ny {
                buf[(z * ny + y) * nx + x] = line[y];
            }
        }
    }
    let fz = plan(nz);
    let mut scratch = vec![Complex32::default(); fz.get_inplace_scratch_len()];
    let mut line = vec![Complex32::default(); nz];
    for y in 0..ny {
        for x in 0..nx {
            for z in 0..nz {
                line[z] = buf[(z * ny + y) * nx + x];
            }
            fz.process_with_scratch(&mut line, &mut scratch);
            for z in 0..nz {
                buf[(z * ny + y) * nx + x] = line[z];
            }
        }
    }
}

/// Linear convolution of `sample` with `psf`, output the size of `sample`
/// ("same" mode, the PSF centre voxel at (nz/2, ny/2, nx/2) maps to no shift).
/// The PSF is normalized to unit sum so intensities are preserved.
pub fn convolve3d(sample: &Array3<f32>, psf: &Array3<f32>) -> Result<Array3<f32>, String> {
    let (sz, sy, sx) = sample.dim();
    let (kz, ky, kx) = psf.dim();
    if sz == 0 || sy == 0 || sx == 0 || kz == 0 || ky == 0 || kx == 0 {
        return Err("empty volume".into());
    }
    let dims = (next_fast(sz + kz - 1), next_fast(sy + ky - 1), next_fast(sx + kx - 1));
    let (pz, py, px) = dims;
    let n = pz * py * px;
    if n > 256 * 1024 * 1024 {
        return Err("convolution too large".into());
    }

    let mut a = vec![Complex32::default(); n];
    for ((z, y, x), &v) in sample.indexed_iter() {
        a[(z * py + y) * px + x] = Complex32::new(v, 0.0);
    }
    let psf_sum: f64 = psf.iter().map(|&v| v as f64).sum();
    if !(psf_sum > 0.0) {
        return Err("PSF has zero energy".into());
    }
    let mut b = vec![Complex32::default(); n];
    for ((z, y, x), &v) in psf.indexed_iter() {
        b[(z * py + y) * px + x] = Complex32::new((v as f64 / psf_sum) as f32, 0.0);
    }

    fft3(&mut a, dims, false);
    fft3(&mut b, dims, false);
    for (u, v) in a.iter_mut().zip(&b) {
        *u *= *v;
    }
    fft3(&mut a, dims, true);

    let scale = 1.0 / n as f32;
    let (oz, oy, ox) = (kz / 2, ky / 2, kx / 2);
    let mut out = Array3::<f32>::zeros((sz, sy, sx));
    for ((z, y, x), o) in out.indexed_iter_mut() {
        *o = (a[((z + oz) * py + (y + oy)) * px + (x + ox)].re * scale).max(0.0);
    }
    Ok(out)
}

/// Add Poisson shot noise: the brightest voxel receives `photons` expected
/// counts on average. The result is in photon counts. `photons <= 0` is a no-op.
pub fn add_shot_noise(image: &mut Array3<f32>, photons: f64, seed: u64) {
    if photons <= 0.0 {
        return;
    }
    let max = image.iter().copied().fold(0.0_f32, f32::max) as f64;
    if max <= 0.0 {
        return;
    }
    let mut rng = Rng::new(seed ^ 0xA5A5_5A5A_1234_5678);
    let k = photons / max;
    for v in image.iter_mut() {
        *v = rng.poisson(*v as f64 * k) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_sizes() {
        assert_eq!(next_fast(7), 8);
        assert_eq!(next_fast(97), 100);
        assert_eq!(next_fast(128), 128);
    }

    #[test]
    fn delta_reproduces_the_kernel() {
        // A unit impulse at the sample centre imaged through a kernel gives the
        // kernel (normalized) centred at the impulse.
        let mut sample = Array3::<f32>::zeros((9, 11, 13));
        sample[[4, 5, 6]] = 2.0;
        let mut psf = Array3::<f32>::zeros((5, 5, 5));
        for ((z, y, x), v) in psf.indexed_iter_mut() {
            *v = (1 + z + 2 * y + 3 * x) as f32;
        }
        let sum: f32 = psf.iter().sum();
        let out = convolve3d(&sample, &psf).unwrap();
        assert_eq!(out.dim(), (9, 11, 13));
        for ((z, y, x), &v) in psf.indexed_iter() {
            let got = out[[4 + z - 2, 5 + y - 2, 6 + x - 2]];
            assert!((got - 2.0 * v / sum).abs() < 1e-4, "at {:?}: {got} vs {}", (z, y, x), 2.0 * v / sum);
        }
        // energy preserved
        let total: f32 = out.iter().sum();
        assert!((total - 2.0).abs() < 1e-3);
    }

    #[test]
    fn noise_scales_to_photons() {
        let mut img = Array3::<f32>::from_elem((4, 8, 8), 1.0);
        add_shot_noise(&mut img, 1000.0, 3);
        let mean = img.iter().sum::<f32>() / img.len() as f32;
        assert!((mean - 1000.0).abs() < 20.0, "mean {mean}");
    }
}
