'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { PsfResult } from './use-psf-worker';
import { psfToVolume, psfVoxel, SAMPLE_KINDS, type SampleKind, type SampleSpec, type Volume } from './volume';

/** What sits at the focus: a single bead (its image is the PSF) or a simulated sample imaged through the PSF. */
export type FocusMode = 'bead' | 'sample';

export interface ImagingSettings {
  mode: FocusMode;
  /** A synthetic sample kind, or 'image': your own picture as a thin fluorescent slab. */
  kind: SampleKind | 'image';
  seed: number;
  /** Field of view in voxels of the PSF grid. */
  nxy: number;
  nz: number;
  count: number;
  radius: number;
  spacing: number;
  /** Expected photons in the brightest voxel; 0 = no shot noise. */
  photons: number;
  /** Uploaded image as grayscale in [0, 1], resampled to `imageSize`² (top row first). */
  image: Float32Array | null;
  imageSize: number;
  imageName: string | null;
  /** Thickness of the slab the image is painted into (µm). */
  imageThickness: number;
}

export const DEFAULT_IMAGING: ImagingSettings = {
  mode: 'bead',
  kind: 'beads',
  seed: 1,
  nxy: 96,
  nz: 48,
  count: 30,
  radius: 0.15,
  spacing: 0.6,
  photons: 0,
  image: null,
  imageSize: 128,
  imageName: null,
  imageThickness: 0.3,
};

/** Sensible object count / size per sample kind, applied when the kind changes. */
export const KIND_DEFAULTS: Record<SampleKind, Pick<ImagingSettings, 'count' | 'radius' | 'spacing'>> = {
  beads: { count: 30, radius: 0.15, spacing: 0.6 },
  filaments: { count: 6, radius: 0.06, spacing: 0.6 },
  cells: { count: 2, radius: 0.1, spacing: 0.6 },
  lattice: { count: 1, radius: 0.08, spacing: 0.6 },
  spokes: { count: 24, radius: 0.15, spacing: 0.3 },
};

export const SAMPLE_OPTIONS: { value: ImagingSettings['kind']; label: string; description: string }[] = [
  ...SAMPLE_KINDS,
  { value: 'image', label: 'Your image', description: 'A picture of your own (gray = brightness) painted into a thin slab at the focal plane.' },
];

/** Rough cost guard: the padded FFT grid must stay reasonable for one core. */
export const MAX_FFT_VOXELS = 24_000_000;

/** Paint a grayscale picture into a slab of `thickness` µm at the centre of an (nz, nxy, nxy) volume. */
export function imageSample(gray: Float32Array, size: number, s: ImagingSettings, dx: number, dz: number): Volume {
  const { nxy, nz } = s;
  const data = new Float32Array(nz * nxy * nxy);
  const half = Math.max(s.imageThickness / 2, dz / 2);
  let max = 0;
  for (let k = 0; k < nz; k++) {
    const z = (k - (nz - 1) / 2) * dz;
    if (Math.abs(z) > half) continue;
    for (let j = 0; j < nxy; j++) {
      const sj = Math.min(Math.floor((j / nxy) * size), size - 1);
      for (let i = 0; i < nxy; i++) {
        const si = Math.min(Math.floor((i / nxy) * size), size - 1);
        const v = gray[sj * size + si];
        data[(k * nxy + (nxy - 1 - j)) * nxy + i] = v; // image rows top-down, +y up
        if (v > max) max = v;
      }
    }
  }
  return { data, nx: nxy, ny: nxy, nz, max, sizeX: nxy * dx, sizeY: nxy * dx, sizeZ: nz * dz };
}

/**
 * Image formation for the "sample" focus mode: a ground-truth volume on the
 * PSF's voxel grid (synthetic, in the worker, or an uploaded picture) and
 * its image through the current PSF (FFT convolution in the worker, with
 * optional Poisson shot noise).
 */
export function useImaging(
  settings: ImagingSettings,
  psf: PsfResult | null,
  ready: boolean,
  generateSample: (spec: SampleSpec) => Promise<Volume>,
  convolve: (sample: Volume, psf: Volume, photons: number, seed: number) => Promise<Volume>,
) {
  const [sample, setSample] = useState<Volume | null>(null);
  const [image, setImage] = useState<Volume | null>(null);
  const [busy, setBusy] = useState<'sample' | 'image' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const active = settings.mode === 'sample';
  const voxel = psf ? psfVoxel(psf) : null;
  const dx = voxel?.dx ?? 0;
  const dz = voxel?.dz ?? 0;
  const fftVoxels = psf ? (settings.nxy + psf.nx) ** 2 * (settings.nz + psf.nz) : 0;
  const tooLarge = fftVoxels > MAX_FFT_VOXELS;

  // 1. the ground truth follows its settings (and the PSF voxel size): an
  //    uploaded picture is painted synchronously, synthetic kinds come from the worker
  const sampleJob = useRef(0);
  const { kind, seed, nxy, nz, count, radius, spacing, image: picture, imageSize, imageThickness } = settings;
  const pictureSample = useMemo(
    () => (active && kind === 'image' && picture && dx > 0 ? imageSample(picture, imageSize, { ...settings, nxy, nz, imageThickness }, dx, dz) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [active, kind, picture, imageSize, nxy, nz, imageThickness, dx, dz],
  );
  useEffect(() => {
    if (!active || !ready || !(dx > 0) || kind === 'image') return;
    const job = ++sampleJob.current;
    const spec: SampleSpec = { kind, seed, nx: nxy, ny: nxy, nz, dx, dz, count, radius, spacing };
    const id = setTimeout(() => {
      setBusy('sample');
      generateSample(spec)
        .then((v) => {
          if (job !== sampleJob.current) return;
          setSample(v);
          setError(null);
        })
        .catch((e) => job === sampleJob.current && setError(e instanceof Error ? e.message : String(e)))
        .finally(() => job === sampleJob.current && setBusy((b) => (b === 'sample' ? null : b)));
    }, 150);
    return () => clearTimeout(id);
  }, [active, ready, dx, dz, kind, seed, nxy, nz, count, radius, spacing, picture, imageSize, imageThickness, generateSample]);

  // 2. the image follows sample + PSF (+ noise)
  const truth = kind === 'image' ? pictureSample : sample;
  const imageJob = useRef(0);
  const runConvolution = useCallback(() => {
    if (!active || !truth || !psf || tooLarge) return;
    const job = ++imageJob.current;
    setBusy('image');
    convolve(truth, psfToVolume(psf), settings.photons, settings.seed)
      .then((v) => {
        if (job !== imageJob.current) return;
        setImage(v);
        setError(null);
      })
      .catch((e) => job === imageJob.current && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => job === imageJob.current && setBusy((b) => (b === 'image' ? null : b)));
  }, [active, truth, psf, tooLarge, convolve, settings.photons, settings.seed]);
  useEffect(() => {
    const id = setTimeout(runConvolution, 150);
    return () => clearTimeout(id);
  }, [runConvolution]);

  return {
    sample: active ? truth : null,
    image: active && truth ? image : null,
    busy,
    error,
    tooLarge,
    voxel,
  };
}
