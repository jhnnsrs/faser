import type { PsfResult } from './use-psf-worker';

/** A float volume in (z, y, x) C-order with its physical size in µm. */
export interface Volume {
  data: Float32Array;
  nx: number;
  ny: number;
  nz: number;
  max: number;
  sizeX: number;
  sizeY: number;
  sizeZ: number;
  /** Compute time in the worker, ms (if known). */
  ms?: number;
}

export function psfToVolume(r: PsfResult): Volume {
  return {
    data: r.data,
    nx: r.nx,
    ny: r.ny,
    nz: r.nz,
    max: r.max,
    sizeX: 2 * r.params.L_obs_XY,
    sizeY: 2 * r.params.L_obs_XY,
    sizeZ: 2 * r.params.L_obs_Z,
    ms: r.ms,
  };
}

/** Voxel size of a PSF grid (µm); the grid spans [-L, L] with N samples. */
export function psfVoxel(r: PsfResult): { dx: number; dz: number } {
  const p = r.params;
  return {
    dx: p.Nxy > 1 ? (2 * p.L_obs_XY) / (p.Nxy - 1) : 2 * p.L_obs_XY,
    dz: p.Nz > 1 ? (2 * p.L_obs_Z) / (p.Nz - 1) : 2 * p.L_obs_Z,
  };
}

/** Index of the brightest voxel, as (z, y, x). */
export function argmax3(v: Pick<Volume, 'data' | 'nx' | 'ny'>): [number, number, number] {
  let best = -Infinity;
  let idx = 0;
  for (let i = 0; i < v.data.length; i++) {
    if (v.data[i] > best) {
      best = v.data[i];
      idx = i;
    }
  }
  const x = idx % v.nx;
  const y = Math.floor(idx / v.nx) % v.ny;
  const z = Math.floor(idx / (v.nx * v.ny));
  return [z, y, x];
}

/**
 * Full width at half maximum along one axis through a voxel, in voxels,
 * with linear interpolation of the half-maximum crossings. NaN if the
 * profile does not drop below half maximum inside the grid.
 */
export function fwhm(v: Volume, axis: 'x' | 'y' | 'z', at: [number, number, number]): number {
  const [z0, y0, x0] = at;
  const n = axis === 'x' ? v.nx : axis === 'y' ? v.ny : v.nz;
  const get = (i: number) => {
    const z = axis === 'z' ? i : z0;
    const y = axis === 'y' ? i : y0;
    const x = axis === 'x' ? i : x0;
    return v.data[(z * v.ny + y) * v.nx + x];
  };
  const c = axis === 'x' ? x0 : axis === 'y' ? y0 : z0;
  const half = get(c) / 2;
  let left = NaN;
  for (let i = c; i > 0; i--) {
    const a = get(i);
    const b = get(i - 1);
    if (b <= half) {
      left = i - (a - half) / Math.max(a - b, 1e-12);
      break;
    }
  }
  let right = NaN;
  for (let i = c; i < n - 1; i++) {
    const a = get(i);
    const b = get(i + 1);
    if (b <= half) {
      right = i + (a - half) / Math.max(a - b, 1e-12);
      break;
    }
  }
  return right - left;
}

/** Element-wise a - b (same shape assumed). */
export function difference(a: Volume, b: Volume): Volume {
  const data = new Float32Array(a.data.length);
  let max = 0;
  for (let i = 0; i < data.length; i++) {
    data[i] = a.data[i] - b.data[i];
    max = Math.max(max, Math.abs(data[i]));
  }
  return { ...a, data, max };
}

export function rms(v: Volume): number {
  let s = 0;
  for (let i = 0; i < v.data.length; i++) s += v.data[i] * v.data[i];
  return Math.sqrt(s / Math.max(v.data.length, 1));
}

/** Synthetic sample description understood by the wasm module (rust/core/src/sample.rs). */
export type SampleKind = 'beads' | 'filaments' | 'cells' | 'lattice' | 'spokes';

export interface SampleSpec {
  kind: SampleKind;
  seed: number;
  nx: number;
  ny: number;
  nz: number;
  dx: number;
  dz: number;
  count: number;
  radius: number;
  spacing: number;
}

export const SAMPLE_KINDS: { value: SampleKind; label: string; description: string }[] = [
  { value: 'beads', label: 'Beads', description: 'Randomly placed spheres of varying size and brightness.' },
  { value: 'filaments', label: 'Filaments', description: 'Smooth random 3-D curves painted as thin tubes, like a cytoskeleton.' },
  { value: 'cells', label: 'Cells', description: 'Ellipsoidal membranes with a nucleus and vesicles inside.' },
  { value: 'lattice', label: 'Point lattice', description: 'A regular grid of points: a resolution target in xy and z.' },
  { value: 'spokes', label: 'Siemens star', description: 'Spokes in the focal plane that get finer towards the centre.' },
];
