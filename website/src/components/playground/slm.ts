/**
 * The spatial light modulator: a pixelated phase panel conjugate to the back
 * pupil. `SlmDesign` is what the user edits (a pattern generator and the
 * device properties); `buildSlm` turns it into the `SLM` block of the
 * parameters (the pixel phases the simulator samples, see `Slm` in
 * rust/core/src/lib.rs). Sampling, wrapping and quantization here mirror the
 * Rust `Slm::phase_at` exactly so the pupil preview matches the simulation.
 */
import { ZERNIKE_KEYS, ZERO_ZERNIKE, type Slm, type ZernikeCoeffs } from './params';
import { zernikePhase } from './zernike';

export type SlmPattern = 'flat' | 'vortex' | 'bottle' | 'halfmoon' | 'grating' | 'axicon' | 'image' | 'custom';

export interface SlmDesign {
  enabled: boolean;
  pattern: SlmPattern;
  /** Pixels per side. */
  n: number;
  /** Phase levels (0 = continuous, 256 = 8-bit). */
  levels: number;
  /** Fraction of the pixel area that modulates (dead space passes unmodulated). */
  fillFactor: number;
  /** Panel half-width in units of the pupil radius. */
  extent: number;
  /** Panel misalignment in units of the pupil radius. */
  offsetX: number;
  offsetY: number;
  /** Vortex topological charge. */
  charge: number;
  /** Bottle: inner disc radius (units of r0) and its phase step in units of π. */
  ringRadius: number;
  ringStep: number;
  /** Half-moon / grating orientation (degrees). */
  angle: number;
  /** Blazed grating period in units of r0. */
  period: number;
  /** Axicon: waves of phase from the centre to r0. */
  cone: number;
  /** Imported image (grayscale 0..1, n×n) and its phase range in waves. */
  image: Float32Array | null;
  imageName: string | null;
  imageWaves: number;
  /** Raw phases for `pattern = 'custom'` (loaded from a config), n×n. */
  custom: Float32Array | null;
  /**
   * Zernike modes displayed on top of the pattern (adaptive optics), in
   * radians on the unit pupil like the system aberrations. On an ideal panel
   * this is identical to the system aberrations; a real panel adds its
   * pixelation, quantization and fill-factor effects.
   */
  zernike: ZernikeCoeffs;
}

export function hasZernike(z: ZernikeCoeffs): boolean {
  return ZERNIKE_KEYS.some((k) => z[k] !== 0);
}

export function negatedZernike(z: ZernikeCoeffs): ZernikeCoeffs {
  const out = { ...ZERO_ZERNIKE };
  for (const k of ZERNIKE_KEYS) out[k] = -z[k];
  return out;
}

export const SLM_SIZES = [32, 64, 128, 256, 512];
export const SLM_LEVELS: { value: number; label: string }[] = [
  { value: 0, label: 'Continuous' },
  { value: 256, label: '8-bit (256)' },
  { value: 64, label: '6-bit (64)' },
  { value: 16, label: '4-bit (16)' },
  { value: 4, label: '2-bit (4)' },
  { value: 2, label: 'Binary (2)' },
];

export const SLM_PATTERNS: { value: SlmPattern; label: string; description: string }[] = [
  { value: 'flat', label: 'Flat', description: 'No modulation; the SLM only adds its pixelation and dead-space leakage.' },
  { value: 'vortex', label: 'Vortex', description: 'Helical phase (charge m): a donut focus, the lateral STED depletion pattern.' },
  { value: 'bottle', label: 'Bottle (π disc)', description: 'A phase step on the inner disc of the pupil: a dark focus enclosed axially.' },
  { value: 'halfmoon', label: 'Half-moon', description: 'A π step across a line through the pupil: two lobes side by side, the 1-D STED pattern.' },
  { value: 'grating', label: 'Blazed grating', description: 'A linear phase ramp steers the focus sideways by λ / (period · sin α).' },
  { value: 'axicon', label: 'Axicon', description: 'A conical phase turns the focus into a ring of radius λ · waves / sin α.' },
  { value: 'image', label: 'Image', description: 'Your own bitmap: gray 0 → 0, white → the chosen number of waves.' },
  { value: 'custom', label: 'Loaded pattern', description: 'Pixel phases loaded from a config file.' },
];

export const DEFAULT_SLM_DESIGN: SlmDesign = {
  enabled: false,
  pattern: 'vortex',
  n: 256,
  levels: 256,
  fillFactor: 0.93,
  extent: 1.0,
  offsetX: 0,
  offsetY: 0,
  charge: 1,
  ringRadius: 0.707,
  ringStep: 1,
  angle: 0,
  period: 1,
  cone: 1,
  image: null,
  imageName: null,
  imageWaves: 1,
  custom: null,
  zernike: { ...ZERO_ZERNIKE },
};

/** Unwrapped phase (rad) of a pattern at pupil coordinates (u, v) in units of r0. */
export function patternPhase(d: SlmDesign, u: number, v: number): number {
  switch (d.pattern) {
    case 'flat':
      return 0;
    case 'vortex':
      return d.charge * Math.atan2(v, u);
    case 'bottle':
      return Math.hypot(u, v) <= d.ringRadius ? d.ringStep * Math.PI : 0;
    case 'halfmoon': {
      const a = (d.angle * Math.PI) / 180;
      return u * Math.cos(a) + v * Math.sin(a) > 0 ? Math.PI : 0;
    }
    case 'grating': {
      const a = (d.angle * Math.PI) / 180;
      return (2 * Math.PI * (u * Math.cos(a) + v * Math.sin(a))) / Math.max(d.period, 1e-3);
    }
    case 'axicon':
      return 2 * Math.PI * d.cone * Math.hypot(u, v);
    default:
      return 0;
  }
}

/** Pixel centre of (i, j) on an n-pixel panel, in units of r0. */
export function pixelCenter(d: Pick<SlmDesign, 'n' | 'extent'>, i: number, j: number): [number, number] {
  return [(((i + 0.5) / d.n) * 2 - 1) * d.extent, (1 - ((j + 0.5) / d.n) * 2) * d.extent];
}

/** The n×n pixel phases (row-major, first row at +y) for a design: the pattern plus the Zernike layer. */
export function designPhases(d: SlmDesign): Float32Array {
  const n = d.n;
  const out = new Float32Array(n * n);
  if (d.pattern === 'custom' && d.custom && d.custom.length === n * n) {
    out.set(d.custom);
  } else if (d.pattern === 'image') {
    if (d.image && d.image.length === n * n) {
      const scale = 2 * Math.PI * d.imageWaves;
      for (let k = 0; k < n * n; k++) out[k] = d.image[k] * scale;
    }
  } else if (d.pattern !== 'flat') {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const [u, v] = pixelCenter(d, i, j);
        out[j * n + i] = patternPhase(d, u, v);
      }
    }
  }
  if (hasZernike(d.zernike)) {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const [u, v] = pixelCenter(d, i, j);
        out[j * n + i] += zernikePhase(d.zernike, u, v);
      }
    }
  }
  return out;
}

/** The `SLM` parameter block for a design (null when disabled). */
export function buildSlm(d: SlmDesign): Slm | null {
  if (!d.enabled) return null;
  const phase = Array.from(designPhases(d), (v) => Math.round(v * 1e4) / 1e4);
  return {
    n: d.n,
    phase,
    extent: d.extent,
    levels: d.levels,
    fill_factor: d.fillFactor,
    offset_x: d.offsetX,
    offset_y: d.offsetY,
  };
}

/**
 * A design that reproduces an `SLM` block loaded from a config file. The
 * pixel phases carry pattern and Zernike layer baked together, so the loaded
 * design is a `custom` pattern with an empty layer.
 */
export function designFromSlm(s: Slm, base: SlmDesign = DEFAULT_SLM_DESIGN): SlmDesign {
  return {
    ...base,
    enabled: true,
    pattern: 'custom',
    zernike: { ...ZERO_ZERNIKE },
    n: s.n,
    levels: s.levels,
    fillFactor: s.fill_factor,
    extent: s.extent,
    offsetX: s.offset_x,
    offsetY: s.offset_y,
    custom: Float32Array.from(s.phase),
  };
}

/** Wrap to [0, 2π) and quantize like the device (mirrors `Slm::phase_at`). */
export function displayPhase(raw: number, levels: number): number {
  const wrapped = ((raw % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  if (levels > 1) return Math.min(Math.floor((wrapped / (2 * Math.PI)) * levels), levels - 1) * ((2 * Math.PI) / levels);
  return wrapped;
}

/** Displayed phase of the pixel under (u, v) in units of r0, or null outside the panel (mirrors `Slm::phase_at`). */
export function slmPhaseAt(s: Slm, phase: ArrayLike<number>, u: number, v: number): number | null {
  const fu = ((u - s.offset_x) / s.extent + 1) * 0.5;
  const fv = (1 - (v - s.offset_y) / s.extent) * 0.5;
  if (fu < 0 || fu >= 1 || fv < 0 || fv >= 1) return null;
  const i = Math.min(Math.floor(fu * s.n), s.n - 1);
  const j = Math.min(Math.floor(fv * s.n), s.n - 1);
  return displayPhase(phase[j * s.n + i], s.levels);
}

/**
 * Complex transmission of the panel at (u, v) as [amplitude, phase]: the
 * pixel phase weighted by the fill factor plus the unmodulated dead-space
 * fraction (mirrors `Slm::factor_at`); [1, 0] outside the panel.
 */
export function slmFactorAt(s: Slm, phase: ArrayLike<number>, u: number, v: number): [number, number] {
  const ph = slmPhaseAt(s, phase, u, v);
  if (ph == null) return [1, 0];
  const re = s.fill_factor * Math.cos(ph) + (1 - s.fill_factor);
  const im = s.fill_factor * Math.sin(ph);
  return [Math.hypot(re, im), Math.atan2(im, re)];
}

/**
 * Draw the panel as the device shows it: the wrapped, quantized phase as a
 * gray ramp (0 = black, 2π = white) like an SLM bitmap, at pixel resolution,
 * with the pupil circle overlaid.
 */
export function drawSlmPanel(canvas: HTMLCanvasElement, s: Slm, phase: ArrayLike<number>, size = 256) {
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const img = ctx.createImageData(size, size);
  const px = img.data;
  const twoPi = 2 * Math.PI;
  for (let j = 0; j < size; j++) {
    const sj = Math.min(Math.floor((j / size) * s.n), s.n - 1);
    for (let i = 0; i < size; i++) {
      const si = Math.min(Math.floor((i / size) * s.n), s.n - 1);
      const g = Math.round((displayPhase(phase[sj * s.n + si], s.levels) / twoPi) * 255);
      const k = (j * size + i) * 4;
      px[k] = g;
      px[k + 1] = g;
      px[k + 2] = g;
      px[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  // pixel grid hint when coarse
  if (s.n <= 64) {
    ctx.strokeStyle = 'rgba(0,0,0,0.25)';
    ctx.lineWidth = 1;
    const step = size / s.n;
    ctx.beginPath();
    for (let k = 1; k < s.n; k++) {
      ctx.moveTo(k * step, 0);
      ctx.lineTo(k * step, size);
      ctx.moveTo(0, k * step);
      ctx.lineTo(size, k * step);
    }
    ctx.stroke();
  }
  // the pupil on the panel
  const r = (size / 2) / s.extent;
  const cx = size / 2 + (s.offset_x * size) / 2 / s.extent;
  const cy = size / 2 - (s.offset_y * size) / 2 / s.extent;
  ctx.strokeStyle = 'rgba(255,120,220,0.9)';
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.setLineDash([]);
}

/** Load an image file as an n×n grayscale array in [0, 1] (luma, top row first). */
export async function imageToGray(file: File, n: number): Promise<Float32Array> {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement('canvas');
  canvas.width = n;
  canvas.height = n;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('canvas not available');
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(bitmap, 0, 0, n, n);
  bitmap.close();
  const { data } = ctx.getImageData(0, 0, n, n);
  const out = new Float32Array(n * n);
  for (let k = 0; k < n * n; k++) {
    out[k] = (0.2126 * data[k * 4] + 0.7152 * data[k * 4 + 1] + 0.0722 * data[k * 4 + 2]) / 255;
  }
  return out;
}

/** Resample a square gray array to a new side length (nearest neighbour). */
export function resampleSquare(src: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return src;
  const out = new Float32Array(to * to);
  for (let j = 0; j < to; j++) {
    const sj = Math.min(Math.floor((j / to) * from), from - 1);
    for (let i = 0; i < to; i++) {
      const si = Math.min(Math.floor((i / to) * from), from - 1);
      out[j * to + i] = src[sj * from + si];
    }
  }
  return out;
}
