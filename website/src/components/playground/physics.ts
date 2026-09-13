/**
 * Optical quantities derived from the parameters, mirrored from
 * rust/core/src/params.rs so the page can size the simulation grid and draw
 * the microscope synchronously, without a round trip to the worker.
 */
import type { Params } from './params';
import { zernikeRange } from './zernike';

export const TWO_PI = 2 * Math.PI;

export function k0(p: Params): number {
  return TWO_PI / p.Wavelength;
}

/** Semi-aperture angle in the immersion medium (rad). */
export function alpha(p: Params): number {
  return Math.asin(Math.min(p.NA / p.n1, 1));
}

function tWind(p: Params): number {
  return p.Window === 'CUSTOM' ? p.Wind_Depth * 1e3 : 2.23e3;
}

function rWind(p: Params): number {
  return p.Window === 'NO' ? 100 * tWind(p) : p.Wind_Radius * 1e3;
}

/** Semi-aperture after clipping by the cranial window (rad). */
export function alphaEff(p: Params): number {
  return Math.min(Math.atan(rWind(p) / tWind(p)), alpha(p));
}

export function naEff(p: Params): number {
  return Math.min(p.n1 * Math.sin(alphaEff(p)), p.NA);
}

/** Pupil radius (µm). */
export function r0(p: Params): number {
  return p.WD * Math.sin(alpha(p));
}

export function alphaInt(p: Params): number {
  return alphaEff(p) + Math.abs((p.Tilt * Math.PI) / 180);
}

/** Marginal-ray angle in the coverslip / sample (rad), clamped at grazing. */
export function alpha2Eff(p: Params): number {
  return Math.asin(Math.min((p.n1 / p.n2) * Math.sin(alphaEff(p)), 1));
}

export function alpha3Eff(p: Params): number {
  return Math.asin(Math.min((p.n1 / p.n3) * Math.sin(alphaEff(p)), 1));
}

/**
 * Axial position of the aberrated focus relative to the nominal one (µm):
 * the balanced defocus of the coverslip/depth aberration (see `Params::dfoc`).
 */
export function dfoc(p: Params): number {
  const N = 2001;
  const aEff = alphaEff(p);
  const { Thickness: d, Collar: dDesign, Depth: z, n1, n2, n3 } = p;
  let sw = 0, sg = 0, sgg = 0, sp = 0, spg = 0;
  for (let i = 0; i < N; i++) {
    const theta = (aEff * i) / (N - 1);
    const s1 = Math.sin(theta);
    const c1 = Math.cos(theta);
    const w = s1;
    const s2 = (n1 / n2) * s1;
    const c2 = Math.sqrt(Math.max(1 - s2 * s2, 0));
    const s3 = (n1 / n3) * s1;
    const c3 = Math.sqrt(Math.max(1 - s3 * s3, 0));
    const psi = n3 * z * c3 + n2 * (d - dDesign) * c2 - n1 * (d + z) * c1 + n1 * dDesign * c1;
    const g = n3 * c3;
    sw += w;
    sg += w * g;
    sgg += w * g * g;
    sp += w * psi;
    spg += w * psi * g;
  }
  const meanG = sg / sw;
  const varG = sgg / sw - meanG * meanG;
  if (varG === 0) return 0;
  const cov = spg / sw - (sp / sw) * meanG;
  return -cov / varG;
}

// ---------------------------------------------------------------------------
// Automatic simulation grid
// ---------------------------------------------------------------------------

export interface Grid {
  Nxy: number;
  Nz: number;
  Ntheta: number;
  Nphi: number;
}

export const GRID_LIMITS: Record<keyof Grid, [number, number]> = {
  Nxy: [16, 256],
  Nz: [5, 128],
  Ntheta: [16, 400],
  Nphi: [16, 320],
};

/** Quadrature points per 2π of integrand phase along θ, at the fastest-varying point. */
export const THETA_SAMPLES_PER_CYCLE = 8;
/**
 * The same for the w = cosθ₃ segment of the adaptive rule: the phase is
 * linear there and the integrand vanishes towards the critical angle, so
 * fewer points per cycle reach the same accuracy.
 */
export const THETA_SAMPLES_PER_CYCLE_W = 5;
/** Voxels per λ/(2 NA) lateral resolution unit (2 is Nyquist). */
export const XY_OVERSAMPLING = 5;
/** Voxels per axial resolution unit λ/(2 n₃(1 − cos α₃)). */
export const Z_OVERSAMPLING = 4;
/** Mirrors `ADAPTIVE_SPLIT` / `ADAPTIVE_DENSITY` in rust/core/src/lib.rs. */
const ADAPTIVE_SPLIT = 0.6;
const ADAPTIVE_DENSITY = 2;

function clampRound(v: number, [lo, hi]: [number, number], multiple = 1): number {
  const r = Math.ceil(v / multiple) * multiple;
  return Math.max(lo, Math.min(hi, r));
}

/**
 * Phase of the Debye integrand as a function of θ for the four worst-case
 * voxels of the observation volume (±corner laterally, ±L_z axially): the
 * lateral plane-wave term k₀ n₁ sinθ·ρ, the axial term k₀ n₃ cosθ₃·z with
 * the focus shift, the coverslip / depth aberration (`psi_w`) and the
 * coverslip path inside the Fresnel factor (`beta`). Evanescent angles add
 * no phase (real part of cosθ₃).
 */
function phaseFunction(p: Params): (theta: number) => [number, number, number, number] {
  const k = k0(p);
  const rho = Math.SQRT2 * p.L_obs_XY;
  const shift = dfoc(p);
  const { n1, n2, n3, Thickness: t, Depth: depth, Collar: collar } = p;
  const zp = p.L_obs_Z + shift;
  const zm = -p.L_obs_Z + shift;
  return (theta) => {
    const s1 = Math.sin(theta);
    const c1 = Math.cos(theta);
    const s2 = (n1 / n2) * s1;
    const c2 = Math.sqrt(Math.max(1 - s2 * s2, 0));
    const s3 = (n1 / n3) * s1;
    const c3 = Math.sqrt(Math.max(1 - s3 * s3, 0));
    const lateral = k * n1 * s1 * rho;
    const ab = k * (n3 * depth * c3 - n1 * (t + depth) * c1 + n1 * collar * c1 + n2 * (t - collar) * c2);
    return [lateral + ab + k * n3 * c3 * zp, lateral + ab + k * n3 * c3 * zm, -lateral + ab + k * n3 * c3 * zp, -lateral + ab + k * n3 * c3 * zm];
  };
}


/**
 * Number of θ quadrature nodes that resolve the integrand: for each segment
 * the simulator samples uniformly in a variable v (θ itself, or w = cosθ₃
 * for the adaptive rule below the critical angle), THETA_SAMPLES_PER_CYCLE
 * points per 2π at the fastest phase rate max|dΦ/dv| found on that segment.
 * For the adaptive rule the total is chosen so that the fixed allocation in
 * `theta_nodes` (rust/core) gives every segment at least what it needs.
 */
export function autoNtheta(p: Params, adaptive: boolean, extraPhase = 0): number {
  return thetaBudget(p, adaptive, extraPhase).total;
}

/**
 * The per-segment node needs behind `autoNtheta` (for diagnostics).
 * `extraPhase` is additional pupil phase range (rad) not in `p`, e.g. the
 * Zernike layer displayed on the SLM.
 */
export function thetaBudget(p: Params, adaptive: boolean, extraPhase = 0): { n1: number; n2: number; n3: number; total: number } {
  const S = THETA_SAMPLES_PER_CYCLE;
  const aMax = alphaInt(p);
  const ratio = p.n1 / p.n3;
  const critical = ratio > 1 ? Math.asin(1 / ratio) : Infinity;
  const phase = phaseFunction(p);
  const zern = zernikeRange(p) + extraPhase;
  const M = 512;
  // Beyond the critical angle the wave is evanescent in the sample and decays
  // over the distance from the interface to the nearest voxel; its phase
  // (which keeps oscillating with cos θ) only matters as far as it survives.
  const k = k0(p);
  const dMin = Math.max(p.Depth + dfoc(p) - p.L_obs_Z, 0);
  const weight = (theta: number) => {
    const s3 = ratio * Math.sin(theta);
    return s3 <= 1 ? 1 : Math.exp(-k * p.n3 * Math.sqrt(s3 * s3 - 1) * dMin);
  };

  // nodes needed on [v0, v1] sampled uniformly in v; θ = thetaOf(v)
  const need = (v0: number, v1: number, thetaOf: (v: number) => number, byVariation = false, samples = S) => {
    const h = (v1 - v0) / M;
    if (!(h > 0)) return 0;
    let prev = phase(thetaOf(v0));
    let maxRate = 0;
    const tv = [0, 0, 0, 0];
    for (let i = 1; i <= M; i++) {
      const cur = phase(thetaOf(v0 + i * h));
      const w = weight(thetaOf(v0 + (i - 0.5) * h));
      for (let q = 0; q < 4; q++) {
        const d = Math.abs(cur[q] - prev[q]) * w;
        tv[q] += d;
        maxRate = Math.max(maxRate, d / h);
      }
      prev = cur;
    }
    // Zernike terms vary over the pupil radius ~ sin θ; spread over the segment
    const zernRate = (zern * ((v1 - v0) / aMax)) / (v1 - v0);
    const cycles = byVariation ? (Math.max(...tv) + zern * ((v1 - v0) / aMax)) / TWO_PI : ((maxRate + zernRate) * (v1 - v0)) / TWO_PI;
    return samples * cycles;
  };

  if (!adaptive || critical >= aMax) {
    // Uniform θ. Through a critical angle the phase rate diverges: budget by
    // total variation instead (the rule cannot converge there anyway).
    const n1 = need(0, aMax, (v) => v, critical < aMax);
    return { n1, n2: 0, n3: 0, total: clampRound(n1 + 8, GRID_LIMITS.Ntheta, 4) };
  }
  const thetaA = ADAPTIVE_SPLIT * critical;
  const len = thetaA + ADAPTIVE_DENSITY * (critical - thetaA) + (aMax - critical);
  const wA = Math.sqrt(Math.max(1 - (ratio * Math.sin(thetaA)) ** 2, 0));
  const n1 = need(0, thetaA, (v) => v);
  const n2 = need(0, wA, (w) => Math.asin(Math.min(Math.sqrt(1 - w * w) / ratio, 1)), false, THETA_SAMPLES_PER_CYCLE_W);
  const n3 = need(critical, aMax, (v) => v);
  const n =
    Math.max((n1 * len) / thetaA, (n2 * len) / (ADAPTIVE_DENSITY * (critical - thetaA)), (n3 * len) / Math.max(aMax - critical, 1e-6), 24) + 8;
  return { n1, n2, n3, total: clampRound(n, GRID_LIMITS.Ntheta, 4) };
}

/**
 * The grid that resolves the PSF the parameters imply:
 * - Nxy: XY_OVERSAMPLING voxels per λ/(2 NA), the lateral resolution unit;
 * - Nz: the same voxel size as XY unless the axial band limit allows coarser;
 * - Ntheta: see `autoNtheta`;
 * - Nphi: above the highest azimuthal harmonic present at the field corner
 *   (Bessel bound k·ρ + 3·(k·ρ)^⅓) plus the vortex charge and the
 *   polarization terms, so the trapezoid rule is exact and no aliased ghost
 *   foci land inside the field; a tilted coverslip cuts the aperture
 *   non-circularly in φ, which needs more.
 */
export function autoGrid(p: Params, extraPhase = 0): Grid {
  const na = Math.max(naEff(p), 0.05);
  const voxXy = p.Wavelength / (2 * na) / XY_OVERSAMPLING;
  const Nxy = clampRound((2 * p.L_obs_XY) / voxXy + 1, GRID_LIMITS.Nxy, 4);

  const axialBand = Math.max(p.n3 * (1 - Math.cos(alpha3Eff(p))), 1e-3);
  const voxZ = Math.min(p.Wavelength / (2 * axialBand) / Z_OVERSAMPLING, 1.5 * voxXy);
  const Nz = Math.min(clampRound((2 * p.L_obs_Z) / voxZ + 1, GRID_LIMITS.Nz, 1) | 1, GRID_LIMITS.Nz[1]);

  const Ntheta = autoNtheta(p, p.Theta_sampling === 'ADAPTIVE', extraPhase);

  const x = k0(p) * na * Math.SQRT2 * p.L_obs_XY;
  const tiltFactor = 1 + (3 * Math.abs((p.Tilt * Math.PI) / 180)) / Math.max(alphaEff(p), 0.1);
  const harmonics = (x + 3 * Math.cbrt(x) + 2 * Math.abs(p.VC) + 12) * tiltFactor;
  const Nphi = clampRound(harmonics, GRID_LIMITS.Nphi, 8);

  return { Nxy, Nz, Ntheta, Nphi };
}

/** A cheaper grid with the same extent, for live previews while dragging. */
export function previewGrid(g: Grid): Grid {
  return {
    Nxy: Math.max(24, Math.round(g.Nxy / 2 / 4) * 4),
    Nz: Math.max(7, Math.round(g.Nz / 2) | 1),
    Ntheta: Math.max(20, Math.round(g.Ntheta / 2 / 4) * 4),
    Nphi: Math.max(24, Math.round(g.Nphi / 2 / 8) * 8),
  };
}

/** Work units of a grid; compute time is roughly proportional (see `CostModel`). */
export function gridCost(g: Grid, p: Pick<Params, 'Mode'>): number {
  return g.Ntheta * (g.Nphi + g.Nz) * g.Nxy * g.Nxy * (p.Mode === 'DONUT BOTTLE' ? 2 : 1);
}
