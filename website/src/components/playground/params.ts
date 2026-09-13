/**
 * The simulation parameters, field-for-field the Python `PSFConfig` model
 * (and therefore the `psf_config.json` the CLI / napari plugin write). The
 * wasm module deserializes exactly this object.
 */

export type Mode = 'GAUSSIAN' | 'DONUT' | 'BOTTLE' | 'DONUT BOTTLE' | 'LOADED';
export type YesNo = 'YES' | 'NO';
export type Window = 'NO' | 'CUSTOM';
/** 1 = elliptical, 2 = radial, 3 = azimuthal */
export type Polarization = 1 | 2 | 3;
/** θ quadrature; ADAPTIVE resolves the critical angle of the sample (see rust/core). */
export type ThetaSampling = 'UNIFORM' | 'ADAPTIVE';

/**
 * A spatial light modulator conjugate to the back pupil: `n × n` pixel
 * phases (radians, row-major, first row at +y) over the square
 * [-extent·r0, extent·r0]², wrapped to 2π and quantized to `levels` when
 * sampled (see `Slm` in rust/core/src/lib.rs and ./slm.ts).
 */
export interface Slm {
  n: number;
  phase: number[];
  extent: number;
  levels: number;
  fill_factor: number;
  offset_x: number;
  offset_y: number;
}

export const ZERNIKE_KEYS = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a12', 'a24'] as const;
export type ZernikeKey = (typeof ZERNIKE_KEYS)[number];
/** Zernike coefficients in radians of phase (the same modes as the simulator). */
export type ZernikeCoeffs = Record<ZernikeKey, number>;
export const ZERO_ZERNIKE: ZernikeCoeffs = { a0: 0, a1: 0, a2: 0, a3: 0, a4: 0, a5: 0, a6: 0, a7: 0, a8: 0, a9: 0, a12: 0, a24: 0 };

export interface Params {
  L_obs_XY: number;
  L_obs_Z: number;
  Nxy: number;
  Nz: number;
  Ntheta: number;
  Nphi: number;
  Normalize: YesNo;
  NA: number;
  WD: number;
  n1: number;
  n2: number;
  n3: number;
  Thickness: number;
  Collar: number;
  Depth: number;
  Tilt: number;
  Window: Window;
  Wind_Radius: number;
  Wind_Depth: number;
  Wind_Offset_x: number;
  Wind_Offset_y: number;
  a0: number;
  a1: number;
  a2: number;
  a3: number;
  a4: number;
  a5: number;
  a6: number;
  a7: number;
  a8: number;
  a9: number;
  a12: number;
  a24: number;
  Aberration_offset_x: number;
  Aberration_offset_y: number;
  Mode: Mode;
  Polarization: Polarization;
  Wavelength: number;
  Waist: number;
  Ampl_offset_x: number;
  Ampl_offset_y: number;
  Psi: number;
  Epsilon: number;
  VC: number;
  RC: number;
  Ring_Radius: number;
  Mask_offset_x: number;
  Mask_offset_y: number;
  p: number;
  SLM: Slm | null;
  Theta_sampling: ThetaSampling;
}

/** Quantities derived by the simulator from the parameters (see rust/core/src/params.rs). */
export interface Derived {
  k0: number;
  alpha: number;
  r0: number;
  gamma: number;
  alpha_eff: number;
  na_eff: number;
  r0_eff: number;
  alpha_int: number;
  alpha2_eff: number;
  alpha3_eff: number;
  dfoc: number;
  deltatheta: number;
  deltaphi: number;
}

/**
 * The Python defaults, except for a normalized output (the viewer expects
 * [0, 1]) and the adaptive θ quadrature. The grid (Nxy, Nz, Ntheta, Nphi) is
 * normally sized automatically from the optics (see physics.ts `autoGrid`);
 * these values are only the starting point of the manual mode.
 */
export const DEFAULTS: Params = {
  L_obs_XY: 2.0,
  L_obs_Z: 2.0,
  Nxy: 84,
  Nz: 41,
  Ntheta: 80,
  Nphi: 56,
  Normalize: 'YES',
  NA: 1.0,
  WD: 2800.0,
  n1: 1.33,
  n2: 1.52,
  n3: 1.38,
  Thickness: 170.0,
  Collar: 170.0,
  Depth: 0.0,
  Tilt: 0.0,
  Window: 'NO',
  Wind_Radius: 2.3,
  Wind_Depth: 2.23,
  Wind_Offset_x: 0.0,
  Wind_Offset_y: 0.0,
  a0: 0, a1: 0, a2: 0, a3: 0, a4: 0, a5: 0, a6: 0, a7: 0, a8: 0, a9: 0, a12: 0, a24: 0,
  Aberration_offset_x: 0.0,
  Aberration_offset_y: 0.0,
  Mode: 'GAUSSIAN',
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
  SLM: null,
  Theta_sampling: 'ADAPTIVE',
};

export const GRID_KEYS = ['Nxy', 'Nz', 'Ntheta', 'Nphi'] as const;

export interface Preset {
  name: string;
  description: string;
  params: Partial<Params>;
  /** Pattern to put on the SLM (enables it); see slm.ts `SlmDesign`. */
  slm?: Partial<SlmDesignLike>;
}

/** The subset of `SlmDesign` presets may set (kept here to avoid a cycle with slm.ts). */
export interface SlmDesignLike {
  pattern: 'flat' | 'vortex' | 'bottle' | 'halfmoon' | 'grating' | 'axicon';
  /** Zernike layer added on top of the pattern. */
  zernike: Partial<ZernikeCoeffs>;
  n: number;
  levels: number;
  fillFactor: number;
  extent: number;
  charge: number;
  ringRadius: number;
  ringStep: number;
  angle: number;
  period: number;
  cone: number;
}

export const PRESETS: Preset[] = [
  {
    name: 'Water immersion, 1.0 NA',
    description: 'The default configuration: a water objective focusing just below the coverslip.',
    params: {},
  },
  {
    name: 'Oil objective into aqueous sample',
    description: '1.4 NA oil immersion imaging 20 µm deep into water: refractive-index mismatch broadens the PSF axially and shifts the focus.',
    params: { NA: 1.4, n1: 1.518, n2: 1.518, n3: 1.33, Depth: 20, Wavelength: 0.488, L_obs_Z: 3 },
  },
  {
    name: 'STED donut',
    description: 'A vortex phase mask (charge 1) with circular polarization gives the lateral STED depletion pattern.',
    params: { Mode: 'DONUT', NA: 1.4, n1: 1.518, n2: 1.518, n3: 1.518, Wavelength: 0.775 },
  },
  {
    name: 'Bottle beam (3D STED)',
    description: 'A π phase step on the inner half of the pupil produces a dark focus enclosed axially.',
    params: { Mode: 'BOTTLE', NA: 1.4, n1: 1.518, n2: 1.518, n3: 1.518, Wavelength: 0.775, L_obs_Z: 3 },
  },
  {
    name: 'Tilted coverslip',
    description: 'A coverslip tilted by 8° relative to the optical axis introduces coma-like asymmetry.',
    params: { Tilt: 8, NA: 1.2, Depth: 10, L_obs_XY: 3, L_obs_Z: 3 },
  },
  {
    name: 'Cranial window',
    description: 'In vivo imaging through a 1 mm radius, 2 mm deep cranial window clips the focusing cone.',
    params: { Window: 'CUSTOM', Wind_Radius: 1.0, Wind_Depth: 2.0, NA: 1.0, Depth: 100 },
  },
  {
    name: 'Spherical aberration',
    description: 'Primary spherical aberration (a12) with a correction-collar mismatch.',
    params: { a12: 0.8, Thickness: 190, Collar: 170, L_obs_Z: 3 },
  },
  {
    name: 'Radial polarization',
    description: 'Radially polarized input creates a strong longitudinal field component at the focus.',
    params: { Polarization: 2, NA: 1.4, n1: 1.518, n2: 1.518, n3: 1.518 },
  },
  {
    name: 'STED donut on an 8-bit SLM',
    description: 'The vortex is displayed on a 256 × 256 pixel, 8-bit SLM with 93 % fill factor instead of a phase plate: pixelation and dead-space leakage lift the donut minimum slightly.',
    params: { Mode: 'GAUSSIAN', NA: 1.4, n1: 1.518, n2: 1.518, n3: 1.518, Wavelength: 0.775 },
    slm: { pattern: 'vortex', charge: 1, n: 256, levels: 256, fillFactor: 0.93 },
  },
  {
    name: 'Binary SLM bottle beam',
    description: 'A π disc on a binary (2-level) SLM makes a bottle beam; compare with the analytic phase plate preset.',
    params: { Mode: 'GAUSSIAN', NA: 1.4, n1: 1.518, n2: 1.518, n3: 1.518, Wavelength: 0.775, L_obs_Z: 3 },
    slm: { pattern: 'bottle', ringRadius: 0.707, ringStep: 1, n: 128, levels: 2, fillFactor: 1 },
  },
  {
    name: 'Beam steering (blazed grating)',
    description: 'A blazed grating on the SLM tilts the wavefront and moves the focus sideways by λ / (period · sin α); the 8-bit staircase puts a little light into other orders.',
    params: { Mode: 'GAUSSIAN', L_obs_XY: 3 },
    slm: { pattern: 'grating', period: 1.5, angle: 0, n: 256, levels: 256, fillFactor: 0.93 },
  },
  {
    name: 'Aberration correction on the SLM',
    description: 'Primary spherical aberration of the system (a12 = 0.8, collar mismatch) is corrected by the opposite Zernike mode displayed on an 8-bit SLM: what remains is the quantization and fill-factor residual.',
    params: { a12: 0.8, Thickness: 190, Collar: 170, L_obs_Z: 3 },
    slm: { pattern: 'flat', zernike: { a12: -0.8 }, n: 256, levels: 256, fillFactor: 0.93 },
  },
  {
    name: 'Ring focus (axicon on the SLM)',
    description: 'A conical phase deflects every annulus of the pupil by the same angle: the focus becomes a ring.',
    params: { Mode: 'GAUSSIAN', NA: 1.2, L_obs_XY: 3 },
    slm: { pattern: 'axicon', cone: 1.5, n: 256, levels: 256, fillFactor: 1 },
  },
];

export interface NumberField {
  key: keyof Params;
  label: string;
  min: number;
  max: number;
  step: number;
  unit?: string;
  hint?: string;
  integer?: boolean;
}

export interface SelectField {
  key: keyof Params;
  label: string;
  options: { value: string | number; label: string }[];
  hint?: string;
}

export type Field = NumberField | SelectField;

export interface FieldGroup {
  title: string;
  description?: string;
  fields: Field[];
}

export function isSelect(f: Field): f is SelectField {
  return 'options' in f;
}

/** The Zernike mode sliders, shared by the aberration card and the SLM's Zernike layer. */
export const ZERNIKE_FIELDS: (NumberField & { key: ZernikeKey })[] = [
  { key: 'a4', label: 'Defocus', min: -1, max: 1, step: 0.01 },
  { key: 'a12', label: 'Primary spherical', min: -1, max: 1, step: 0.01 },
  { key: 'a24', label: 'Secondary spherical', min: -1, max: 1, step: 0.01 },
  { key: 'a3', label: 'Oblique astigmatism', min: -1, max: 1, step: 0.01 },
  { key: 'a5', label: 'Vertical astigmatism', min: -1, max: 1, step: 0.01 },
  { key: 'a7', label: 'Vertical coma', min: -1, max: 1, step: 0.01 },
  { key: 'a8', label: 'Horizontal coma', min: -1, max: 1, step: 0.01 },
  { key: 'a6', label: 'Vertical trefoil', min: -1, max: 1, step: 0.01 },
  { key: 'a9', label: 'Oblique trefoil', min: -1, max: 1, step: 0.01 },
  { key: 'a1', label: 'Vertical tilt', min: -1, max: 1, step: 0.01 },
  { key: 'a2', label: 'Horizontal tilt', min: -1, max: 1, step: 0.01 },
  { key: 'a0', label: 'Piston', min: -1, max: 1, step: 0.01 },
];

/** The clickable parts of the microscope; each owns a group of parameters. */
export type ComponentId =
  | 'laser'
  | 'polarization'
  | 'phaseplate'
  | 'slm'
  | 'objective'
  | 'pupil'
  | 'coverslip'
  | 'sample'
  | 'window'
  | 'focus';

export interface ComponentDef extends FieldGroup {
  id: ComponentId;
  /** One line shown under the title in the inspector. */
  summary: string;
}

export const COMPONENTS: ComponentDef[] = [
  {
    id: 'laser',
    title: 'Laser',
    summary: 'The excitation beam entering the back pupil.',
    fields: [
      { key: 'Wavelength', label: 'Wavelength', min: 0.3, max: 1.3, step: 0.001, unit: 'µm' },
      { key: 'Waist', label: 'Beam waist', min: 100, max: 24000, step: 100, unit: 'µm', hint: '1/e field radius of the Gaussian profile on the pupil; large = flat top' },
      { key: 'Ampl_offset_x', label: 'Beam offset x', min: -10, max: 10, step: 0.1, hint: 'In units of r0 / Nxy' },
      { key: 'Ampl_offset_y', label: 'Beam offset y', min: -10, max: 10, step: 0.1 },
    ],
  },
  {
    id: 'polarization',
    title: 'Polarization',
    summary: 'Wave plates set the polarization state of the beam.',
    fields: [
      {
        key: 'Polarization',
        label: 'State',
        options: [
          { value: 1, label: 'Elliptical (λ/2 + λ/4 plates)' },
          { value: 2, label: 'Radial' },
          { value: 3, label: 'Azimuthal' },
        ],
      },
      { key: 'Psi', label: 'Direction ψ', min: 0, max: 180, step: 1, unit: '°' },
      { key: 'Epsilon', label: 'Ellipticity ε', min: -45, max: 45, step: 1, unit: '°', hint: '0 linear, ±45 circular' },
    ],
  },
  {
    id: 'phaseplate',
    title: 'Phase plate',
    summary: 'An analytic (ideal) phase mask in the beam path.',
    fields: [
      {
        key: 'Mode',
        label: 'Mask',
        options: [
          { value: 'DONUT', label: 'Vortex (donut)' },
          { value: 'BOTTLE', label: 'π disc (bottle)' },
          { value: 'DONUT BOTTLE', label: 'Donut + bottle (incoherent mix)' },
        ],
      },
      { key: 'VC', label: 'Vortex charge', min: -5.9, max: 5.9, step: 0.1 },
      { key: 'RC', label: 'Disc phase step', min: -5.9, max: 5.9, step: 0.1, unit: 'π' },
      { key: 'Ring_Radius', label: 'Disc radius', min: 0.01, max: 0.99, step: 0.001, hint: 'On the unit pupil' },
      { key: 'p', label: 'Donut / bottle mix', min: 0.01, max: 0.99, step: 0.01 },
      { key: 'Mask_offset_x', label: 'Mask offset x', min: -10, max: 10, step: 0.1, hint: 'In units of r0 / Nxy' },
      { key: 'Mask_offset_y', label: 'Mask offset y', min: -10, max: 10, step: 0.1 },
    ],
  },
  {
    id: 'slm',
    title: 'Spatial light modulator',
    summary: 'A pixelated, quantized phase pattern conjugate to the pupil.',
    fields: [],
  },
  {
    id: 'objective',
    title: 'Objective',
    summary: 'Numerical aperture, immersion medium and correction collar.',
    fields: [
      { key: 'NA', label: 'Numerical aperture', min: 0.1, max: 1.5, step: 0.01, hint: 'Must be below n1' },
      { key: 'n1', label: 'n₁ immersion', min: 1.0, max: 1.8, step: 0.001, hint: 'Air 1.0, water 1.33, glycerol 1.47, oil 1.518' },
      { key: 'Collar', label: 'Correction collar', min: 1, max: 299, step: 1, unit: 'µm', hint: 'Coverslip thickness the objective is corrected for' },
      { key: 'WD', label: 'Focal length', min: 500, max: 10000, step: 10, unit: 'µm', hint: 'Sets the pupil radius r0 = f·sin(α)' },
    ],
  },
  {
    id: 'pupil',
    title: 'Back pupil & aberrations',
    summary: 'Wavefront errors of the system as Zernike modes (radians of phase, 1 rad ≈ 0.16 waves).',
    fields: [
      ...ZERNIKE_FIELDS,
      { key: 'Aberration_offset_x', label: 'Aberration offset x', min: -10, max: 10, step: 0.1, hint: 'In units of r0 / Nxy' },
      { key: 'Aberration_offset_y', label: 'Aberration offset y', min: -10, max: 10, step: 0.1 },
    ],
  },
  {
    id: 'coverslip',
    title: 'Coverslip',
    summary: 'Thickness, refractive index and tilt of the coverslip.',
    fields: [
      { key: 'Thickness', label: 'Thickness', min: 0, max: 299, step: 1, unit: 'µm' },
      { key: 'n2', label: 'n₂ coverslip', min: 1.0, max: 1.8, step: 0.001 },
      { key: 'Tilt', label: 'Tilt', min: -19.9, max: 19.9, step: 0.1, unit: '°' },
    ],
  },
  {
    id: 'sample',
    title: 'Sample',
    summary: 'The medium below the coverslip, how deep the focus sits in it, and what is imaged there.',
    fields: [
      { key: 'n3', label: 'n₃ sample', min: 1.0, max: 1.8, step: 0.001 },
      { key: 'Depth', label: 'Imaging depth', min: 0, max: 999, step: 1, unit: 'µm', hint: 'Nominal focus below the coverslip' },
    ],
  },
  {
    id: 'window',
    title: 'Cranial window',
    summary: 'An optional skull opening on the coverslip that clips the focusing cone (in vivo imaging).',
    fields: [
      { key: 'Wind_Radius', label: 'Radius', min: 0.1, max: 5, step: 0.05, unit: 'mm' },
      { key: 'Wind_Depth', label: 'Depth', min: 0.1, max: 5, step: 0.05, unit: 'mm', hint: 'Together with the radius this sets the effective NA' },
    ],
  },
  {
    id: 'focus',
    title: 'Focus & sampling',
    summary: 'The observation volume around the focus and how finely it is computed.',
    fields: [
      { key: 'L_obs_XY', label: 'Half-width XY', min: 0.25, max: 10, step: 0.05, unit: 'µm' },
      { key: 'L_obs_Z', label: 'Half-width Z', min: 0.25, max: 20, step: 0.05, unit: 'µm' },
      {
        key: 'Normalize',
        label: 'Normalize',
        options: [
          { value: 'YES', label: 'Peak to 1' },
          { value: 'NO', label: 'Raw intensity' },
        ],
      },
      {
        key: 'Theta_sampling',
        label: 'θ quadrature',
        options: [
          { value: 'ADAPTIVE', label: 'Adaptive (resolves the critical angle)' },
          { value: 'UNIFORM', label: 'Uniform (numpy reference)' },
        ],
        hint: 'Adaptive integrates over cos θ₃ below the critical angle of the sample; identical otherwise',
      },
      { key: 'Nxy', label: 'Nxy', min: 16, max: 256, step: 1, integer: true, hint: 'Lateral samples' },
      { key: 'Nz', label: 'Nz', min: 5, max: 128, step: 1, integer: true, hint: 'Axial samples' },
      { key: 'Ntheta', label: 'Nθ', min: 16, max: 400, step: 1, integer: true, hint: 'Quadrature nodes over the focusing angle' },
      { key: 'Nphi', label: 'Nφ', min: 16, max: 320, step: 1, integer: true, hint: 'Quadrature nodes around the pupil' },
    ],
  },
];

export const COMPONENT_BY_ID: Record<ComponentId, ComponentDef> = Object.fromEntries(
  COMPONENTS.map((c) => [c.id, c]),
) as Record<ComponentId, ComponentDef>;

/** Kept for callers that want a flat list of groups. */
export const FIELD_GROUPS: FieldGroup[] = COMPONENTS;

/** Work units of a config; compute time is roughly proportional. */
export function workUnits(p: Params): number {
  return p.Ntheta * (p.Nphi + p.Nz) * p.Nxy * p.Nxy * (p.Mode === 'DONUT BOTTLE' ? 2 : 1);
}

function coerceSlm(v: unknown): Slm | null {
  if (!v || typeof v !== 'object') return null;
  const o = v as Record<string, unknown>;
  const n = Number(o.n);
  const phase = Array.isArray(o.phase) ? o.phase.map(Number) : null;
  if (!Number.isInteger(n) || n <= 0 || !phase || phase.length !== n * n || phase.some((x) => !Number.isFinite(x))) return null;
  const num = (k: string, d: number) => (Number.isFinite(Number(o[k])) ? Number(o[k]) : d);
  return {
    n,
    phase,
    extent: Math.max(num('extent', 1), 1e-3),
    levels: Math.max(0, Math.round(num('levels', 0))),
    fill_factor: Math.min(1, Math.max(1e-3, num('fill_factor', 1))),
    offset_x: num('offset_x', 0),
    offset_y: num('offset_y', 0),
  };
}

/**
 * Merge a (possibly partial / foreign) config object onto the defaults.
 * Also reports whether the file carried an explicit simulation grid.
 */
export function coerceParams(input: unknown): { params: Params; hasGrid: boolean } {
  const out: Params = { ...DEFAULTS };
  if (!input || typeof input !== 'object') return { params: out, hasGrid: false };
  const src = input as Record<string, unknown>;
  let hasGrid = false;
  for (const key of Object.keys(DEFAULTS) as (keyof Params)[]) {
    const v = src[key];
    if (v === undefined || v === null) continue;
    if (key === 'SLM') {
      out.SLM = coerceSlm(v);
      continue;
    }
    const def = DEFAULTS[key];
    if (typeof def === 'number') {
      const n = typeof v === 'number' ? v : Number(v);
      if (Number.isFinite(n)) {
        (out as unknown as Record<string, unknown>)[key] = n;
        if ((GRID_KEYS as readonly string[]).includes(key)) hasGrid = true;
      }
    } else if (typeof v === 'string' || typeof v === 'number') {
      (out as unknown as Record<string, unknown>)[key] = v;
    }
  }
  if (out.Mode === 'LOADED' && !out.SLM) out.Mode = 'GAUSSIAN';
  return { params: out, hasGrid };
}
