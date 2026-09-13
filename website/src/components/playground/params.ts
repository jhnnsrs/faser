/**
 * The simulation parameters, field-for-field the Python `PSFConfig` model
 * (and therefore the `psf_config.json` the CLI / napari plugin write). The
 * wasm module deserializes exactly this object.
 */

export type Mode = 'GAUSSIAN' | 'DONUT' | 'BOTTLE' | 'DONUT BOTTLE';
export type YesNo = 'YES' | 'NO';
export type Window = 'NO' | 'CUSTOM';
/** 1 = elliptical, 2 = radial, 3 = azimuthal */
export type Polarization = 1 | 2 | 3;

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
 * The Python defaults, except for a grid that is quick enough for live
 * updates in the browser and a normalized output (the viewer expects [0, 1]).
 */
export const DEFAULTS: Params = {
  L_obs_XY: 2.0,
  L_obs_Z: 2.0,
  Nxy: 64,
  Nz: 32,
  Ntheta: 40,
  Nphi: 40,
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
};

export interface Preset {
  name: string;
  description: string;
  params: Partial<Params>;
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

export const FIELD_GROUPS: FieldGroup[] = [
  {
    title: 'Objective & media',
    fields: [
      { key: 'NA', label: 'Numerical aperture', min: 0.1, max: 1.5, step: 0.01, hint: 'Must be below n1' },
      { key: 'Wavelength', label: 'Wavelength', min: 0.3, max: 1.3, step: 0.001, unit: 'µm' },
      { key: 'n1', label: 'n1 immersion', min: 1.0, max: 1.8, step: 0.001, hint: 'Air 1.0, water 1.33, glycerol 1.47, oil 1.518' },
      { key: 'n2', label: 'n2 coverslip', min: 1.0, max: 1.8, step: 0.001 },
      { key: 'n3', label: 'n3 sample', min: 1.0, max: 1.8, step: 0.001 },
      { key: 'WD', label: 'Focal length', min: 500, max: 10000, step: 10, unit: 'µm', hint: 'Sets the pupil radius r0 = f·sin(α)' },
    ],
  },
  {
    title: 'Coverslip & sample',
    fields: [
      { key: 'Thickness', label: 'Coverslip thickness', min: 0, max: 299, step: 1, unit: 'µm' },
      { key: 'Collar', label: 'Correction collar', min: 1, max: 299, step: 1, unit: 'µm', hint: 'Coverslip thickness the objective is corrected for' },
      { key: 'Depth', label: 'Imaging depth', min: 0, max: 999, step: 1, unit: 'µm', hint: 'Nominal focus below the coverslip' },
      { key: 'Tilt', label: 'Coverslip tilt', min: -19.9, max: 19.9, step: 0.1, unit: '°' },
      {
        key: 'Window',
        label: 'Cranial window',
        options: [
          { value: 'NO', label: 'None' },
          { value: 'CUSTOM', label: 'Custom' },
        ],
      },
      { key: 'Wind_Radius', label: 'Window radius', min: 0.1, max: 5, step: 0.05, unit: 'mm' },
      { key: 'Wind_Depth', label: 'Window depth', min: 0.1, max: 5, step: 0.05, unit: 'mm' },
    ],
  },
  {
    title: 'Beam',
    fields: [
      {
        key: 'Mode',
        label: 'Phase mask',
        options: [
          { value: 'GAUSSIAN', label: 'Gaussian (none)' },
          { value: 'DONUT', label: 'Donut (vortex)' },
          { value: 'BOTTLE', label: 'Bottle (π ring)' },
          { value: 'DONUT BOTTLE', label: 'Donut + bottle' },
        ],
      },
      { key: 'Waist', label: 'Beam waist', min: 100, max: 24000, step: 100, unit: 'µm', hint: '1/e field radius on the pupil' },
      { key: 'VC', label: 'Vortex charge', min: -5.9, max: 5.9, step: 0.1 },
      { key: 'RC', label: 'Ring charge', min: -5.9, max: 5.9, step: 0.1 },
      { key: 'Ring_Radius', label: 'Ring radius', min: 0.01, max: 0.99, step: 0.001, hint: 'On the unit pupil' },
      { key: 'p', label: 'Donut / bottle mix', min: 0.01, max: 0.99, step: 0.01 },
      { key: 'Ampl_offset_x', label: 'Amplitude offset x', min: -10, max: 10, step: 0.1 },
      { key: 'Ampl_offset_y', label: 'Amplitude offset y', min: -10, max: 10, step: 0.1 },
      { key: 'Mask_offset_x', label: 'Mask offset x', min: -10, max: 10, step: 0.1 },
      { key: 'Mask_offset_y', label: 'Mask offset y', min: -10, max: 10, step: 0.1 },
    ],
  },
  {
    title: 'Polarization',
    fields: [
      {
        key: 'Polarization',
        label: 'State',
        options: [
          { value: 1, label: 'Elliptical' },
          { value: 2, label: 'Radial' },
          { value: 3, label: 'Azimuthal' },
        ],
      },
      { key: 'Psi', label: 'Direction ψ', min: 0, max: 180, step: 1, unit: '°' },
      { key: 'Epsilon', label: 'Ellipticity ε', min: -45, max: 45, step: 1, unit: '°', hint: '0 linear, ±45 circular' },
    ],
  },
  {
    title: 'Aberrations',
    description: 'Zernike coefficients in radians of phase (1 rad ≈ 0.16 waves).',
    fields: [
      { key: 'a0', label: 'Piston', min: -1, max: 1, step: 0.01 },
      { key: 'a1', label: 'Vertical tilt', min: -1, max: 1, step: 0.01 },
      { key: 'a2', label: 'Horizontal tilt', min: -1, max: 1, step: 0.01 },
      { key: 'a3', label: 'Oblique astigmatism', min: -1, max: 1, step: 0.01 },
      { key: 'a4', label: 'Defocus', min: -1, max: 1, step: 0.01 },
      { key: 'a5', label: 'Vertical astigmatism', min: -1, max: 1, step: 0.01 },
      { key: 'a6', label: 'Vertical trefoil', min: -1, max: 1, step: 0.01 },
      { key: 'a7', label: 'Vertical coma', min: -1, max: 1, step: 0.01 },
      { key: 'a8', label: 'Horizontal coma', min: -1, max: 1, step: 0.01 },
      { key: 'a9', label: 'Oblique trefoil', min: -1, max: 1, step: 0.01 },
      { key: 'a12', label: 'Primary spherical', min: -1, max: 1, step: 0.01 },
      { key: 'a24', label: 'Secondary spherical', min: -1, max: 1, step: 0.01 },
      { key: 'Aberration_offset_x', label: 'Aberration offset x', min: -10, max: 10, step: 0.1 },
      { key: 'Aberration_offset_y', label: 'Aberration offset y', min: -10, max: 10, step: 0.1 },
    ],
  },
  {
    title: 'Simulation grid',
    description: 'Cost grows with Ntheta × (Nphi + Nz) × Nxy².',
    fields: [
      { key: 'Nxy', label: 'Nxy', min: 8, max: 256, step: 1, integer: true, hint: 'Lateral samples' },
      { key: 'Nz', label: 'Nz', min: 1, max: 128, step: 1, integer: true, hint: 'Axial samples' },
      { key: 'Ntheta', label: 'Nθ', min: 4, max: 200, step: 1, integer: true, hint: 'Integration steps over the focusing angle' },
      { key: 'Nphi', label: 'Nφ', min: 4, max: 200, step: 1, integer: true, hint: 'Integration steps around the pupil' },
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
    ],
  },
];

/** Estimated relative cost of a config (1.0 = the default grid). */
export function relativeCost(p: Params): number {
  const cost = (q: Params) => q.Ntheta * (q.Nphi + q.Nz) * q.Nxy * q.Nxy * (q.Mode === 'DONUT BOTTLE' ? 2 : 1);
  return cost(p) / cost(DEFAULTS);
}

/** Merge a (possibly partial / foreign) config object onto the defaults. */
export function coerceParams(input: unknown): Params {
  const out: Params = { ...DEFAULTS };
  if (!input || typeof input !== 'object') return out;
  const src = input as Record<string, unknown>;
  for (const key of Object.keys(DEFAULTS) as (keyof Params)[]) {
    const v = src[key];
    if (v === undefined || v === null) continue;
    const def = DEFAULTS[key];
    if (typeof def === 'number') {
      const n = typeof v === 'number' ? v : Number(v);
      if (Number.isFinite(n)) (out as unknown as Record<string, unknown>)[key] = n;
    } else if (typeof v === 'string' || typeof v === 'number') {
      (out as unknown as Record<string, unknown>)[key] = v;
    }
  }
  return out;
}
