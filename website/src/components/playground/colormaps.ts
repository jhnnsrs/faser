/**
 * Colormaps as functions [0, 1] -> [r, g, b] in [0, 1]. Viridis, inferno,
 * magma and plasma use the 6th-order polynomial fits of the matplotlib maps.
 */

export type ColormapName = 'inferno' | 'viridis' | 'magma' | 'plasma' | 'gray' | 'hot';

type Vec3 = [number, number, number];

const POLY: Record<'viridis' | 'inferno' | 'magma' | 'plasma', Vec3[]> = {
  viridis: [
    [0.2777273272234177, 0.005407344544966578, 0.3340998053353061],
    [0.1050930431085774, 1.404613529898575, 1.384590162594685],
    [-0.3308618287255563, 0.214847559468213, 0.09509516302823659],
    [-4.634230498983486, -5.799100973351585, -19.33244095627987],
    [6.228269936347081, 14.17993336680509, 56.69055260068105],
    [4.776384997670288, -13.74514537774601, -65.35303263337234],
    [-5.435455855934631, 4.645852612178535, 26.3124352495832],
  ],
  inferno: [
    [0.0002189403691192265, 0.001651004631001012, -0.01948089843709184],
    [0.1065134194856116, 0.5639564367884091, 3.932712388889277],
    [11.60249308247187, -3.972853965665698, -15.9423941062914],
    [-41.70399613139459, 17.43639888205313, 44.35414519872813],
    [77.162935699427, -33.40235894210092, -81.80730925738993],
    [-71.31942824499214, 32.62606426397723, 73.20951985803202],
    [25.13112622477341, -12.24266895238567, -23.07032500287172],
  ],
  magma: [
    [-0.002136485053939582, -0.000749655052795221, -0.005386127855323933],
    [0.2516605407371642, 0.6775232436837668, 2.494026599312351],
    [8.353717279216625, -3.577719514958484, 0.3144679030132573],
    [-27.66873308576866, 14.26473078096533, -13.68926204526553],
    [52.17613981234068, -27.94360607168351, 12.94416944238394],
    [-50.76852536473588, 29.04658282127291, 4.23415299384598],
    [18.65570506591883, -11.48977351997711, -5.601961508734096],
  ],
  plasma: [
    [0.05873234392399702, 0.02333670892565664, 0.5433401826748754],
    [2.176514634195958, 0.2383834171260182, 0.7539604599784036],
    [-2.689460476458034, -7.455851135738909, 3.110799939717086],
    [6.130348345893603, 42.3461881477227, -28.51885465332158],
    [-11.10743619062271, -82.66631109428045, 60.13984767418263],
    [10.02306557647065, 71.41361770095349, -54.07218655560067],
    [-3.658713842777788, -22.93153465461149, 18.19190778539828],
  ],
};

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

function poly(c: Vec3[], t: number): Vec3 {
  const out: Vec3 = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    let v = c[6][i];
    for (let k = 5; k >= 0; k--) v = c[k][i] + t * v;
    out[i] = clamp01(v);
  }
  return out;
}

export function colormap(name: ColormapName, t: number): Vec3 {
  t = clamp01(t);
  switch (name) {
    case 'gray':
      return [t, t, t];
    case 'hot':
      return [clamp01(t * 3), clamp01(t * 3 - 1), clamp01(t * 3 - 2)];
    default:
      return poly(POLY[name], t);
  }
}

/** 256 RGBA bytes for a 1-D lookup texture. */
export function colormapBytes(name: ColormapName): Uint8Array {
  const out = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = colormap(name, i / 255);
    out[i * 4] = Math.round(r * 255);
    out[i * 4 + 1] = Math.round(g * 255);
    out[i * 4 + 2] = Math.round(b * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

export const COLORMAPS: ColormapName[] = ['inferno', 'viridis', 'magma', 'plasma', 'hot', 'gray'];

export function colormapCss(name: ColormapName): string {
  const stops: string[] = [];
  for (let i = 0; i <= 10; i++) {
    const [r, g, b] = colormap(name, i / 10);
    stops.push(`rgb(${Math.round(r * 255)} ${Math.round(g * 255)} ${Math.round(b * 255)}) ${i * 10}%`);
  }
  return `linear-gradient(to right, ${stops.join(', ')})`;
}

/** Diverging blue–white–red map for signed data (t = 0.5 is zero). */
export function diverging(t: number): Vec3 {
  t = clamp01(t);
  const s = (t - 0.5) * 2; // -1 .. 1
  const a = Math.abs(s);
  const blue: Vec3 = [0.23, 0.3, 0.75];
  const red: Vec3 = [0.71, 0.02, 0.15];
  const end = s < 0 ? blue : red;
  return [1 + (end[0] - 1) * a, 1 + (end[1] - 1) * a, 1 + (end[2] - 1) * a];
}

export function divergingBytes(): Uint8Array {
  const out = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = diverging(i / 255);
    out[i * 4] = Math.round(r * 255);
    out[i * 4 + 1] = Math.round(g * 255);
    out[i * 4 + 2] = Math.round(b * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

export function divergingCss(): string {
  const stops: string[] = [];
  for (let i = 0; i <= 10; i++) {
    const [r, g, b] = diverging(i / 10);
    stops.push(`rgb(${Math.round(r * 255)} ${Math.round(g * 255)} ${Math.round(b * 255)}) ${i * 10}%`);
  }
  return `linear-gradient(to right, ${stops.join(', ')})`;
}
