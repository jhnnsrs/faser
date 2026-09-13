'use client';

import { useEffect, useMemo, useRef } from 'react';
import { colormapBytes, colormapCss, type ColormapName } from './colormaps';
import type { PsfResult } from './use-psf-worker';
import { argmax3 } from './volume';

interface Props {
  result: PsfResult;
  colormap: ColormapName;
  log: boolean;
  logDecades: number;
  gamma: number;
}

function paint(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
  sample: (i: number, j: number) => number,
  lut: Uint8Array,
  map: (v: number) => number,
) {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const img = ctx.createImageData(width, height);
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      const v = map(sample(i, j));
      const c = Math.max(0, Math.min(255, Math.round(v * 255))) * 4;
      const k = (j * width + i) * 4;
      img.data[k] = lut[c];
      img.data[k + 1] = lut[c + 1];
      img.data[k + 2] = lut[c + 2];
      img.data[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

export function SliceViews({ result, colormap, log, logDecades, gamma }: Props) {
  const xyRef = useRef<HTMLCanvasElement>(null);
  const xzRef = useRef<HTMLCanvasElement>(null);
  const [zmax, ymax] = useMemo(() => argmax3(result), [result]);
  const { nx, ny, nz, data } = result;
  const { L_obs_XY, L_obs_Z } = result.params;

  useEffect(() => {
    const lut = colormapBytes(colormap);
    const logMin = Math.pow(10, -logDecades);
    const map = log
      ? (v: number) => (Math.log(Math.max(v, logMin)) - Math.log(logMin)) / -Math.log(logMin)
      : (v: number) => Math.pow(Math.max(v, 0), gamma);
    const yc = Math.floor(ny / 2);
    if (xyRef.current) {
      // image rows run top-to-bottom, +y up
      paint(xyRef.current, nx, ny, (i, j) => data[zmax * ny * nx + (ny - 1 - j) * nx + i], lut, map);
    }
    if (xzRef.current) {
      // z up (towards the objective), x across
      paint(xzRef.current, nx, nz, (i, j) => data[(nz - 1 - j) * ny * nx + yc * nx + i], lut, map);
    }
  }, [result, colormap, log, logDecades, gamma, nx, ny, nz, data, zmax]);

  const dz = nz > 1 ? (2 * L_obs_Z) / (nz - 1) : 0;
  const zAt = -L_obs_Z + zmax * dz + result.derived.dfoc;
  const dxy = nx > 1 ? (2 * L_obs_XY) / (nx - 1) : 0;

  return (
    <div className="grid grid-cols-2 gap-3 text-xs text-muted-foreground">
      <figure className="min-w-0">
        <div className="overflow-hidden rounded-md border bg-black" style={{ aspectRatio: '1 / 1' }}>
          <canvas ref={xyRef} className="h-full w-full" style={{ imageRendering: nx < 48 ? 'pixelated' : 'auto' }} />
        </div>
        <figcaption className="mt-1">
          XY at brightest z ({zAt >= 0 ? '+' : ''}
          {zAt.toFixed(2)} µm), {(2 * L_obs_XY).toFixed(2)} µm wide
        </figcaption>
      </figure>
      <figure className="min-w-0">
        <div
          className="overflow-hidden rounded-md border bg-black"
          style={{ aspectRatio: `${2 * L_obs_XY} / ${2 * L_obs_Z}`, maxHeight: '100%' }}
        >
          <canvas ref={xzRef} className="h-full w-full" style={{ imageRendering: nx < 48 ? 'pixelated' : 'auto' }} />
        </div>
        <figcaption className="mt-1">
          XZ through y = {((ymax - (ny - 1) / 2) * dxy).toFixed(2)} µm, z up, {(2 * L_obs_Z).toFixed(2)} µm tall
        </figcaption>
      </figure>
      <div className="col-span-2 flex items-center gap-2">
        <span>{log ? `10⁻${logDecades}` : '0'}</span>
        <div className="h-2 flex-1 rounded" style={{ background: colormapCss(colormap) }} />
        <span>1</span>
      </div>
    </div>
  );
}
