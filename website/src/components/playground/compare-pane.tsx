'use client';

import { useMemo } from 'react';
import { colormapCss, divergingCss, type ColormapName } from './colormaps';
import { SliceCanvas, type SliceMapping } from './slice-canvas';
import type { PsfResult } from './use-psf-worker';
import { argmax3, difference, fwhm, psfToVolume, psfVoxel, rms, type Volume } from './volume';

interface Props {
  vectorial: PsfResult;
  scalar: PsfResult | null;
  colormap: ColormapName;
  log: boolean;
  logDecades: number;
  gamma: number;
  busy: boolean;
}

function fmt(v: number, digits = 3) {
  return Number.isFinite(v) ? v.toFixed(digits) : '–';
}

function Column({
  title,
  volume,
  z,
  y,
  colormap,
  mapping,
}: {
  title: string;
  volume: Volume;
  z: number;
  y: number;
  colormap: ColormapName;
  mapping: SliceMapping;
}) {
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <h3 className="text-xs font-semibold">{title}</h3>
      <SliceCanvas volume={volume} plane="xy" index={z} colormap={colormap} mapping={mapping} maxHeight={300} />
      <SliceCanvas volume={volume} plane="xz" index={y} colormap={colormap} mapping={mapping} maxHeight={300} />
    </div>
  );
}

/**
 * Vectorial vs scalar PSF for the same parameters: both normalized to their
 * peak, shown through the brightest voxel of the vectorial PSF, with the
 * signed difference (vectorial − scalar) on a diverging scale.
 */
export function ComparePane({ vectorial, scalar, colormap, log, logDecades, gamma, busy }: Props) {
  const vec = useMemo(() => psfToVolume(vectorial), [vectorial]);
  const sca = useMemo(() => (scalar ? psfToVolume(scalar) : null), [scalar]);
  const sameGrid = sca && sca.nx === vec.nx && sca.ny === vec.ny && sca.nz === vec.nz;
  const diff = useMemo(() => (sameGrid ? difference(vec, sca) : null), [vec, sca, sameGrid]);
  const [z, y] = useMemo(() => argmax3(vec), [vec]);
  const yc = Math.floor(vec.ny / 2);
  const mapping: SliceMapping = log ? { kind: 'log', decades: logDecades, max: 1 } : { kind: 'linear', gamma, max: 1 };
  const { dx, dz } = psfVoxel(vectorial);

  const stats = useMemo(() => {
    if (!sca || !diff) return null;
    const at = argmax3(vec);
    const atS = argmax3(sca);
    const w = (v: Volume, a: [number, number, number]) => ({
      x: fwhm(v, 'x', a) * dx,
      y: fwhm(v, 'y', a) * dx,
      z: fwhm(v, 'z', a) * dz,
    });
    return {
      vec: w(vec, at),
      sca: w(sca, atS),
      maxDiff: diff.max,
      rms: rms(diff),
      peakShift: (atS[0] - at[0]) * dz,
    };
  }, [vec, sca, diff, dx, dz]);

  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-muted-foreground">
        Both PSFs use the same pupil, phase mask, aberrations, coverslip and depth phase. The scalar model drops the
        polarization coupling and the Fresnel transmission of the vectorial (Richards–Wolf / Török) integral, so the
        difference map shows what the vector nature of light adds: at high NA a wider, polarization-shaped focus with a
        longitudinal field component. Each PSF is normalized to its own peak; XY through the brightest vectorial
        voxel, XZ through the centre.
      </p>
      {!sca ? (
        <div className="flex h-48 items-center justify-center rounded-xl border text-sm text-muted-foreground">
          {busy ? 'Computing the scalar PSF…' : 'Scalar PSF not available'}
        </div>
      ) : !sameGrid || !diff ? (
        <div className="flex h-48 items-center justify-center rounded-xl border text-sm text-muted-foreground">
          Waiting for both models on the same grid…
        </div>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-3">
            <Column title="Vectorial" volume={vec} z={z} y={yc} colormap={colormap} mapping={mapping} />
            <Column title="Scalar" volume={sca} z={z} y={yc} colormap={colormap} mapping={mapping} />
            <Column
              title={`Vectorial − scalar (±${(diff.max * 100).toFixed(1)} % of peak)`}
              volume={diff}
              z={z}
              y={yc}
              colormap={colormap}
              mapping={{ kind: 'diverging', scale: diff.max }}
            />
          </div>
          <div className="grid grid-cols-2 gap-3 text-xs text-muted-foreground">
            <div className="flex items-center gap-2">
              <span>{log ? `10⁻${logDecades}` : '0'}</span>
              <div className="h-2 flex-1 rounded" style={{ background: colormapCss(colormap) }} />
              <span>1</span>
            </div>
            <div className="flex items-center gap-2">
              <span>−{(diff.max * 100).toFixed(1)} %</span>
              <div className="h-2 flex-1 rounded" style={{ background: divergingCss() }} />
              <span>+{(diff.max * 100).toFixed(1)} %</span>
            </div>
          </div>
          {stats && (
            <div className="overflow-x-auto rounded-lg border bg-card">
              <table className="w-full text-xs">
                <thead className="text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium">FWHM</th>
                    <th className="px-3 py-2 text-right font-medium">x</th>
                    <th className="px-3 py-2 text-right font-medium">y</th>
                    <th className="px-3 py-2 text-right font-medium">z</th>
                  </tr>
                </thead>
                <tbody className="tabular-nums">
                  <tr className="border-t">
                    <td className="px-3 py-1.5">Vectorial</td>
                    <td className="px-3 py-1.5 text-right">{fmt(stats.vec.x)} µm</td>
                    <td className="px-3 py-1.5 text-right">{fmt(stats.vec.y)} µm</td>
                    <td className="px-3 py-1.5 text-right">{fmt(stats.vec.z)} µm</td>
                  </tr>
                  <tr className="border-t">
                    <td className="px-3 py-1.5">Scalar</td>
                    <td className="px-3 py-1.5 text-right">{fmt(stats.sca.x)} µm</td>
                    <td className="px-3 py-1.5 text-right">{fmt(stats.sca.y)} µm</td>
                    <td className="px-3 py-1.5 text-right">{fmt(stats.sca.z)} µm</td>
                  </tr>
                  <tr className="border-t text-muted-foreground">
                    <td className="px-3 py-1.5">Difference</td>
                    <td className="px-3 py-1.5 text-right" colSpan={3}>
                      max |Δ| {(stats.maxDiff * 100).toFixed(2)} % of peak, RMS {(stats.rms * 100).toFixed(3)} %
                      {Math.abs(stats.peakShift) > 1e-9 ? `, scalar peak shifted ${stats.peakShift > 0 ? '+' : ''}${fmt(stats.peakShift, 3)} µm in z` : ''}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}
          <p className="text-xs text-muted-foreground">
            FWHM is measured on the sampled grid through each PSF&apos;s own peak (voxel {fmt(dx, 4)} × {fmt(dz, 4)} µm);
            increase Nxy / Nz for finer values. For a linearly polarized beam the x and y widths differ in the vectorial
            model only.
          </p>
        </>
      )}
    </div>
  );
}
