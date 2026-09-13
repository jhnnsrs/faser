'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Dices } from 'lucide-react';
import { cn } from '@/lib/cn';
import { COLORMAPS, colormapCss, type ColormapName } from './colormaps';
import { SliceCanvas, type SliceMapping } from './slice-canvas';
import type { PsfResult } from './use-psf-worker';
import { VolumeViewer, type RenderSettings, DEFAULT_RENDER } from './volume-viewer';
import { psfToVolume, psfVoxel, SAMPLE_KINDS, type SampleKind, type SampleSpec, type Volume } from './volume';

interface Props {
  psf: PsfResult | null;
  ready: boolean;
  generateSample: (spec: SampleSpec) => Promise<Volume>;
  convolve: (sample: Volume, psf: Volume, photons: number, seed: number) => Promise<Volume>;
}

interface SampleSettings {
  kind: SampleKind;
  seed: number;
  nxy: number;
  nz: number;
  count: number;
  radius: number;
  spacing: number;
  photons: number;
}

const DEFAULT_SAMPLE: SampleSettings = {
  kind: 'beads',
  seed: 1,
  nxy: 96,
  nz: 48,
  count: 30,
  radius: 0.15,
  spacing: 0.6,
  photons: 0,
};

/** Sensible object count / size per sample kind, applied when the kind changes. */
const KIND_DEFAULTS: Record<SampleKind, Pick<SampleSettings, 'count' | 'radius' | 'spacing'>> = {
  beads: { count: 30, radius: 0.15, spacing: 0.6 },
  filaments: { count: 6, radius: 0.06, spacing: 0.6 },
  cells: { count: 2, radius: 0.1, spacing: 0.6 },
  lattice: { count: 1, radius: 0.08, spacing: 0.6 },
  spokes: { count: 24, radius: 0.15, spacing: 0.3 },
};

/** Rough cost guard: the padded FFT grid must stay reasonable for one core. */
const MAX_FFT_VOXELS = 24_000_000;

function Num({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-xs">
      <span className="w-24 shrink-0 text-muted-foreground">{label}</span>
      <input
        type="range"
        className="pg-range min-w-0 flex-1"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="w-16 text-right tabular-nums">
        {value}
        {unit ? ` ${unit}` : ''}
      </span>
    </label>
  );
}

/**
 * Image formation: a synthetic ground-truth volume, sampled on the PSF's
 * voxel grid, is convolved with the current PSF (FFT, in the worker), with
 * optional Poisson shot noise.
 */
export function SimulationPane({ psf, ready, generateSample, convolve }: Props) {
  const [settings, setSettings] = useState<SampleSettings>(DEFAULT_SAMPLE);
  const [sample, setSample] = useState<Volume | null>(null);
  const [image, setImage] = useState<Volume | null>(null);
  const [busy, setBusy] = useState<'sample' | 'image' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [colormap, setColormap] = useState<ColormapName>('gray');
  const [gamma, setGamma] = useState(1);
  const [view, setView] = useState<'3d' | 'slices'>('slices');
  const [z, setZ] = useState<number | null>(null);
  const render = useMemo<RenderSettings>(
    () => ({ ...DEFAULT_RENDER, mode: 'composite', colormap: colormap === 'gray' ? 'inferno' : colormap, threshold: 0.05, opacity: 0.9, gamma }),
    [colormap, gamma],
  );

  const voxel = psf ? psfVoxel(psf) : null;
  const spec = useMemo<SampleSpec | null>(() => {
    if (!voxel) return null;
    return {
      kind: settings.kind,
      seed: settings.seed,
      nx: settings.nxy,
      ny: settings.nxy,
      nz: settings.nz,
      dx: voxel.dx,
      dz: voxel.dz,
      count: settings.count,
      radius: settings.radius,
      spacing: settings.spacing,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.kind, settings.seed, settings.nxy, settings.nz, settings.count, settings.radius, settings.spacing, voxel?.dx, voxel?.dz]);

  const fftVoxels = psf ? (settings.nxy + psf.nx) ** 2 * (settings.nz + psf.nz) : 0;
  const tooLarge = fftVoxels > MAX_FFT_VOXELS;

  // 1. sample follows its spec
  const sampleJob = useRef(0);
  useEffect(() => {
    if (!spec || !ready) return;
    const job = ++sampleJob.current;
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
  }, [spec, ready, generateSample]);

  // 2. image follows sample + PSF (+ noise)
  const imageJob = useRef(0);
  const runConvolution = useCallback(() => {
    if (!sample || !psf || tooLarge) return;
    const job = ++imageJob.current;
    setBusy('image');
    convolve(sample, psfToVolume(psf), settings.photons, settings.seed)
      .then((v) => {
        if (job !== imageJob.current) return;
        setImage(v);
        setError(null);
      })
      .catch((e) => job === imageJob.current && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => job === imageJob.current && setBusy((b) => (b === 'image' ? null : b)));
  }, [sample, psf, tooLarge, convolve, settings.photons, settings.seed]);

  useEffect(() => {
    const id = setTimeout(runConvolution, 150);
    return () => clearTimeout(id);
  }, [runConvolution]);

  const nz = sample?.nz ?? settings.nz;
  const zi = Math.min(z ?? Math.floor(nz / 2), nz - 1);
  const yc = Math.floor((sample?.ny ?? settings.nxy) / 2);
  const mapping: SliceMapping = { kind: 'linear', gamma };
  const update = (patch: Partial<SampleSettings>) => setSettings((s) => ({ ...s, ...patch }));
  const kindInfo = SAMPLE_KINDS.find((k) => k.value === settings.kind);

  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-muted-foreground">
        A synthetic sample on the PSF&apos;s voxel grid ({voxel ? `${voxel.dx.toFixed(4)} × ${voxel.dz.toFixed(4)} µm` : '–'}) is
        imaged by convolving it with the current PSF (linear 3-D FFT convolution, PSF normalized to unit energy).
        Optional shot noise draws Poisson counts with the given photons in the brightest voxel.
      </p>

      {/* Sample controls */}
      <div className="grid gap-x-6 gap-y-2 rounded-lg border bg-card p-3 md:grid-cols-2">
        <label className="flex items-center gap-2 text-xs">
          <span className="w-24 shrink-0 text-muted-foreground">Sample</span>
          <select
            className="min-w-0 flex-1 rounded-md border bg-background px-2 py-1"
            value={settings.kind}
            onChange={(e) => {
              const kind = e.target.value as SampleKind;
              update({ kind, ...KIND_DEFAULTS[kind] });
            }}
          >
            {SAMPLE_KINDS.map((k) => (
              <option key={k.value} value={k.value}>
                {k.label}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="inline-flex items-center gap-1 rounded-md border px-2 py-1"
            onClick={() => update({ seed: Math.floor(Math.random() * 1e6) })}
            title="New random sample"
          >
            <Dices className="size-3.5" />
            seed {settings.seed}
          </button>
        </label>
        <p className="text-xs text-muted-foreground md:pt-1">{kindInfo?.description}</p>
        <Num label="Field xy" value={settings.nxy} min={32} max={256} step={8} unit="px" onChange={(v) => update({ nxy: v })} />
        <Num label="Field z" value={settings.nz} min={8} max={128} step={4} unit="px" onChange={(v) => update({ nz: v })} />
        {(settings.kind === 'beads' || settings.kind === 'filaments' || settings.kind === 'cells' || settings.kind === 'spokes') && (
          <Num
            label={settings.kind === 'spokes' ? 'Spokes' : 'Count'}
            value={settings.count}
            min={settings.kind === 'spokes' ? 4 : 1}
            max={settings.kind === 'cells' ? 8 : settings.kind === 'spokes' ? 64 : 120}
            step={1}
            onChange={(v) => update({ count: v })}
          />
        )}
        <Num
          label={settings.kind === 'cells' ? 'Membrane' : settings.kind === 'spokes' ? 'Inner radius' : 'Radius'}
          value={settings.radius}
          min={0.02}
          max={1}
          step={0.01}
          unit="µm"
          onChange={(v) => update({ radius: v })}
        />
        {(settings.kind === 'lattice' || settings.kind === 'spokes') && (
          <Num
            label={settings.kind === 'lattice' ? 'Spacing' : 'Slab thickness'}
            value={settings.spacing}
            min={0.05}
            max={3}
            step={0.05}
            unit="µm"
            onChange={(v) => update({ spacing: v })}
          />
        )}
        <Num label="Photons (peak)" value={settings.photons} min={0} max={2000} step={10} onChange={(v) => update({ photons: v })} />
        <div className="text-xs text-muted-foreground md:col-span-2">
          Field of view {voxel ? `${(settings.nxy * voxel.dx).toFixed(2)} × ${(settings.nxy * voxel.dx).toFixed(2)} × ${(settings.nz * voxel.dz).toFixed(2)} µm` : '–'}
          {tooLarge && <span className="ml-2 text-amber-600 dark:text-amber-400">Too large for the FFT in the browser, reduce the field or the PSF grid.</span>}
          {error && <span className="ml-2 text-red-600 dark:text-red-400">{error}</span>}
        </div>
      </div>

      {/* View controls */}
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <div className="inline-flex overflow-hidden rounded-md border">
          {(['slices', '3d'] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setView(m)}
              className={cn('px-2 py-1', view === m ? 'bg-primary text-primary-foreground' : 'text-muted-foreground')}
            >
              {m === 'slices' ? 'Slices' : '3D'}
            </button>
          ))}
        </div>
        <select className="rounded-md border bg-background px-2 py-1" value={colormap} onChange={(e) => setColormap(e.target.value as ColormapName)}>
          {COLORMAPS.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <label className="inline-flex items-center gap-1">
          γ
          <input type="range" className="pg-range w-16" min={0.2} max={2} step={0.05} value={gamma} onChange={(e) => setGamma(Number(e.target.value))} />
          {gamma.toFixed(2)}
        </label>
        {view === 'slices' && (
          <label className="inline-flex items-center gap-1">
            z
            <input type="range" className="pg-range w-32" min={0} max={nz - 1} step={1} value={zi} onChange={(e) => setZ(Number(e.target.value))} />
            {voxel ? `${((zi - (nz - 1) / 2) * voxel.dz).toFixed(2)} µm` : zi}
          </label>
        )}
        <span className="ml-auto text-muted-foreground">
          {busy === 'sample' ? 'Generating sample…' : busy === 'image' ? 'Convolving…' : image?.ms != null ? `convolved in ${image.ms.toFixed(0)} ms` : ''}
        </span>
      </div>

      {/* Views */}
      {!sample ? (
        <div className="flex h-48 items-center justify-center rounded-xl border text-sm text-muted-foreground">
          {psf ? 'Generating the sample…' : 'Generate a PSF first'}
        </div>
      ) : view === '3d' ? (
        <div className="grid gap-3 md:grid-cols-2">
          <div className="flex flex-col gap-1">
            <h3 className="text-xs font-semibold">Ground truth</h3>
            <div className="relative h-[360px] overflow-hidden rounded-xl border bg-[#0b0b10]">
              <VolumeViewer volume={sample} settings={render} />
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <h3 className="text-xs font-semibold">Imaged through the PSF</h3>
            <div className="relative h-[360px] overflow-hidden rounded-xl border bg-[#0b0b10]">
              <VolumeViewer volume={image} settings={render} />
            </div>
          </div>
        </div>
      ) : (
        <div className="grid gap-3 md:grid-cols-2">
          <div className="flex flex-col gap-2">
            <h3 className="text-xs font-semibold">Ground truth</h3>
            <SliceCanvas volume={sample} plane="xy" index={zi} colormap={colormap} mapping={mapping} maxHeight={340} />
            <SliceCanvas volume={sample} plane="xz" index={yc} colormap={colormap} mapping={mapping} maxHeight={340} />
          </div>
          <div className="flex flex-col gap-2">
            <h3 className="text-xs font-semibold">Imaged through the PSF</h3>
            {image ? (
              <>
                <SliceCanvas volume={image} plane="xy" index={zi} colormap={colormap} mapping={mapping} maxHeight={340} />
                <SliceCanvas volume={image} plane="xz" index={yc} colormap={colormap} mapping={mapping} maxHeight={340} />
              </>
            ) : (
              <div className="flex h-48 items-center justify-center rounded-md border text-xs text-muted-foreground">
                {tooLarge ? 'Not computed' : 'Convolving…'}
              </div>
            )}
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground md:col-span-2">
            <span>0</span>
            <div className="h-2 flex-1 rounded" style={{ background: colormapCss(colormap) }} />
            <span>max{settings.photons > 0 ? ` (${settings.photons} photons)` : ''}</span>
          </div>
        </div>
      )}
      <p className="text-xs text-muted-foreground">
        XY planes at the chosen z, XZ planes through the centre of the field. The image is normalized to its own
        maximum; the PSF&apos;s field of view sets the blur reach, so make it at least as large as the visible PSF.
      </p>
    </div>
  );
}
