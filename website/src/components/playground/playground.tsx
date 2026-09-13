'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Download, FolderOpen, Play, RotateCcw, Tag, Zap } from 'lucide-react';
import { cn } from '@/lib/cn';
import { COLORMAPS, type ColormapName } from './colormaps';
import { ComparePane } from './compare-pane';
import { MicroscopeScene } from './microscope-scene';
import { ParamPanel } from './param-panel';
import { coerceParams, DEFAULTS, PRESETS, relativeCost, type Derived, type Params } from './params';
import { SimulationPane } from './simulation-pane';
import { SliceViews } from './slice-views';
import { writeTiff } from './tiff';
import { usePsfWorker, type PsfResult } from './use-psf-worker';
import { psfToVolume } from './volume';
import { DEFAULT_RENDER, VolumeViewer, type RenderSettings } from './volume-viewer';

const AUTO_COST_LIMIT = 6; // relative to the default grid; above it, auto-update is paused

type Pane = 'psf' | 'compare' | 'simulate';

const PANES: { id: Pane; label: string }[] = [
  { id: 'psf', label: 'PSF & microscope' },
  { id: 'compare', label: 'Vectorial vs scalar' },
  { id: 'simulate', label: 'Imaging simulation' },
];

function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function fmt(v: number, digits = 3) {
  return Number.isFinite(v) ? v.toFixed(digits) : '–';
}

export function Playground() {
  const {
    status: workerStatus,
    error: workerError,
    generate: workerGenerate,
    derive: workerDerive,
    sample: workerSample,
    convolve: workerConvolve,
  } = usePsfWorker();
  const [params, setParams] = useState<Params>(DEFAULTS);
  const [result, setResult] = useState<PsfResult | null>(null);
  const [scalarResult, setScalarResult] = useState<PsfResult | null>(null);
  const [derived, setDerived] = useState<Derived | null>(null);
  const [render, setRender] = useState<RenderSettings>(DEFAULT_RENDER);
  const [auto, setAuto] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showLabels, setShowLabels] = useState(true);
  const [preset, setPreset] = useState(0);
  const [pane, setPane] = useState<Pane>('psf');
  const fileRef = useRef<HTMLInputElement>(null);

  // Latest params requested, to coalesce rapid slider changes into
  // "compute the newest once the current one is done".
  const wanted = useRef<Params | null>(null);
  const running = useRef(false);
  const paneRef = useRef<Pane>('psf');
  useEffect(() => {
    paneRef.current = pane;
  }, [pane]);

  const runLatest = useCallback(async () => {
    if (running.current) return;
    running.current = true;
    setBusy(true);
    try {
      while (wanted.current) {
        const next = wanted.current;
        wanted.current = null;
        try {
          const r = await workerGenerate(next, false);
          setResult(r);
          setDerived(r.derived);
          setError(null);
          if (paneRef.current === 'compare') {
            setScalarResult(await workerGenerate(next, true));
          }
        } catch (e) {
          setError(e instanceof Error ? e.message : String(e));
        }
      }
    } finally {
      running.current = false;
      setBusy(false);
    }
  }, [workerGenerate]);

  const generate = useCallback(
    (p: Params) => {
      wanted.current = p;
      void runLatest();
    },
    [runLatest],
  );

  const cost = relativeCost(params);
  const autoActive = auto && cost <= AUTO_COST_LIMIT;

  // Derived quantities follow every edit immediately (cheap); the volume
  // follows with a short debounce when auto-update is on.
  useEffect(() => {
    if (workerStatus !== 'ready') return;
    let cancelled = false;
    workerDerive(params)
      .then((d) => {
        if (!cancelled) {
          setDerived(d);
          setError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    if (!autoActive) return () => void (cancelled = true);
    const id = setTimeout(() => generate(params), 120);
    return () => {
      cancelled = true;
      clearTimeout(id);
    };
  }, [params, workerStatus, workerDerive, autoActive, generate]);

  // Entering the comparison pane with a vectorial result but no matching
  // scalar one: compute the scalar PSF for the same parameters.
  useEffect(() => {
    if (pane !== 'compare' || !result) return;
    if (scalarResult && scalarResult.params === result.params) return;
    let cancelled = false;
    workerGenerate(result.params, true)
      .then((r) => {
        if (!cancelled) setScalarResult(r);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [pane, result, scalarResult, workerGenerate]);

  const update = useCallback((patch: Partial<Params>) => setParams((p) => ({ ...p, ...patch })), []);

  const applyPreset = (i: number) => {
    setPreset(i);
    setParams({ ...DEFAULTS, ...PRESETS[i].params });
  };

  const loadJson = async (file: File) => {
    try {
      const text = await file.text();
      setParams(coerceParams(JSON.parse(text)));
      setError(null);
    } catch (e) {
      setError(`could not read config: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const downloadTiff = () => {
    if (!result) return;
    const p = result.params;
    const dxy = p.Nxy > 1 ? (2 * p.L_obs_XY) / (p.Nxy - 1) : 1;
    const dz = p.Nz > 1 ? (2 * p.L_obs_Z) / (p.Nz - 1) : 1;
    download(writeTiff(result.data, result.nz, result.ny, result.nx, { dx: dxy, dy: dxy, dz }), 'psf.tif');
  };

  const downloadJson = () => {
    download(new Blob([JSON.stringify(params, null, 2)], { type: 'application/json' }), 'psf_config.json');
  };

  const volume = useMemo(() => (result ? psfToVolume(result) : null), [result]);
  const scalarForResult = scalarResult && result && scalarResult.params === result.params ? scalarResult : null;
  const scalarPending = pane === 'compare' && !!result && !scalarForResult;

  const status = useMemo(() => {
    if (workerStatus === 'loading') return { text: 'Loading simulator…', tone: 'muted' as const };
    if (workerStatus === 'error') return { text: workerError ?? 'simulator failed', tone: 'error' as const };
    if (error) return { text: error, tone: 'error' as const };
    if (busy || scalarPending) return { text: 'Computing…', tone: 'busy' as const };
    if (result) return { text: `${result.nx}×${result.ny}×${result.nz} in ${result.ms.toFixed(0)} ms`, tone: 'ok' as const };
    return { text: 'Ready', tone: 'muted' as const };
  }, [workerStatus, workerError, error, busy, scalarPending, result]);

  const renderControls = (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-xs">
      <div className="inline-flex overflow-hidden rounded-md border">
        {(['mip', 'composite'] as const).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setRender((r) => ({ ...r, mode: m }))}
            className={cn('px-2 py-1', render.mode === m ? 'bg-primary text-primary-foreground' : 'text-muted-foreground')}
          >
            {m === 'mip' ? 'Max projection' : 'Composite'}
          </button>
        ))}
      </div>
      <select
        className="rounded-md border bg-background px-2 py-1"
        value={render.colormap}
        onChange={(e) => setRender((r) => ({ ...r, colormap: e.target.value as ColormapName }))}
      >
        {COLORMAPS.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </select>
      <label className="inline-flex items-center gap-1">
        <input type="checkbox" checked={render.log} onChange={(e) => setRender((r) => ({ ...r, log: e.target.checked }))} />
        log
      </label>
      {render.log ? (
        <label className="inline-flex items-center gap-1">
          decades
          <input
            type="range"
            className="pg-range w-16"
            min={1}
            max={7}
            step={1}
            value={render.logDecades}
            onChange={(e) => setRender((r) => ({ ...r, logDecades: Number(e.target.value) }))}
          />
          {render.logDecades}
        </label>
      ) : (
        <label className="inline-flex items-center gap-1">
          γ
          <input
            type="range"
            className="pg-range w-16"
            min={0.2}
            max={2}
            step={0.05}
            value={render.gamma}
            onChange={(e) => setRender((r) => ({ ...r, gamma: Number(e.target.value) }))}
          />
          {render.gamma.toFixed(2)}
        </label>
      )}
      <label className="inline-flex items-center gap-1">
        cut
        <input
          type="range"
          className="pg-range w-16"
          min={0}
          max={0.9}
          step={0.01}
          value={render.threshold}
          onChange={(e) => setRender((r) => ({ ...r, threshold: Number(e.target.value) }))}
        />
      </label>
      {render.mode === 'composite' && (
        <label className="inline-flex items-center gap-1">
          opacity
          <input
            type="range"
            className="pg-range w-16"
            min={0.05}
            max={2}
            step={0.05}
            value={render.opacity}
            onChange={(e) => setRender((r) => ({ ...r, opacity: Number(e.target.value) }))}
          />
        </label>
      )}
    </div>
  );

  return (
    <div className="flex flex-col gap-4 px-4 pb-10 pt-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="mr-auto">
          <h1 className="text-2xl font-bold tracking-tight">Playground</h1>
          <p className="text-sm text-muted-foreground">
            The faser simulator running as WebAssembly in your browser. Nothing leaves your machine.
          </p>
        </div>
        <span
          className={cn(
            'inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs',
            status.tone === 'error' && 'border-red-500/40 bg-red-500/10 text-red-600 dark:text-red-400',
            status.tone === 'busy' && 'border-primary/40 bg-primary/10 text-primary',
            status.tone === 'ok' && 'text-muted-foreground',
            status.tone === 'muted' && 'text-muted-foreground',
          )}
        >
          {status.tone === 'busy' && <span className="size-2 animate-pulse rounded-full bg-primary" />}
          {status.text}
        </span>
      </div>

      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <label className="flex items-center gap-2">
          <Tag className="size-4 text-muted-foreground" />
          <select
            className="rounded-md border bg-background px-2 py-1.5 text-sm"
            value={preset}
            onChange={(e) => applyPreset(Number(e.target.value))}
            title={PRESETS[preset].description}
          >
            {PRESETS.map((p, i) => (
              <option key={p.name} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => setAuto((a) => !a)}
          className={cn(
            'inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5',
            auto ? 'border-primary/40 bg-primary/10 text-primary' : 'text-muted-foreground',
          )}
          title="Recompute the volume whenever a parameter changes"
        >
          <Zap className="size-4" />
          Live
        </button>
        <button
          type="button"
          onClick={() => generate(params)}
          disabled={workerStatus !== 'ready'}
          className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 font-medium text-primary-foreground disabled:opacity-50"
        >
          <Play className="size-4" />
          Generate
        </button>
        <span className="mx-1 h-5 w-px bg-border" />
        <button
          type="button"
          onClick={downloadTiff}
          disabled={!result}
          className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 disabled:opacity-50"
          title="32-bit multi-page TIFF with voxel size, opens in Fiji and napari"
        >
          <Download className="size-4" />
          TIFF
        </button>
        <button
          type="button"
          onClick={downloadJson}
          className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5"
          title="psf_config.json, readable by the CLI and the napari plugin"
        >
          <Download className="size-4" />
          Config
        </button>
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5"
          title="Load a psf_config.json"
        >
          <FolderOpen className="size-4" />
          Load
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void loadJson(f);
            e.target.value = '';
          }}
        />
        <button
          type="button"
          onClick={() => applyPreset(0)}
          className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-muted-foreground"
        >
          <RotateCcw className="size-4" />
          Reset
        </button>
        {auto && !autoActive && (
          <span className="text-xs text-amber-600 dark:text-amber-400">
            Grid is {cost.toFixed(0)}× the default; live updates paused, press Generate.
          </span>
        )}
      </div>

      {/* Main grid */}
      <div className="grid gap-4 xl:grid-cols-[330px_minmax(0,1fr)]">
        <aside className="xl:max-h-[calc(100vh-12rem)] xl:overflow-y-auto xl:pr-1">
          <ParamPanel params={params} onChange={update} />
        </aside>

        <div className="flex min-w-0 flex-col gap-4">
          {/* Pane tabs */}
          <div role="tablist" className="flex flex-wrap gap-1 border-b">
            {PANES.map((p) => (
              <button
                key={p.id}
                role="tab"
                type="button"
                aria-selected={pane === p.id}
                onClick={() => setPane(p.id)}
                className={cn(
                  '-mb-px border-b-2 px-3 py-2 text-sm',
                  pane === p.id
                    ? 'border-primary font-semibold text-foreground'
                    : 'border-transparent text-muted-foreground hover:text-foreground',
                )}
              >
                {p.label}
              </button>
            ))}
          </div>

          {pane === 'psf' && (
            <div className="grid gap-4 lg:grid-cols-2">
              <section className="flex min-w-0 flex-col gap-3">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                  <h2 className="mr-auto text-sm font-semibold">PSF volume</h2>
                  {renderControls}
                </div>
                <div className="relative h-[420px] overflow-hidden rounded-xl border bg-[#0b0b10]">
                  <VolumeViewer volume={volume} settings={render} />
                  {busy && <div className="pointer-events-none absolute right-3 top-3 size-2 animate-pulse rounded-full bg-primary" />}
                </div>
                {result && (
                  <SliceViews
                    result={result}
                    colormap={render.colormap}
                    log={render.log}
                    logDecades={render.logDecades}
                    gamma={render.gamma}
                  />
                )}
              </section>

              <section className="flex min-w-0 flex-col gap-3">
                <div className="flex items-center gap-3 text-xs">
                  <h2 className="mr-auto text-sm font-semibold">Microscope</h2>
                  <label className="inline-flex items-center gap-1">
                    <input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} />
                    labels
                  </label>
                </div>
                <div className="h-[420px] overflow-hidden rounded-xl border bg-gradient-to-b from-card to-background">
                  <MicroscopeScene params={params} derived={derived} showLabels={showLabels} />
                </div>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-1 rounded-lg border bg-card p-3 text-xs sm:grid-cols-3">
                  <Stat label="α (immersion)" value={derived ? `${fmt((derived.alpha * 180) / Math.PI, 1)}°` : '–'} />
                  <Stat label="α (sample)" value={derived ? `${fmt((derived.alpha3_eff * 180) / Math.PI, 1)}°` : '–'} />
                  <Stat label="effective NA" value={derived ? fmt(derived.na_eff, 3) : '–'} />
                  <Stat label="pupil radius r₀" value={derived ? `${fmt(derived.r0, 0)} µm` : '–'} />
                  <Stat label="focus shift Δz" value={derived ? `${derived.dfoc > 0 ? '+' : ''}${fmt(derived.dfoc, 3)} µm` : '–'} />
                  <Stat label="k₀" value={derived ? `${fmt(derived.k0, 3)} µm⁻¹` : '–'} />
                  <Stat label="voxel xy" value={`${fmt(params.Nxy > 1 ? (2 * params.L_obs_XY) / (params.Nxy - 1) : 0, 4)} µm`} />
                  <Stat label="voxel z" value={`${fmt(params.Nz > 1 ? (2 * params.L_obs_Z) / (params.Nz - 1) : 0, 4)} µm`} />
                  <Stat label="λ / (2 NA)" value={`${fmt(params.Wavelength / (2 * params.NA), 3)} µm`} />
                </dl>
                <p className="text-xs text-muted-foreground">
                  Schematic, not to scale: the coverslip is drawn to its thickness, the imaging depth is compressed so deep foci
                  still fit, and the objective sits at a working distance that follows the NA. The cone angles are the real
                  refraction angles in immersion, coverslip and sample; a shifted focus (index mismatch) shows the nominal focus
                  as a wire sphere. The disc on top of the objective is the back pupil as the simulator sees it (brightness =
                  amplitude, hue = phase) with the incident polarization drawn on it.
                </p>
              </section>
            </div>
          )}

          {pane === 'compare' &&
            (result ? (
              <>
                <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                  <h2 className="mr-auto text-sm font-semibold">Vectorial vs scalar</h2>
                  {renderControls}
                </div>
                <ComparePane
                  vectorial={result}
                  scalar={scalarForResult}
                  colormap={render.colormap}
                  log={render.log}
                  logDecades={render.logDecades}
                  gamma={render.gamma}
                  busy={busy || scalarPending}
                />
              </>
            ) : (
              <div className="flex h-48 items-center justify-center rounded-xl border text-sm text-muted-foreground">
                Generate a PSF first
              </div>
            ))}

          {pane === 'simulate' && (
            <SimulationPane
              psf={result}
              ready={workerStatus === 'ready'}
              generateSample={workerSample}
              convolve={workerConvolve}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-medium tabular-nums">{value}</dd>
    </div>
  );
}
