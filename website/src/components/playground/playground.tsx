'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Download, FolderOpen, Play, RotateCcw, Tag, Zap } from 'lucide-react';
import { cn } from '@/lib/cn';
import { COLORMAPS, type ColormapName } from './colormaps';
import { ComparePane } from './compare-pane';
import { Inspector } from './inspector';
import { MicroscopeScene } from './microscope-scene';
import { coerceParams, DEFAULTS, GRID_KEYS, PRESETS, workUnits, ZERNIKE_KEYS, ZERO_ZERNIKE, type ComponentId, type Derived, type Params, type ZernikeCoeffs } from './params';
import { autoGrid, previewGrid } from './physics';
import { SimulationPane } from './simulation-pane';
import { buildSlm, DEFAULT_SLM_DESIGN, designFromSlm, hasZernike, negatedZernike, type SlmDesign } from './slm';
import { zernikeRange } from './zernike';
import { writeTiff } from './tiff';
import { usePsfWorker, type PsfResult } from './use-psf-worker';
import { psfToVolume } from './volume';
import { DEFAULT_RENDER, type RenderSettings } from './volume-viewer';
import { PsfInset } from './psf-inset';

/** Above this estimated time the accurate volume waits for "Generate". */
const AUTO_MS_LIMIT = 6000;
const PREVIEW_DEBOUNCE = 40;
const FINAL_DEBOUNCE = 350;

type Pane = 'compare' | 'simulate';

const PANES: { id: Pane; label: string }[] = [
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

/** Runs `generate` for the newest requested params once the previous run is done. */
function useLatestRunner(generate: (p: Params) => Promise<PsfResult>, onResult: (r: PsfResult) => void, onError: (e: string) => void) {
  const wanted = useRef<Params | null>(null);
  const running = useRef(false);
  const [busy, setBusy] = useState(false);
  const run = useCallback(async () => {
    if (running.current) return;
    running.current = true;
    setBusy(true);
    try {
      while (wanted.current) {
        const next = wanted.current;
        wanted.current = null;
        try {
          onResult(await generate(next));
        } catch (e) {
          onError(e instanceof Error ? e.message : String(e));
        }
      }
    } finally {
      running.current = false;
      setBusy(false);
    }
  }, [generate, onResult, onError]);
  const request = useCallback(
    (p: Params) => {
      wanted.current = p;
      void run();
    },
    [run],
  );
  return { request, busy };
}

export function Playground() {
  // Two simulator instances: a fast preview that follows every edit, and the
  // accurate one that catches up once the parameters settle.
  const previewWorker = usePsfWorker();
  const finalWorker = usePsfWorker();
  const ready = previewWorker.status === 'ready' && finalWorker.status === 'ready';
  const workerError = previewWorker.error ?? finalWorker.error;
  const { derive, generate: previewGenerate } = previewWorker;
  const { generate: finalGenerate, sample: generateSample, convolve } = finalWorker;

  const [params, setParams] = useState<Params>(DEFAULTS);
  const [design, setDesign] = useState<SlmDesign>(DEFAULT_SLM_DESIGN);
  const [autoGridOn, setAutoGridOn] = useState(true);
  const [selected, setSelected] = useState<ComponentId | null>(null);
  const [hovered, setHovered] = useState<ComponentId | null>(null);
  const [result, setResult] = useState<PsfResult | null>(null);
  const [scalarResult, setScalarResult] = useState<PsfResult | null>(null);
  const [derived, setDerived] = useState<Derived | null>(null);
  const [render, setRender] = useState<RenderSettings>(DEFAULT_RENDER);
  const [live, setLive] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [alwaysLabels, setAlwaysLabels] = useState(false);
  const [preset, setPreset] = useState(0);
  const [pane, setPane] = useState<Pane | null>(null);
  const [focusPt, setFocusPt] = useState<{ x: number; y: number } | null>(null);
  // ms per 1e6 work units, a running average of measured accurate runs
  // (~2.6 for V8/wasm on a desktop core, slower on laptops).
  const [msPerMega, setMsPerMega] = useState(4);
  const fileRef = useRef<HTMLInputElement>(null);

  // The parameters that are simulated: the automatic grid resolved. The
  // Zernike layer on the SLM adds pupil phase the grid must resolve too.
  const slmPhaseRange = design.enabled ? zernikeRange(design.zernike) : 0;
  const effective = useMemo<Params>(
    () => (autoGridOn ? { ...params, ...autoGrid(params, slmPhaseRange) } : params),
    [params, autoGridOn, slmPhaseRange],
  );
  const previewParams = useMemo<Params>(
    () => ({ ...effective, ...previewGrid({ Nxy: effective.Nxy, Nz: effective.Nz, Ntheta: effective.Ntheta, Nphi: effective.Nphi }) }),
    [effective],
  );
  const effectiveRef = useRef(effective);
  useEffect(() => {
    effectiveRef.current = effective;
  }, [effective]);
  const estimateMs = useMemo(() => (msPerMega * workUnits(effective)) / 1e6, [msPerMega, effective]);

  const onPreviewResult = useCallback((r: PsfResult) => {
    // never replace an accurate result for the current parameters with a preview
    setResult((cur) => (cur && cur.params === effectiveRef.current ? cur : r));
    setError(null);
  }, []);
  const onFinalResult = useCallback((r: PsfResult) => {
    const cost = workUnits(r.params);
    if (cost > 2e6 && r.ms > 5) setMsPerMega((m) => 0.7 * m + 0.3 * (r.ms / (cost / 1e6)));
    if (r.params !== effectiveRef.current) return; // stale: newer parameters are queued
    setResult(r);
    setError(null);
  }, []);
  const onError = useCallback((e: string) => setError(e), []);

  const { request: requestPreview, busy: previewBusy } = useLatestRunner(previewGenerate, onPreviewResult, onError);
  const { request: requestFinal, busy: finalBusy } = useLatestRunner(finalGenerate, onFinalResult, onError);

  // The displayed volume is accurate only if it was computed for exactly the
  // current parameters; anything else (preview grid, older parameters) is a preview.
  const quality: 'preview' | 'final' = result && result.params === effective ? 'final' : 'preview';
  const autoActive = live && estimateMs <= AUTO_MS_LIMIT;

  // Derived quantities and the preview follow every edit; the accurate
  // volume follows once the parameters settle.
  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    derive(effective)
      .then((d) => {
        if (!cancelled) {
          setDerived(d);
          setError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    const idPreview = setTimeout(() => requestPreview(previewParams), PREVIEW_DEBOUNCE);
    const idFinal = autoActive ? setTimeout(() => requestFinal(effective), FINAL_DEBOUNCE) : null;
    return () => {
      cancelled = true;
      clearTimeout(idPreview);
      if (idFinal) clearTimeout(idFinal);
    };
  }, [effective, previewParams, ready, autoActive, derive, requestPreview, requestFinal]);

  // Comparison pane: the scalar PSF for the displayed accurate result.
  useEffect(() => {
    if (pane !== 'compare' || !result || quality !== 'final') return;
    if (scalarResult && scalarResult.params === result.params) return;
    let cancelled = false;
    finalGenerate(result.params, true)
      .then((r) => {
        if (!cancelled) setScalarResult(r);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [pane, result, quality, scalarResult, finalGenerate]);

  const update = useCallback((patch: Partial<Params>) => setParams((p) => ({ ...p, ...patch })), []);

  const applyDesign = useCallback((next: SlmDesign) => {
    setDesign(next);
    setParams((p) => ({ ...p, SLM: buildSlm(next) }));
  }, []);
  const updateDesign = useCallback((patch: Partial<SlmDesign>) => applyDesign({ ...design, ...patch }), [design, applyDesign]);

  // Transfer the system aberrations to the SLM's Zernike layer: 'move'
  // displays them there (and clears the system ones), 'correct' displays
  // their negative. An SLM that was off comes on with a flat pattern (just
  // the layer); one that was on keeps its pattern underneath. The
  // aberration offsets stay with the system aberrations.
  const sendAberrations = useCallback(
    (mode: 'move' | 'correct') => {
      const coeffs = { ...ZERO_ZERNIKE } as ZernikeCoeffs;
      for (const k of ZERNIKE_KEYS) coeffs[k] = params[k];
      const nextDesign: SlmDesign = {
        ...design,
        enabled: true,
        pattern: design.enabled ? design.pattern : 'flat',
        zernike: mode === 'move' ? coeffs : negatedZernike(coeffs),
      };
      setDesign(nextDesign);
      setParams((p) => ({ ...p, ...(mode === 'move' ? ZERO_ZERNIKE : {}), SLM: buildSlm(nextDesign) }));
    },
    [params, design],
  );

  const applyPreset = (i: number) => {
    setPreset(i);
    const pr = PRESETS[i];
    const nextDesign: SlmDesign = pr.slm
      ? { ...DEFAULT_SLM_DESIGN, ...pr.slm, zernike: { ...ZERO_ZERNIKE, ...pr.slm.zernike }, enabled: true }
      : { ...design, enabled: false };
    setDesign(nextDesign);
    setParams({ ...DEFAULTS, ...pr.params, SLM: buildSlm(nextDesign) });
    setAutoGridOn(true);
  };

  const loadJson = async (file: File) => {
    try {
      const { params: p, hasGrid } = coerceParams(JSON.parse(await file.text()));
      setParams(p);
      setDesign(p.SLM ? designFromSlm(p.SLM, design) : { ...design, enabled: false });
      const auto = autoGrid(p);
      setAutoGridOn(!hasGrid || GRID_KEYS.every((k) => auto[k] === p[k]));
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
    download(writeTiff(result.data, result.nz, result.ny, result.nx, { dx: dxy, dy: dxy, dz }), quality === 'final' ? 'psf.tif' : 'psf_preview.tif');
  };

  const downloadJson = () => {
    download(new Blob([JSON.stringify(effective, null, 2)], { type: 'application/json' }), 'psf_config.json');
  };

  const volume = useMemo(() => (result ? psfToVolume(result) : null), [result]);
  const scalarForResult = scalarResult && result && scalarResult.params === result.params ? scalarResult : null;
  const scalarPending = pane === 'compare' && !!result && quality === 'final' && !scalarForResult;
  const finalPending = finalBusy || (autoActive && result?.params !== effective);

  const status = useMemo(() => {
    if (!ready && !workerError) return { text: 'Loading simulator…', tone: 'muted' as const };
    if (workerError) return { text: workerError, tone: 'error' as const };
    if (error) return { text: error, tone: 'error' as const };
    if (finalBusy || scalarPending) {
      return { text: `Computing ${effective.Nxy}² × ${effective.Nz}${estimateMs > 400 ? ` (≈ ${(estimateMs / 1000).toFixed(1)} s)` : ''}…`, tone: 'busy' as const };
    }
    if (result && quality === 'final') return { text: `${result.nx}×${result.ny}×${result.nz}, θ ${result.params.Ntheta} φ ${result.params.Nphi}, ${result.ms.toFixed(0)} ms`, tone: 'ok' as const };
    if (result) return { text: `Preview ${result.nx}×${result.ny}×${result.nz}${autoActive ? '' : ', press Generate for the accurate volume'}`, tone: 'preview' as const };
    return { text: 'Ready', tone: 'muted' as const };
  }, [ready, workerError, error, finalBusy, scalarPending, result, quality, effective, estimateMs, autoActive]);

  const updateRender = useCallback((patch: Partial<RenderSettings>) => setRender((r) => ({ ...r, ...patch })), []);

  return (
    <div className="flex flex-col gap-3 px-4 pb-10 pt-4 sm:px-6 lg:px-8">
      {/* Header + toolbar */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="mr-auto">
          <h1 className="text-2xl font-bold tracking-tight">Playground</h1>
          <p className="text-sm text-muted-foreground">
            The faser simulator as WebAssembly in your browser. Click any part of the microscope to change it.
          </p>
        </div>
        <span
          className={cn(
            'inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs',
            status.tone === 'error' && 'border-red-500/40 bg-red-500/10 text-red-600 dark:text-red-400',
            status.tone === 'busy' && 'border-primary/40 bg-primary/10 text-primary',
            status.tone === 'preview' && 'border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300',
            (status.tone === 'ok' || status.tone === 'muted') && 'text-muted-foreground',
          )}
        >
          {status.tone === 'busy' && <span className="size-2 animate-pulse rounded-full bg-primary" />}
          {status.text}
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-sm">
        <label className="flex items-center gap-2">
          <Tag className="size-4 text-muted-foreground" />
          <select
            className="max-w-[16rem] rounded-md border bg-background px-2 py-1.5 text-sm"
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
          onClick={() => setLive((a) => !a)}
          className={cn('inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5', live ? 'border-primary/40 bg-primary/10 text-primary' : 'text-muted-foreground')}
          title="Compute the accurate volume automatically whenever a parameter settles"
        >
          <Zap className="size-4" />
          Live
        </button>
        <button
          type="button"
          onClick={() => requestFinal(effective)}
          disabled={!ready}
          className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 font-medium text-primary-foreground disabled:opacity-50"
          title="Compute the accurate volume now"
        >
          <Play className="size-4" />
          Generate
        </button>
        <span className="mx-1 h-5 w-px bg-border" />
        <button type="button" onClick={downloadTiff} disabled={!result} className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 disabled:opacity-50" title="32-bit multi-page TIFF with voxel size, opens in Fiji and napari">
          <Download className="size-4" />
          TIFF
        </button>
        <button type="button" onClick={downloadJson} className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5" title="psf_config.json (with the resolved grid and the SLM pattern), readable by the CLI and the napari plugin">
          <Download className="size-4" />
          Config
        </button>
        <button type="button" onClick={() => fileRef.current?.click()} className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5" title="Load a psf_config.json">
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
        <button type="button" onClick={() => applyPreset(0)} className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-muted-foreground">
          <RotateCcw className="size-4" />
          Reset
        </button>
        <label className="ml-auto inline-flex items-center gap-1.5 text-xs text-muted-foreground" title="Otherwise labels appear when you hover a part">
          <input type="checkbox" checked={alwaysLabels} onChange={(e) => setAlwaysLabels(e.target.checked)} />
          all labels
        </label>
        {live && !autoActive && (
          <span className="basis-full text-xs text-amber-600 dark:text-amber-400">
            The accurate volume takes ≈ {(estimateMs / 1000).toFixed(1)} s; live updates show the preview only, press Generate for the accurate one.
          </span>
        )}
      </div>

      {/* Hero: the microscope with the PSF zoom-in floating at its right, and the inspector next to it */}
      <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_370px]">
        <div className="flex h-[58vh] min-h-[440px] flex-col overflow-hidden rounded-xl border bg-gradient-to-b from-card to-background sm:flex-row">
          <div className="relative min-h-0 min-w-0 flex-1">
            <MicroscopeScene
              params={effective}
              derived={derived}
              labels={alwaysLabels ? 'always' : 'hover'}
              selected={selected}
              hovered={hovered}
              onSelect={setSelected}
              onHover={setHovered}
              slmZernike={design.enabled && hasZernike(design.zernike)}
              onFocusScreen={setFocusPt}
            />
            {/* loupe on the focus and the leader line to the zoom-in */}
            {focusPt && (
              <svg className="pointer-events-none absolute inset-0 hidden h-full w-full sm:block" aria-hidden>
                <circle cx={focusPt.x} cy={focusPt.y} r={22} fill="none" stroke="var(--color-fd-primary)" strokeWidth={1.5} strokeDasharray="4 3" />
                <line x1={focusPt.x + 22} y1={focusPt.y} x2="100%" y2={44} stroke="var(--color-fd-primary)" strokeWidth={1.5} strokeDasharray="4 3" />
              </svg>
            )}
            {(previewBusy || finalPending) && <div className="pointer-events-none absolute right-3 top-3 size-2 animate-pulse rounded-full bg-primary" />}
          </div>
          <aside className="h-[46vh] w-full shrink-0 border-t bg-card/80 backdrop-blur sm:h-auto sm:w-[300px] sm:border-l sm:border-t-0">
            <PsfInset result={result} volume={volume} quality={quality} render={render} onRender={updateRender} />
          </aside>
        </div>
        <aside className="h-[58vh] min-h-[440px] overflow-hidden rounded-xl border bg-card">
          <Inspector
            params={params}
            effective={effective}
            derived={derived}
            selected={selected}
            onSelect={setSelected}
            onChange={update}
            design={design}
            onDesign={updateDesign}
            onSendAberrations={sendAberrations}
            autoGridOn={autoGridOn}
            onAutoGrid={setAutoGridOn}
            estimateMs={estimateMs}
          />
        </aside>
      </div>

      {/* Analyses */}
      <div role="tablist" className="flex flex-wrap items-center gap-1 border-b">
        <span className="px-1 py-2 text-xs text-muted-foreground">Analyses</span>
        {PANES.map((p) => (
          <button
            key={p.id}
            role="tab"
            type="button"
            aria-selected={pane === p.id}
            onClick={() => setPane((cur) => (cur === p.id ? null : p.id))}
            className={cn('-mb-px border-b-2 px-3 py-2 text-sm', pane === p.id ? 'border-primary font-semibold text-foreground' : 'border-transparent text-muted-foreground hover:text-foreground')}
          >
            {p.label}
          </button>
        ))}
      </div>

      {pane === 'compare' &&
        (result && quality === 'final' ? (
          <>
            <h2 className="text-sm font-semibold">Vectorial vs scalar</h2>
            <ComparePane vectorial={result} scalar={scalarForResult} colormap={render.colormap} log={render.log} logDecades={render.logDecades} gamma={render.gamma} busy={finalBusy || scalarPending} />
          </>
        ) : (
          <div className="flex h-48 items-center justify-center rounded-xl border text-sm text-muted-foreground">
            {result ? 'Waiting for the accurate volume…' : 'Generate a PSF first'}
          </div>
        ))}

      {pane === 'simulate' && <SimulationPane psf={quality === 'final' ? result : null} ready={ready} generateSample={generateSample} convolve={convolve} />}
    </div>
  );
}
