'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Microscope, Play, RotateCcw, Settings2, SlidersHorizontal, Target, Zap } from 'lucide-react';
import { ScenePicker } from './scene-picker';
import { ExportMenu } from './export-menu';
import { sceneState, type SavedScene } from './scenes';
import { cn } from '@/lib/cn';
import { COLORMAPS, type ColormapName } from './colormaps';
import { Inspector } from './inspector';
import { MicroscopeScene } from './microscope-scene';
import { coerceParams, DEFAULTS, GRID_KEYS, PRESETS, workUnits, ZERNIKE_KEYS, ZERO_ZERNIKE, type ComponentId, type Derived, type Params, type ZernikeCoeffs } from './params';
import { autoGrid, previewGrid } from './physics';
import { buildSlm, DEFAULT_SLM_DESIGN, designFromSlm, hasZernike, negatedZernike, type SlmDesign } from './slm';
import { zernikeRange } from './zernike';
import { writeTiff } from './tiff';
import { usePsfWorker, type PsfResult } from './use-psf-worker';
import { psfToVolume } from './volume';
import { DEFAULT_RENDER, type RenderSettings } from './volume-viewer';
import { PsfInset } from './psf-inset';
import { DEFAULT_IMAGING, useImaging, type ImagingSettings } from './use-imaging';

/** Above this estimated time the accurate volume waits for "Generate". */
const AUTO_MS_LIMIT = 6000;
const PREVIEW_DEBOUNCE = 40;
const FINAL_DEBOUNCE = 350;

function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** True while the media query matches (false during the first render). */
function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const update = () => setMatches(mq.matches);
    update();
    mq.addEventListener('change', update);
    return () => mq.removeEventListener('change', update);
  }, [query]);
  return matches;
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
  const [derived, setDerived] = useState<Derived | null>(null);
  const [render, setRender] = useState<RenderSettings>(DEFAULT_RENDER);
  const [live, setLive] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [alwaysLabels, setAlwaysLabels] = useState(false);
  const [showDetector, setShowDetector] = useState(false);
  const [viewMenu, setViewMenu] = useState(false);
  // The settings sidebar: a sticky column on wide screens, a slide-over drawer otherwise.
  const wide = useMediaQuery('(min-width: 1024px)');
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [sidebarVisible, setSidebarVisible] = useState(true);
  // What the main area shows: the microscope scene with the PSF zoom-in, or the PSF alone, large.
  const [view, setView] = useState<'scene' | 'psf'>('scene');
  const selectPart = useCallback(
    (id: ComponentId | null) => {
      setSelected(id);
      if (id && !wide) setDrawerOpen(true);
      if (id && wide) setSidebarVisible(true);
    },
    [wide],
  );
  const [sceneName, setSceneName] = useState(PRESETS[0].name);
  // Snapshot of the state as loaded from a scene; the setup is "edited" once it differs.
  const [loaded, setLoaded] = useState<{ params: Params; design: SlmDesign; imaging: ImagingSettings; autoGridOn: boolean }>(() => ({
    params: DEFAULTS,
    design: DEFAULT_SLM_DESIGN,
    imaging: DEFAULT_IMAGING,
    autoGridOn: true,
  }));
  const [focusPt, setFocusPt] = useState<{ x: number; y: number } | null>(null);
  const [imaging, setImaging] = useState<ImagingSettings>(DEFAULT_IMAGING);
  const updateImaging = useCallback((patch: Partial<ImagingSettings>) => setImaging((s) => ({ ...s, ...patch })), []);
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

  /** Put a whole scene state in place and remember it as the reference for "edited". */
  const loadState = useCallback((name: string, st: { params: Params; design: SlmDesign; imaging: ImagingSettings; autoGridOn: boolean }) => {
    setSceneName(name);
    setParams(st.params);
    setDesign(st.design);
    setImaging(st.imaging);
    setAutoGridOn(st.autoGridOn);
    setLoaded(st);
  }, []);

  const applyPreset = (i: number) => {
    const pr = PRESETS[i];
    const nextDesign: SlmDesign = pr.slm
      ? { ...DEFAULT_SLM_DESIGN, ...pr.slm, zernike: { ...ZERO_ZERNIKE, ...pr.slm.zernike }, enabled: true }
      : { ...design, enabled: false };
    loadState(pr.name, { params: { ...DEFAULTS, ...pr.params, SLM: buildSlm(nextDesign) }, design: nextDesign, imaging: DEFAULT_IMAGING, autoGridOn: true });
  };

  const applySaved = (scene: SavedScene) => loadState(scene.name, sceneState(scene));

  const dirty = useMemo(() => {
    if (autoGridOn !== loaded.autoGridOn || imaging !== loaded.imaging) return true;
    const { SLM: a, ...pa } = params;
    const { SLM: b, ...pb } = loaded.params;
    if (JSON.stringify(pa) !== JSON.stringify(pb)) return true;
    if ((a === null) !== (b === null)) return true;
    // the SLM pattern is fully described by the design
    const { image: ia, custom: ca, ...da } = design;
    const { image: ib, custom: cb, ...db } = loaded.design;
    return ia !== ib || ca !== cb || JSON.stringify(da) !== JSON.stringify(db);
  }, [params, design, imaging, autoGridOn, loaded]);

  const loadJson = async (file: File) => {
    try {
      const { params: p, hasGrid } = coerceParams(JSON.parse(await file.text()));
      const auto = autoGrid(p);
      loadState(file.name.replace(/\.json$/i, ''), {
        params: p,
        design: p.SLM ? designFromSlm(p.SLM, design) : { ...design, enabled: false },
        imaging,
        autoGridOn: !hasGrid || GRID_KEYS.every((k) => auto[k] === p[k]),
      });
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
  // The sample at the focus and its image, when the Sample card asks for one.
  const sim = useImaging(imaging, quality === 'final' ? result : null, ready, generateSample, convolve);
  const finalPending = finalBusy || (autoActive && result?.params !== effective);

  const status = useMemo(() => {
    if (!ready && !workerError) return { text: 'Loading simulator…', tone: 'muted' as const };
    if (workerError) return { text: workerError, tone: 'error' as const };
    if (error) return { text: error, tone: 'error' as const };
    if (finalBusy) {
      return { text: `computing ${effective.Nxy}² × ${effective.Nz}${estimateMs > 400 ? ` · ≈ ${(estimateMs / 1000).toFixed(1)} s` : ''}`, tone: 'busy' as const };
    }
    if (result && quality === 'final') return { text: `${result.nx}×${result.ny}×${result.nz}, θ ${result.params.Ntheta} φ ${result.params.Nphi}, ${result.ms.toFixed(0)} ms`, tone: 'ok' as const };
    if (result && !autoActive) return { text: `preview · accurate volume ≈ ${(estimateMs / 1000).toFixed(1)} s, press Generate`, tone: 'preview' as const };
    if (result) return { text: `preview ${result.nx}×${result.ny}×${result.nz}`, tone: 'preview' as const };
    return { text: 'Ready', tone: 'muted' as const };
  }, [ready, workerError, error, finalBusy, result, quality, effective, estimateMs, autoActive]);

  const updateRender = useCallback((patch: Partial<RenderSettings>) => setRender((r) => ({ ...r, ...patch })), []);

  /** View toggle, view settings, compute and file actions: floating over the renderer, or a row above the PSF view. */
  const controls = (floating: boolean) => (
      <div
        className={cn(
          'flex flex-wrap items-center gap-2 text-sm',
          floating && 'pointer-events-none absolute left-3 top-3 z-10 max-w-[calc(100%-1.5rem)] [&>*]:pointer-events-auto',
        )}
      >
        <div className="inline-flex items-center rounded-full border p-0.5" role="tablist" aria-label="View">
          {(['scene', 'psf'] as const).map((v) => {
            const Icon = v === 'scene' ? Microscope : Target;
            return (
              <button
                key={v}
                type="button"
                role="tab"
                aria-selected={view === v}
                onClick={() => setView(v)}
                className={cn(
                  'inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs transition-colors',
                  view === v ? 'bg-accent text-accent-foreground' : 'text-muted-foreground hover:text-foreground',
                )}
              >
                <Icon className="size-3.5" />
                {v === 'scene' ? 'Microscope' : 'PSF'}
              </button>
            );
          })}
        </div>
        {view === 'scene' && (
          <>
            <div className="relative text-xs">
              <button
                type="button"
                className={cn('rounded-md border bg-background/80 p-1.5 text-muted-foreground backdrop-blur hover:text-foreground', viewMenu && 'text-foreground')}
                onClick={() => setViewMenu((v) => !v)}
                aria-expanded={viewMenu}
                aria-label="View settings"
                title="View settings"
              >
                <Settings2 className="size-4" />
              </button>
              {viewMenu && (
                <div className="absolute left-0 top-full z-20 mt-1 flex w-56 flex-col gap-2 rounded-lg border bg-popover p-3 shadow-md">
                  <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">View</div>
                  <label className="flex items-center justify-between gap-2">
                    <span>Show image plane (camera)</span>
                    <input type="checkbox" className="accent-primary" checked={showDetector} onChange={(e) => setShowDetector(e.target.checked)} />
                  </label>
                  <label className="flex items-center justify-between gap-2" title="Otherwise labels appear when you hover a part">
                    <span>Always show labels</span>
                    <input type="checkbox" className="accent-primary" checked={alwaysLabels} onChange={(e) => setAlwaysLabels(e.target.checked)} />
                  </label>
                </div>
              )}
            </div>
          </>
        )}
      </div>
  );

  /** Compute, export and reset on one line: the footer of the settings card. */
  const actions = (
    <div className="flex items-center justify-between gap-2 text-sm">
      <div className="inline-flex overflow-hidden rounded-md border">
        <button
          type="button"
          onClick={() => setLive((a) => !a)}
          className={cn('inline-flex items-center gap-1.5 px-2.5 py-1.5', live ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground')}
          title="Compute the accurate volume automatically whenever a parameter settles"
          aria-pressed={live}
        >
          <Zap className="size-4" />
          Live
        </button>
        <button
          type="button"
          onClick={() => requestFinal(effective)}
          disabled={!ready}
          className="inline-flex items-center gap-1.5 border-l bg-primary px-3 py-1.5 font-medium text-primary-foreground disabled:opacity-50"
          title="Compute the accurate volume now"
        >
          <Play className="size-4" />
          Generate
        </button>
      </div>
      <div className="inline-flex items-center gap-1.5">
        <ExportMenu
          image={
            sim.image
              ? { volume: sim.image, brightestPlane: false, colormap: render.colormap, mapping: render.log ? { kind: 'log', decades: render.logDecades } : { kind: 'linear', gamma: render.gamma } }
              : volume
                ? { volume, brightestPlane: true, colormap: render.colormap, mapping: render.log ? { kind: 'log', decades: render.logDecades } : { kind: 'linear', gamma: render.gamma } }
                : null
          }
          onTiff={downloadTiff}
          onConfig={downloadJson}
          download={download}
          disabledVolume={!result}
          openUpward
        />
        <button
          type="button"
          onClick={() => applyPreset(0)}
          className="inline-flex items-center rounded-md border p-1.5 text-muted-foreground hover:text-foreground"
          title="Reset to the default scene"
          aria-label="Reset to the default scene"
        >
          <RotateCcw className="size-4" />
        </button>
      </div>
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
    </div>
  );

  const scenePicker = (
    <ScenePicker
      variant="inline"
      currentName={sceneName}
      dirty={dirty}
      busy={finalBusy || previewBusy || !ready}
      onPreset={applyPreset}
      onSaved={(scene) => applySaved(scene)}
      onLoadFile={() => fileRef.current?.click()}
      getState={() => ({ params, design, imaging, autoGridOn })}
    />
  );

  const inspector = (
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
            imaging={imaging}
            onImaging={updateImaging}
            imagingStatus={{ voxel: sim.voxel, tooLarge: sim.tooLarge, error: sim.error, busy: sim.busy }}
            onClose={wide ? () => setSidebarVisible(false) : () => setDrawerOpen(false)}
            scenePicker={scenePicker}
            footer={actions}
          />
  );

  return (
    <div className="mx-auto flex w-full max-w-[var(--fd-layout-width)] items-start lg:gap-4 lg:px-4">
      <div className="relative flex h-[calc(100dvh-3.5rem)] min-w-0 flex-1 flex-col lg:py-4">
      {/* The microscope with the PSF zoom-in, or the PSF on its own */}
      {view === 'psf' ? (
        <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-2 overflow-y-auto px-3 pb-3 pt-3">
          {controls(false)}
          <PsfInset result={result} volume={volume} quality={quality} render={render} onRender={updateRender} image={sim.image} layout="wide" />
        </div>
      ) : (
      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden sm:flex-row">
          <div className="relative h-[55vh] min-h-[320px] min-w-0 flex-1 sm:h-full sm:min-h-0">
            {controls(true)}
            <MicroscopeScene
              params={effective}
              derived={derived}
              labels={alwaysLabels ? 'always' : 'hover'}
              selected={selected}
              hovered={hovered}
              onSelect={selectPart}
              onHover={setHovered}
              slmZernike={design.enabled && hasZernike(design.zernike)}
              onFocusScreen={setFocusPt}
              sample={sim.sample}
              detector={sim.image ? { volume: sim.image, brightestPlane: false, render } : volume ? { volume, brightestPlane: true, render } : null}
              showDetector={showDetector}
            />
            {/* loupe on the focus and the leader line to the zoom-in */}
            {focusPt && (
              <svg className="pointer-events-none absolute inset-0 hidden h-full w-full sm:block" aria-hidden>
                <circle cx={focusPt.x} cy={focusPt.y} r={sim.sample ? 46 : 22} fill="none" stroke="var(--color-fd-primary)" strokeWidth={1.5} strokeDasharray="4 3" />
                <line x1={focusPt.x + (sim.sample ? 46 : 22)} y1={focusPt.y} x2="100%" y2={44} stroke="var(--color-fd-primary)" strokeWidth={1.5} strokeDasharray="4 3" />
              </svg>
            )}
            {/* progress / problems, small, bottom-right of the renderer */}
            {status.tone !== 'ok' && status.tone !== 'muted' && (
              <span
                className={cn(
                  'pointer-events-none absolute bottom-3 right-3 z-10 inline-flex max-w-[60%] items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] leading-4 backdrop-blur',
                  status.tone === 'error' && 'border-red-500/40 bg-red-500/10 text-red-600 dark:text-red-400',
                  status.tone === 'busy' && 'border-primary/40 bg-primary/10 text-primary',
                  status.tone === 'preview' && 'border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300',
                )}
              >
                {status.tone === 'busy' && <span className="size-1.5 shrink-0 animate-pulse rounded-full bg-primary" />}
                <span className="truncate">{status.text}</span>
              </span>
            )}
          </div>
          <aside className="w-full shrink-0 border-l bg-background/70 backdrop-blur sm:h-full sm:w-[clamp(210px,30%,320px)]">
            <PsfInset result={result} volume={volume} quality={quality} render={render} onRender={updateRender} image={sim.image} />
          </aside>
      </div>
      )}

      </div>

      {/* Settings: sidebar on wide screens, drawer otherwise (rendered once) */}
      {wide ? (
        <>
          {/* the card slides out to the right and its column collapses */}
          <div
            className={cn(
              'sticky top-[4.5rem] hidden h-[calc(100dvh-5.5rem)] shrink-0 overflow-hidden transition-[width] duration-300 ease-out lg:block',
              sidebarVisible ? 'w-[clamp(300px,27vw,400px)]' : 'w-0',
            )}
            aria-hidden={!sidebarVisible}
          >
            <aside
              className={cn(
                'mt-4 h-[calc(100%-1rem)] w-[clamp(300px,27vw,400px)] overflow-hidden rounded-xl border bg-card shadow-md transition-transform duration-300 ease-out',
                sidebarVisible ? 'translate-x-0' : 'translate-x-[calc(100%+1rem)]',
              )}
            >
              {inspector}
            </aside>
          </div>
          {!sidebarVisible && (
            <button
              type="button"
              className="fixed bottom-4 right-4 z-30 inline-flex items-center gap-2 rounded-full border bg-background/90 px-4 py-2.5 text-sm font-medium shadow-lg backdrop-blur hover:bg-accent"
              onClick={() => setSidebarVisible(true)}
              aria-label="Show settings"
              title="Show the settings panel"
            >
              <SlidersHorizontal className="size-4" />
              Settings
            </button>
          )}
        </>
      ) : (
        <>
          <button
            type="button"
            className="fixed bottom-4 right-4 z-30 inline-flex items-center gap-2 rounded-full border bg-background/90 px-4 py-2.5 text-sm font-medium shadow-lg backdrop-blur hover:bg-accent"
            onClick={() => setDrawerOpen(true)}
            aria-label="Open settings"
          >
            <SlidersHorizontal className="size-4" />
            Settings
          </button>
          <div
            className={cn('fixed inset-0 z-40 transition-opacity duration-300', drawerOpen ? 'opacity-100' : 'pointer-events-none opacity-0')}
            role="dialog"
            aria-modal="true"
            aria-label="Settings"
            aria-hidden={!drawerOpen}
          >
            <div className="absolute inset-0 bg-black/40" onClick={() => setDrawerOpen(false)} />
            {/* the same card, filling the screen, sliding in from the right */}
            <div
              className={cn(
                'absolute inset-3 flex flex-col overflow-hidden rounded-xl border bg-card shadow-2xl transition-transform duration-300 ease-out',
                drawerOpen ? 'translate-x-0' : 'translate-x-[110%]',
              )}
            >
              <div className="min-h-0 flex-1">{inspector}</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
