'use client';

import { useEffect, useRef, useState } from 'react';
import { Aperture, ChevronDown, CircleDot, Dices, Grid3x3, Layers, Lightbulb, Microscope, Sigma, Sparkles, Sun, Target, Upload, X, type LucideIcon } from 'lucide-react';
import { cn } from '@/lib/cn';
import { alpha, autoGrid, type Grid } from './physics';
import { zernikeRange } from './zernike';
import {
  COMPONENTS,
  COMPONENT_BY_ID,
  DEFAULTS,
  GRID_KEYS,
  type ComponentDef,
  ZERNIKE_FIELDS,
  ZERNIKE_KEYS,
  ZERO_ZERNIKE,
  isSelect,
  type ComponentId,
  type Derived,
  type Field,
  type NumberField,
  type Params,
} from './params';
import { drawSlmPanel, hasZernike, imageToGray, resampleSquare, SLM_LEVELS, SLM_PATTERNS, SLM_SIZES, designPhases, type SlmDesign } from './slm';
import { ZERNIKE_KEYS as ZKEYS } from './params';
import { KIND_DEFAULTS, SAMPLE_OPTIONS, type ImagingSettings } from './use-imaging';
import type { SampleKind } from './volume';

export const COMPONENT_ICONS: Record<ComponentId, LucideIcon> = {
  laser: Sun,
  polarization: Sparkles,
  phaseplate: Aperture,
  slm: Grid3x3,
  objective: Microscope,
  pupil: Lightbulb,
  coverslip: Layers,
  sample: Layers,
  window: CircleDot,
  focus: Target,
};

interface Props {
  params: Params;
  /** The parameters actually simulated (grid resolved). */
  effective: Params;
  derived: Derived | null;
  selected: ComponentId | null;
  onSelect: (id: ComponentId | null) => void;
  onChange: (patch: Partial<Params>) => void;
  design: SlmDesign;
  onDesign: (patch: Partial<SlmDesign>) => void;
  /** Transfer the system aberrations to the SLM's Zernike layer (see playground.tsx). */
  onSendAberrations: (mode: 'move' | 'correct') => void;
  autoGridOn: boolean;
  onAutoGrid: (on: boolean) => void;
  estimateMs: number;
  /** What sits at the focus and how the sample is imaged. */
  imaging: ImagingSettings;
  onImaging: (patch: Partial<ImagingSettings>) => void;
  /** Status of the imaging simulation: voxel size, load guard and errors. */
  imagingStatus: { voxel: { dx: number; dz: number } | null; tooLarge: boolean; error: string | null; busy: 'sample' | 'image' | null };
  /** Shown as a close button next to the title (the drawer on small screens). */
  onClose?: () => void;
  /** The scene picker (presets, saved scenes, save), rendered under the title. */
  scenePicker?: React.ReactNode;
  /** Actions (compute, export, load, reset), rendered as the card's footer. */
  footer?: React.ReactNode;
}

/** Fields that only matter for some settings are dimmed otherwise. */
function relevant(field: Field, p: Params): boolean {
  switch (field.key) {
    case 'VC':
      return p.Mode === 'DONUT' || p.Mode === 'DONUT BOTTLE';
    case 'RC':
    case 'Ring_Radius':
      return p.Mode === 'BOTTLE' || p.Mode === 'DONUT BOTTLE';
    case 'p':
      return p.Mode === 'DONUT BOTTLE';
    case 'Mask_offset_x':
    case 'Mask_offset_y':
      return p.Mode !== 'GAUSSIAN' && p.Mode !== 'LOADED';
    case 'Wind_Radius':
    case 'Wind_Depth':
      return p.Window === 'CUSTOM';
    case 'Psi':
    case 'Epsilon':
      return p.Polarization === 1;
    default:
      return true;
  }
}

function NumberInput({
  field,
  value,
  onChange,
  disabled,
}: {
  field: NumberField;
  value: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  const [text, setText] = useState<string | null>(null);
  const shown = text ?? String(value);
  const commit = () => {
    if (text == null) return;
    const n = Number(text);
    if (Number.isFinite(n)) onChange(field.integer ? Math.round(n) : n);
    setText(null);
  };
  return (
    <div className="flex items-center gap-2">
      <input
        type="range"
        className="pg-range h-1.5 min-w-0 flex-1"
        min={field.min}
        max={field.max}
        step={field.step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(field.integer ? Math.round(Number(e.target.value)) : Number(e.target.value))}
        aria-label={field.label}
      />
      <input
        type="text"
        inputMode="decimal"
        disabled={disabled}
        className="w-[4.6rem] rounded-md border bg-background px-1.5 py-0.5 text-right text-xs tabular-nums outline-none focus:ring-2 focus:ring-ring disabled:opacity-60"
        value={shown}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
          if (e.key === 'Escape') setText(null);
        }}
        aria-label={`${field.label} value`}
      />
    </div>
  );
}

function FieldRow({
  field,
  params,
  value,
  onChange,
  disabled,
}: {
  field: Field;
  params: Params;
  value: Params[keyof Params];
  onChange: Props['onChange'];
  disabled?: boolean;
}) {
  const dim = !relevant(field, params);
  return (
    <label className={cn('block', dim && 'opacity-45')} title={field.hint}>
      <div className="mb-1 flex items-baseline justify-between gap-2 text-xs">
        <span className="font-medium text-foreground">{field.label}</span>
        {!isSelect(field) && field.unit && <span className="text-muted-foreground">{field.unit}</span>}
      </div>
      {isSelect(field) ? (
        <select
          className="w-full rounded-md border bg-background px-2 py-1 text-xs outline-none focus:ring-2 focus:ring-ring"
          value={String(value)}
          disabled={disabled}
          onChange={(e) => {
            const raw = e.target.value;
            const opt = field.options.find((o) => String(o.value) === raw);
            onChange({ [field.key]: opt ? opt.value : raw } as Partial<Params>);
          }}
        >
          {field.options.map((o) => (
            <option key={String(o.value)} value={String(o.value)}>
              {o.label}
            </option>
          ))}
        </select>
      ) : (
        <NumberInput field={field} value={value as number} disabled={disabled} onChange={(v) => onChange({ [field.key]: v } as Partial<Params>)} />
      )}
    </label>
  );
}

function fmt(v: number, digits = 3) {
  return Number.isFinite(v) ? v.toFixed(digits) : '–';
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-medium tabular-nums">{value}</dd>
    </div>
  );
}

export function DerivedStats({ params, derived }: { params: Params; derived: Derived | null }) {
  const d = derived;
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-3">
      <Stat label="α (immersion)" value={d ? `${fmt((d.alpha * 180) / Math.PI, 1)}°` : '–'} />
      <Stat label="α (sample)" value={d ? `${fmt((d.alpha3_eff * 180) / Math.PI, 1)}°` : '–'} />
      <Stat label="effective NA" value={d ? fmt(d.na_eff, 3) : '–'} />
      <Stat label="pupil radius r₀" value={d ? `${fmt(d.r0, 0)} µm` : '–'} />
      <Stat label="focus shift Δz" value={d ? `${d.dfoc > 0 ? '+' : ''}${fmt(d.dfoc, 3)} µm` : '–'} />
      <Stat label="λ / (2 NA)" value={`${fmt(params.Wavelength / (2 * params.NA), 3)} µm`} />
      <Stat label="voxel xy" value={`${fmt(params.Nxy > 1 ? (2 * params.L_obs_XY) / (params.Nxy - 1) : 0, 4)} µm`} />
      <Stat label="voxel z" value={`${fmt(params.Nz > 1 ? (2 * params.L_obs_Z) / (params.Nz - 1) : 0, 4)} µm`} />
      <Stat label="grid" value={`${params.Nxy}² × ${params.Nz}, θ ${params.Ntheta} φ ${params.Nphi}`} />
    </dl>
  );
}

/** The "in the beam path / in the setup" row of an optional element. */
function PresenceToggle({ label, checked, onChange, hint }: { label: string; checked: boolean; onChange: (on: boolean) => void; hint?: string }) {
  return (
    <label className="flex items-center justify-between gap-3 rounded-lg border bg-background px-3 py-2 text-sm" title={hint}>
      <span className="font-medium">{label}</span>
      <input type="checkbox" className="size-4 shrink-0 accent-primary" checked={checked} onChange={(e) => onChange(e.target.checked)} />
    </label>
  );
}

// ---------------------------------------------------------------------------
// SLM designer
// ---------------------------------------------------------------------------

function Num({
  label,
  value,
  min,
  max,
  step,
  unit,
  hint,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  hint?: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block" title={hint}>
      <div className="mb-1 flex items-baseline justify-between gap-2 text-xs">
        <span className="font-medium">{label}</span>
        {unit && <span className="text-muted-foreground">{unit}</span>}
      </div>
      <div className="flex items-center gap-2">
        <input type="range" className="pg-range h-1.5 min-w-0 flex-1" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
        <span className="w-[4.6rem] text-right text-xs tabular-nums">{Number.isInteger(step) ? value : value.toFixed(step < 0.01 ? 3 : 2)}</span>
      </div>
    </label>
  );
}

function SlmDesigner({ params, design, onDesign, onChange }: { params: Params; design: SlmDesign; onDesign: Props['onDesign']; onChange: Props['onChange'] }) {
  const preview = useRef<HTMLCanvasElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const info = SLM_PATTERNS.find((x) => x.value === design.pattern);
  const sinA = Math.sin(alpha(params));

  useEffect(() => {
    if (!preview.current) return;
    const s = params.SLM ?? {
      n: design.n,
      phase: [],
      extent: design.extent,
      levels: design.levels,
      fill_factor: design.fillFactor,
      offset_x: design.offsetX,
      offset_y: design.offsetY,
    };
    const phase = params.SLM ? params.SLM.phase : designPhases(design);
    drawSlmPanel(preview.current, s, phase, 192);
  }, [params.SLM, design]);

  const loadImage = async (file: File) => {
    try {
      const gray = await imageToGray(file, design.n);
      onDesign({ pattern: 'image', image: gray, imageName: file.name, enabled: true });
      setImageError(null);
    } catch (e) {
      setImageError(e instanceof Error ? e.message : String(e));
    }
  };

  const setSize = (n: number) => {
    const patch: Partial<SlmDesign> = { n };
    if (design.image) patch.image = resampleSquare(design.image, design.n, n);
    if (design.custom) patch.custom = resampleSquare(design.custom, design.n, n);
    onDesign(patch);
  };

  const plateActive = params.Mode !== 'GAUSSIAN' && params.Mode !== 'LOADED';

  return (
    <div className="flex flex-col gap-4">
      <PresenceToggle label="SLM in the beam path" checked={design.enabled} onChange={(on) => onDesign({ enabled: on })} />

      <div className={cn('flex flex-col gap-4', !design.enabled && 'pointer-events-none opacity-45')}>
        <div className="flex gap-3">
          <canvas ref={preview} className="size-[7.5rem] shrink-0 rounded-md border bg-black" style={{ imageRendering: design.n <= 64 ? 'pixelated' : 'auto' }} />
          <div className="flex min-w-0 flex-1 flex-col gap-2">
            <label className="block">
              <div className="mb-1 text-xs font-medium">Pattern</div>
              <select
                className="w-full rounded-md border bg-background px-2 py-1 text-xs"
                value={design.pattern}
                onChange={(e) => onDesign({ pattern: e.target.value as SlmDesign['pattern'] })}
              >
                {SLM_PATTERNS.filter((x) => x.value !== 'custom' || design.custom).map((x) => (
                  <option key={x.value} value={x.value}>
                    {x.label}
                  </option>
                ))}
              </select>
            </label>
            <p className="text-xs text-muted-foreground">{info?.description}</p>
          </div>
        </div>

        {design.pattern === 'vortex' && <Num label="Topological charge" value={design.charge} min={-6} max={6} step={1} onChange={(v) => onDesign({ charge: v })} />}
        {design.pattern === 'bottle' && (
          <>
            <Num label="Disc radius" value={design.ringRadius} min={0.05} max={0.99} step={0.005} unit="r₀" onChange={(v) => onDesign({ ringRadius: v })} />
            <Num label="Phase step" value={design.ringStep} min={-2} max={2} step={0.05} unit="π" onChange={(v) => onDesign({ ringStep: v })} />
          </>
        )}
        {(design.pattern === 'halfmoon' || design.pattern === 'grating') && (
          <Num label="Orientation" value={design.angle} min={0} max={180} step={1} unit="°" onChange={(v) => onDesign({ angle: v })} />
        )}
        {design.pattern === 'grating' && (
          <Num
            label="Period"
            value={design.period}
            min={0.2}
            max={4}
            step={0.05}
            unit="r₀"
            hint={`Moves the focus by λ / (period · sin α) ≈ ${(params.Wavelength / (design.period * sinA)).toFixed(2)} µm`}
            onChange={(v) => onDesign({ period: v })}
          />
        )}
        {design.pattern === 'axicon' && (
          <Num
            label="Cone"
            value={design.cone}
            min={0}
            max={6}
            step={0.1}
            unit="waves at r₀"
            hint={`Ring radius λ · waves / sin α ≈ ${((params.Wavelength * design.cone) / sinA).toFixed(2)} µm`}
            onChange={(v) => onDesign({ cone: v })}
          />
        )}
        {design.pattern === 'image' && (
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 text-xs">
              <button type="button" className="inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1" onClick={() => fileRef.current?.click()}>
                <Upload className="size-3.5" />
                Load image
              </button>
              <span className="truncate text-muted-foreground">{design.imageName ?? 'PNG / JPG, gray = phase'}</span>
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void loadImage(f);
                  e.target.value = '';
                }}
              />
            </div>
            {imageError && <p className="text-xs text-red-600 dark:text-red-400">{imageError}</p>}
            <Num label="White level" value={design.imageWaves} min={0.1} max={4} step={0.05} unit="waves" onChange={(v) => onDesign({ imageWaves: v })} />
          </div>
        )}
        {design.pattern === 'custom' && <p className="text-xs text-muted-foreground">A {design.n} × {design.n} pattern loaded from a config file.</p>}

        <div className="grid grid-cols-2 gap-x-3 gap-y-3">
          <label className="block">
            <div className="mb-1 text-xs font-medium">Resolution</div>
            <select className="w-full rounded-md border bg-background px-2 py-1 text-xs" value={design.n} onChange={(e) => setSize(Number(e.target.value))}>
              {SLM_SIZES.map((n) => (
                <option key={n} value={n}>
                  {n} × {n} px
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            <div className="mb-1 text-xs font-medium">Phase levels</div>
            <select className="w-full rounded-md border bg-background px-2 py-1 text-xs" value={design.levels} onChange={(e) => onDesign({ levels: Number(e.target.value) })}>
              {SLM_LEVELS.map((l) => (
                <option key={l.value} value={l.value}>
                  {l.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <Num label="Fill factor" value={design.fillFactor} min={0.5} max={1} step={0.01} hint="Fraction of each pixel that modulates; the dead space passes unmodulated light into the focus" onChange={(v) => onDesign({ fillFactor: v })} />
        <Num label="Panel half-width" value={design.extent} min={0.5} max={2} step={0.01} unit="r₀" hint="1: the pupil just fits the panel; > 1: the beam uses the central part" onChange={(v) => onDesign({ extent: v })} />
        <div className="grid grid-cols-2 gap-x-3">
          <Num label="Offset x" value={design.offsetX} min={-0.5} max={0.5} step={0.01} unit="r₀" onChange={(v) => onDesign({ offsetX: v })} />
          <Num label="Offset y" value={design.offsetY} min={-0.5} max={0.5} step={0.01} unit="r₀" onChange={(v) => onDesign({ offsetY: v })} />
        </div>
        <ZernikeLayer design={design} onDesign={onDesign} />
        {design.enabled && plateActive && (
          <p className="rounded-md border border-amber-500/40 bg-amber-500/10 px-2.5 py-2 text-xs text-amber-700 dark:text-amber-300">
            The phase plate is also active ({params.Mode.toLowerCase()}); both multiply the pupil.{' '}
            <button type="button" className="underline" onClick={() => onChange({ Mode: 'GAUSSIAN' })}>
              Use only the SLM
            </button>
          </p>
        )}
      </div>
    </div>
  );
}

/** The Zernike modes displayed on the SLM on top of its pattern (adaptive optics). */
function ZernikeLayer({ design, onDesign }: { design: SlmDesign; onDesign: Props['onDesign'] }) {
  const active = hasZernike(design.zernike);
  const [open, setOpen] = useState(false);
  const shown = open || active;
  return (
    <section className="rounded-lg border bg-background">
      <button type="button" className="flex w-full items-center justify-between px-3 py-2 text-left text-xs font-semibold" onClick={() => setOpen((o) => !o)} aria-expanded={shown}>
        <span>
          Zernike layer{active ? ` (${ZERNIKE_KEYS.filter((k) => design.zernike[k] !== 0).length} modes)` : ''}
        </span>
        <ChevronDown className={cn('size-4 text-muted-foreground transition-transform', shown && 'rotate-180')} />
      </button>
      {shown && (
        <div className="flex flex-col gap-3 border-t px-3 py-3">
          <p className="text-xs text-muted-foreground">
            Added to the pattern, in radians on the unit pupil like the system aberrations. On an ideal panel this equals the system
            aberrations; the panel&apos;s levels and fill factor act on it.
          </p>
          {ZERNIKE_FIELDS.map((f) => (
            <Num key={f.key} label={f.label} value={design.zernike[f.key]} min={f.min} max={f.max} step={f.step} onChange={(v) => onDesign({ zernike: { ...design.zernike, [f.key]: v } })} />
          ))}
          <button type="button" className="self-start text-xs text-primary underline disabled:opacity-50" disabled={!active} onClick={() => onDesign({ zernike: { ...ZERO_ZERNIKE } })}>
            Clear the layer
          </button>
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Sample: what sits at the focus
// ---------------------------------------------------------------------------

function ImagingSection({ imaging, onImaging, imagingStatus }: Pick<Props, 'imaging' | 'onImaging' | 'imagingStatus'>) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const s = imaging;
  const info = SAMPLE_OPTIONS.find((k) => k.value === s.kind);
  const { voxel, tooLarge, error, busy } = imagingStatus;
  const loadImage = async (file: File) => {
    try {
      const gray = await imageToGray(file, s.imageSize);
      onImaging({ image: gray, imageName: file.name, kind: 'image', mode: 'sample' });
      setImageError(null);
    } catch (e) {
      setImageError(e instanceof Error ? e.message : String(e));
    }
  };
  return (
    <div className="flex flex-col gap-3">
      <label className="block">
        <div className="mb-1 text-xs font-medium">At the focus</div>
        <select className="w-full rounded-md border bg-background px-2 py-1 text-xs" value={s.mode} onChange={(e) => onImaging({ mode: e.target.value as ImagingSettings['mode'] })}>
          <option value="bead">A fluorescent bead (its image is the PSF)</option>
          <option value="sample">A sample, imaged through the PSF</option>
        </select>
      </label>
      {s.mode === 'sample' && (
        <div className="flex flex-col gap-3 rounded-lg border bg-background p-3">
          <label className="block">
            <div className="mb-1 text-xs font-medium">Sample</div>
            <div className="flex gap-2">
              <select
                className="min-w-0 flex-1 rounded-md border bg-background px-2 py-1 text-xs"
                value={s.kind}
                onChange={(e) => {
                  const kind = e.target.value as ImagingSettings['kind'];
                  onImaging(kind === 'image' ? { kind } : { kind, ...KIND_DEFAULTS[kind as SampleKind] });
                }}
              >
                {SAMPLE_OPTIONS.map((k) => (
                  <option key={k.value} value={k.value}>
                    {k.label}
                  </option>
                ))}
              </select>
              {s.kind !== 'image' && (
                <button type="button" className="inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs" onClick={() => onImaging({ seed: Math.floor(Math.random() * 1e6) })} title="New random sample">
                  <Dices className="size-3.5" />
                  {s.seed}
                </button>
              )}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">{info?.description}</p>
          </label>
          {s.kind === 'image' && (
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2 text-xs">
                <button type="button" className="inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1" onClick={() => fileRef.current?.click()}>
                  <Upload className="size-3.5" />
                  Load image
                </button>
                <span className="truncate text-muted-foreground">{s.imageName ?? 'PNG / JPG, brightness = fluorescence'}</span>
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) void loadImage(f);
                    e.target.value = '';
                  }}
                />
              </div>
              {imageError && <p className="text-xs text-red-600 dark:text-red-400">{imageError}</p>}
              <Num label="Slab thickness" value={s.imageThickness} min={0.05} max={3} step={0.05} unit="µm" onChange={(v) => onImaging({ imageThickness: v })} />
            </div>
          )}
          {(s.kind === 'beads' || s.kind === 'filaments' || s.kind === 'cells' || s.kind === 'spokes') && (
            <Num
              label={s.kind === 'spokes' ? 'Spokes' : 'Count'}
              value={s.count}
              min={s.kind === 'spokes' ? 4 : 1}
              max={s.kind === 'cells' ? 8 : s.kind === 'spokes' ? 64 : 120}
              step={1}
              onChange={(v) => onImaging({ count: v })}
            />
          )}
          {s.kind !== 'image' && (
            <Num
              label={s.kind === 'cells' ? 'Membrane' : s.kind === 'spokes' ? 'Inner radius' : 'Radius'}
              value={s.radius}
              min={0.02}
              max={1}
              step={0.01}
              unit="µm"
              onChange={(v) => onImaging({ radius: v })}
            />
          )}
          {(s.kind === 'lattice' || s.kind === 'spokes') && (
            <Num label={s.kind === 'lattice' ? 'Spacing' : 'Slab thickness'} value={s.spacing} min={0.05} max={3} step={0.05} unit="µm" onChange={(v) => onImaging({ spacing: v })} />
          )}
          <div className="grid grid-cols-2 gap-x-3">
            <Num label="Field xy" value={s.nxy} min={32} max={256} step={8} unit="px" onChange={(v) => onImaging({ nxy: v })} />
            <Num label="Field z" value={s.nz} min={8} max={128} step={4} unit="px" onChange={(v) => onImaging({ nz: v })} />
          </div>
          <Num label="Photons (peak)" value={s.photons} min={0} max={2000} step={10} hint="Poisson shot noise with this many expected counts in the brightest voxel; 0 = none" onChange={(v) => onImaging({ photons: v })} />
          <p className="text-xs text-muted-foreground">
            Field of view {voxel ? `${(s.nxy * voxel.dx).toFixed(2)} × ${(s.nxy * voxel.dx).toFixed(2)} × ${(s.nz * voxel.dz).toFixed(2)} µm` : '–'} on the PSF&apos;s voxel grid; the image is the sample
            convolved with the PSF (3-D FFT).
            {busy === 'sample' ? ' Generating the sample…' : busy === 'image' ? ' Convolving…' : ''}
            {tooLarge && <span className="ml-1 text-amber-600 dark:text-amber-400">Too large for the FFT in the browser, reduce the field or the PSF grid.</span>}
            {error && <span className="ml-1 text-red-600 dark:text-red-400">{error}</span>}
          </p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Focus & sampling
// ---------------------------------------------------------------------------

function GridSection({
  params,
  effective,
  design,
  autoGridOn,
  onAutoGrid,
  onChange,
  estimateMs,
}: Pick<Props, 'params' | 'effective' | 'design' | 'autoGridOn' | 'onAutoGrid' | 'onChange' | 'estimateMs'>) {
  const def = COMPONENT_BY_ID.focus;
  const gridFields = def.fields.filter((f) => (GRID_KEYS as readonly string[]).includes(f.key));
  const auto: Grid = autoGrid(params, design.enabled ? zernikeRange(design.zernike) : 0);
  return (
    <div className="flex flex-col gap-3">
      <label className="flex items-center justify-between gap-3 rounded-lg border bg-background px-3 py-2 text-sm">
        <span>
          <span className="font-medium">Automatic grid</span>
          <span className="block text-xs text-muted-foreground">Sized from the optics so the PSF is resolved and the integral converges</span>
        </span>
        <input type="checkbox" className="size-4 shrink-0 accent-primary" checked={autoGridOn} onChange={(e) => onAutoGrid(e.target.checked)} />
      </label>
      <div className="grid grid-cols-4 gap-2 text-xs">
        {(['Nxy', 'Nz', 'Ntheta', 'Nphi'] as const).map((k) => (
          <div key={k} className="rounded-md border bg-background px-2 py-1.5 text-center">
            <div className="text-muted-foreground">{k === 'Ntheta' ? 'Nθ' : k === 'Nphi' ? 'Nφ' : k}</div>
            <div className="font-medium tabular-nums">{effective[k]}</div>
            {!autoGridOn && auto[k] !== effective[k] && <div className="text-[10px] text-muted-foreground">auto {auto[k]}</div>}
          </div>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        ≈ {estimateMs < 1000 ? `${estimateMs.toFixed(0)} ms` : `${(estimateMs / 1000).toFixed(1)} s`} per volume on this machine
      </p>
      {!autoGridOn && (
        <div className="flex flex-col gap-3">
          {gridFields.map((f) => (
            <FieldRow key={f.key} field={f} params={params} value={params[f.key]} onChange={onChange} />
          ))}
          <button type="button" className="self-start text-xs text-primary underline" onClick={() => onChange(auto)}>
            Copy the automatic values
          </button>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inspector
// ---------------------------------------------------------------------------

/** The parameters of one component (the content of its accordion section). */
/** Whether an optional element is part of the setup. */
function isPresent(id: ComponentId, p: Params): boolean {
  switch (id) {
    case 'phaseplate':
      return p.Mode !== 'GAUSSIAN' && p.Mode !== 'LOADED';
    case 'window':
      return p.Window === 'CUSTOM';
    case 'coverslip':
      return p.Thickness > 0;
    default:
      return true;
  }
}

function SectionBody({ def, ...props }: Props & { def: ComponentDef }) {
  const { params, derived, onChange, design, onDesign } = props;
  // remembered when an element is switched off, so switching it on restores its setting
  const lastMask = useRef<Params['Mode']>('DONUT');
  const lastThickness = useRef(DEFAULTS.Thickness);
  const optional = def.id === 'phaseplate' || def.id === 'window' || def.id === 'coverslip';
  const present = isPresent(def.id, params);
  const toggle =
    def.id === 'phaseplate' ? (
      <PresenceToggle
        label="Phase plate in the beam path"
        checked={present}
        hint="An analytic phase mask; without it the beam reaches the pupil flat (the SLM can still shape it)"
        onChange={(on) => {
          if (!on) lastMask.current = params.Mode;
          onChange({ Mode: on ? lastMask.current : 'GAUSSIAN' });
        }}
      />
    ) : def.id === 'window' ? (
      <PresenceToggle label="Cranial window in the setup" checked={present} onChange={(on) => onChange({ Window: on ? 'CUSTOM' : 'NO' })} />
    ) : def.id === 'coverslip' ? (
      <PresenceToggle
        label="Coverslip in the setup"
        checked={present}
        hint="Without a coverslip the immersion medium touches the sample (thickness 0)"
        onChange={(on) => {
          if (!on) lastThickness.current = params.Thickness;
          onChange({ Thickness: on ? lastThickness.current : 0 });
        }}
      />
    ) : null;
  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-muted-foreground">{def.summary}</p>
      {toggle}
      {optional && !present ? null : def.id === 'slm' ? (
        <SlmDesigner params={params} design={design} onDesign={onDesign} onChange={onChange} />
      ) : def.id === 'focus' ? (
        <>
          {def.fields
            .filter((f) => !(GRID_KEYS as readonly string[]).includes(f.key))
            .map((f) => (
              <FieldRow key={f.key} field={f} params={params} value={params[f.key]} onChange={onChange} />
            ))}
          <GridSection {...props} />
        </>
      ) : (
        def.fields.map((f) => <FieldRow key={f.key} field={f} params={params} value={params[f.key]} onChange={onChange} />)
      )}

      {def.id === 'pupil' && (
        <div className="flex flex-col gap-2 rounded-lg border bg-background p-3">
          <p className="text-xs text-muted-foreground">
            Send these modes to the spatial light modulator{design.enabled ? ' (on top of its pattern)' : ' (it comes on with a flat pattern)'}.
            The SLM sits in the same pupil plane, so on an ideal panel the result is identical; a real panel adds its pixelation and
            quantization. The aberration offsets stay here.
          </p>
          <div className="flex flex-wrap gap-2 text-xs">
            <button
              type="button"
              className="rounded-md border px-2.5 py-1 disabled:opacity-50"
              disabled={!ZERNIKE_KEYS.some((k) => params[k] !== 0)}
              title="Display these modes on the SLM as a hologram and clear the system aberrations"
              onClick={() => props.onSendAberrations('move')}
            >
              Display on the SLM
            </button>
            <button
              type="button"
              className="rounded-md border px-2.5 py-1 disabled:opacity-50"
              disabled={!ZERNIKE_KEYS.some((k) => params[k] !== 0)}
              title="Put the opposite phase on the SLM, leaving the system aberrations in place (adaptive-optics correction)"
              onClick={() => props.onSendAberrations('correct')}
            >
              Correct on the SLM
            </button>
          </div>
        </div>
      )}
      {def.id === 'objective' && derived && (
        <p className="text-xs text-muted-foreground">
          α = {((derived.alpha * 180) / Math.PI).toFixed(1)}° in the immersion, {((derived.alpha3_eff * 180) / Math.PI).toFixed(1)}° in the sample
          {params.n1 * Math.sin(derived.alpha_eff) > params.n3 ? ' (part of the aperture is beyond the critical angle: evanescent in the sample)' : ''}. Pupil radius r₀ = {derived.r0.toFixed(0)} µm.
        </p>
      )}
      {def.id === 'sample' && <ImagingSection imaging={props.imaging} onImaging={props.onImaging} imagingStatus={props.imagingStatus} />}
      {def.id === 'window' && derived && params.Window === 'CUSTOM' && (
        <p className="text-xs text-muted-foreground">
          The window clips the aperture to an effective NA of {derived.na_eff.toFixed(3)} (α = {((derived.alpha_eff * 180) / Math.PI).toFixed(1)}°).
        </p>
      )}
      {def.id === 'sample' && derived && Math.abs(derived.dfoc) > 1e-3 && (
        <p className="text-xs text-muted-foreground">
          The index mismatch shifts the focus by {derived.dfoc > 0 ? '+' : ''}
          {derived.dfoc.toFixed(2)} µm from its nominal position.
        </p>
      )}
      {def.id === 'phaseplate' && params.SLM && (params.Mode === 'GAUSSIAN' || params.Mode === 'LOADED') && (
        <p className="text-xs text-muted-foreground">The pattern is on the SLM; this plate is flat.</p>
      )}
    </div>
  );
}

/** Short names for the badges that show what differs from the standard PSF. */
const SHORT: Partial<Record<keyof Params, string>> = {
  NA: 'NA', n1: 'n₁', n2: 'n₂', n3: 'n₃', Wavelength: 'λ', Waist: 'waist', WD: 'f', Collar: 'collar',
  Thickness: 't', Tilt: 'tilt', Depth: 'depth', Polarization: '', Psi: 'ψ', Epsilon: 'ε', Mode: '',
  VC: 'charge', RC: 'step', Ring_Radius: 'r', p: 'mix', Mask_offset_x: 'mask x', Mask_offset_y: 'mask y',
  Ampl_offset_x: 'beam x', Ampl_offset_y: 'beam y', Aberration_offset_x: 'ab. x', Aberration_offset_y: 'ab. y',
  L_obs_XY: 'XY', L_obs_Z: 'Z', Normalize: 'norm', Theta_sampling: 'θ', Nxy: 'Nxy', Nz: 'Nz', Ntheta: 'Nθ', Nphi: 'Nφ',
  Window: '', Wind_Radius: 'r', Wind_Depth: 'd',
};

function fmtValue(field: Field, v: Params[keyof Params]): string {
  if (isSelect(field)) {
    const opt = field.options.find((o) => String(o.value) === String(v));
    const label = (opt?.label ?? String(v)).replace(/\s*\(.*\)$/, '');
    return label.length > 18 ? `${label.slice(0, 17)}…` : label.toLowerCase();
  }
  const n = Number(v);
  const text = Math.abs(n) >= 100 ? n.toFixed(0) : Number(n.toFixed(3)).toString();
  return field.unit ? `${text} ${field.unit}` : text;
}

/** What a section changes relative to the default (standard PSF): short "name value" badges. */
function sectionChanges(c: ComponentDef, props: Props): string[] {
  const { params, design, imaging, autoGridOn } = props;
  const out: string[] = [];
  if (c.id === 'slm') {
    if (design.enabled && params.SLM) {
      out.push(`${SLM_PATTERNS.find((x) => x.value === design.pattern)?.label.toLowerCase() ?? design.pattern}`);
      out.push(`${design.n}², ${design.levels > 1 ? `${design.levels} levels` : 'continuous'}`);
      if (design.fillFactor < 1) out.push(`fill ${design.fillFactor.toFixed(2)}`);
      const z = ZKEYS.filter((k) => design.zernike[k] !== 0).length;
      if (z) out.push(`${z} zernike mode${z > 1 ? 's' : ''}`);
    }
    return out;
  }
  if ((c.id === 'phaseplate' || c.id === 'window' || c.id === 'coverslip') && !isPresent(c.id, params)) {
    return c.id === 'coverslip' ? ['none'] : [];
  }
  for (const f of c.fields) {
    if (c.id === 'focus' && (GRID_KEYS as readonly string[]).includes(f.key)) continue;
    const v = params[f.key];
    if (v === DEFAULTS[f.key]) continue;
    if (c.id === 'window' && f.key !== 'Window' && params.Window !== 'CUSTOM') continue;
    const name = SHORT[f.key] ?? f.label.toLowerCase();
    out.push(name ? `${name} ${fmtValue(f, v)}` : fmtValue(f, v));
  }
  if (c.id === 'focus' && !autoGridOn) out.push(`manual grid ${params.Nxy}² × ${params.Nz}`);
  if (c.id === 'sample' && imaging.mode === 'sample') out.push(`imaging ${SAMPLE_OPTIONS.find((k) => k.value === imaging.kind)?.label.toLowerCase() ?? imaging.kind}`);
  return out;
}

function ChangeBadges({ items }: { items: string[] }) {
  if (items.length === 0) return null;
  const shown = items.slice(0, 2);
  const more = items.length - shown.length;
  return (
    <span className="flex min-w-0 flex-wrap items-center justify-end gap-1">
      {shown.map((t) => (
        <span key={t} className="max-w-[9rem] truncate rounded-full bg-primary/10 px-1.5 py-px text-[10px] font-medium leading-4 text-primary" title={t}>
          {t}
        </span>
      ))}
      {more > 0 && (
        <span className="rounded-full bg-muted px-1.5 py-px text-[10px] leading-4 text-muted-foreground" title={items.slice(2).join(', ')}>
          +{more} more
        </span>
      )}
    </span>
  );
}

export function Inspector(props: Props) {
  const { effective, derived, selected, onSelect, onClose, scenePicker, footer } = props;
  const [derivedOpen, setDerivedOpen] = useState(false);

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* title */}
      <div className="flex items-start justify-between gap-3 border-b px-4 pb-3 pt-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight">Playground</h1>
          <p className="mt-1 text-xs text-muted-foreground">
            The faser simulator as WebAssembly in your browser: a vectorial PSF for the microscope on the left, computed on your machine.
            Click a part of the microscope, or open a section below, to change it.
          </p>
        </div>
        {onClose && (
          <button type="button" className="rounded-md p-1 text-muted-foreground hover:bg-accent" onClick={onClose} aria-label="Hide settings" title="Hide the settings panel">
            <X className="size-4" />
          </button>
        )}
      </div>
      {scenePicker && <div className="border-b px-4 py-3">{scenePicker}</div>}

      {/* accordion of the components */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        {COMPONENTS.map((c) => {
          const Icon = COMPONENT_ICONS[c.id];
          const open = selected === c.id;
          return (
            <section key={c.id} className={cn('border-b', open && 'bg-background/60')}>
              <button
                type="button"
                className={cn('flex w-full items-center gap-2.5 px-4 py-2.5 text-left text-sm hover:bg-accent/60', open ? 'font-semibold text-foreground' : 'text-foreground/90')}
                onClick={() => onSelect(open ? null : c.id)}
                aria-expanded={open}
              >
                <Icon className={cn('size-4 shrink-0', open ? 'text-primary' : 'text-muted-foreground')} />
                <span className="shrink-0">{c.title}</span>
                <span className="flex min-w-0 flex-1 justify-end">
                  <ChangeBadges items={sectionChanges(c, props)} />
                </span>
                <ChevronDown className={cn('size-4 shrink-0 text-muted-foreground transition-transform', open && 'rotate-180')} />
              </button>
              {open && (
                <div className="px-4 pb-4 pt-1">
                  <SectionBody def={c} {...props} />
                </div>
              )}
            </section>
          );
        })}
        <section className={cn('border-b', derivedOpen && 'bg-background/60')}>
          <button
            type="button"
            className={cn('flex w-full items-center gap-2.5 px-4 py-2.5 text-left text-sm hover:bg-accent/60', derivedOpen ? 'font-semibold' : 'text-foreground/90')}
            onClick={() => setDerivedOpen((o) => !o)}
            aria-expanded={derivedOpen}
          >
            <Sigma className={cn('size-4 shrink-0', derivedOpen ? 'text-primary' : 'text-muted-foreground')} />
            <span className="flex-1">Derived quantities</span>
            <ChevronDown className={cn('size-4 shrink-0 text-muted-foreground transition-transform', derivedOpen && 'rotate-180')} />
          </button>
          {derivedOpen && (
            <div className="px-4 pb-4 pt-1">
              <DerivedStats params={effective} derived={derived} />
            </div>
          )}
        </section>
      </div>
      {footer && <div className="border-t bg-card px-3 py-2.5">{footer}</div>}
    </div>
  );
}
