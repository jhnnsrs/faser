'use client';

import { useEffect, useRef, useState } from 'react';
import { Aperture, ChevronDown, CircleDot, Grid3x3, Layers, Lightbulb, Microscope, MousePointerClick, Sparkles, Sun, Target, Upload, X, type LucideIcon } from 'lucide-react';
import { cn } from '@/lib/cn';
import { alpha, autoGrid, type Grid } from './physics';
import { zernikeRange } from './zernike';
import {
  COMPONENTS,
  COMPONENT_BY_ID,
  GRID_KEYS,
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
      <label className="flex items-center justify-between gap-3 rounded-lg border bg-background px-3 py-2 text-sm">
        <span className="font-medium">SLM in the beam path</span>
        <input type="checkbox" className="size-4 accent-primary" checked={design.enabled} onChange={(e) => onDesign({ enabled: e.target.checked })} />
      </label>

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
            <button type="button" className="underline" onClick={() => onChange({ Mode: 'LOADED' })}>
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

export function Inspector(props: Props) {
  const { params, effective, derived, selected, onSelect, onChange, design, onDesign } = props;
  const def = selected ? COMPONENT_BY_ID[selected] : null;

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* component chips */}
      <div className="flex flex-wrap gap-1 border-b p-2">
        {COMPONENTS.map((c) => {
          const Icon = COMPONENT_ICONS[c.id];
          const on = selected === c.id;
          return (
            <button
              key={c.id}
              type="button"
              onClick={() => onSelect(on ? null : c.id)}
              className={cn(
                'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] transition-colors',
                on ? 'border-primary bg-primary text-primary-foreground' : 'text-muted-foreground hover:border-primary/50 hover:text-foreground',
              )}
              title={c.summary}
            >
              <Icon className="size-3" />
              {c.title}
            </button>
          );
        })}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {!def ? (
          <div className="flex flex-col gap-4">
            <div className="flex items-start gap-3 rounded-lg border border-dashed p-3 text-sm text-muted-foreground">
              <MousePointerClick className="mt-0.5 size-4 shrink-0" />
              <p>Click a part of the microscope (or a chip above) to unfold its parameters. Drag to orbit, scroll to zoom.</p>
            </div>
            <div>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Derived</h3>
              <DerivedStats params={effective} derived={derived} />
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="text-base font-semibold">{def.title}</h2>
                <p className="text-xs text-muted-foreground">{def.summary}</p>
              </div>
              <button type="button" className="rounded-md p-1 text-muted-foreground hover:bg-accent" onClick={() => onSelect(null)} aria-label="Close">
                <X className="size-4" />
              </button>
            </div>

            {def.id === 'slm' ? (
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
        )}
      </div>
    </div>
  );
}
