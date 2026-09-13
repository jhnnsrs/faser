'use client';

import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/cn';
import { FIELD_GROUPS, isSelect, type Field, type NumberField, type Params } from './params';

interface Props {
  params: Params;
  onChange: (patch: Partial<Params>) => void;
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
      return p.Mode !== 'GAUSSIAN';
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

function NumberInput({ field, value, onChange }: { field: NumberField; value: number; onChange: (v: number) => void }) {
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
        onChange={(e) => onChange(field.integer ? Math.round(Number(e.target.value)) : Number(e.target.value))}
        aria-label={field.label}
      />
      <input
        type="text"
        inputMode="decimal"
        className="w-[4.6rem] rounded-md border bg-background px-1.5 py-0.5 text-right text-xs tabular-nums outline-none focus:ring-2 focus:ring-ring"
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

function FieldRow({ field, params, onChange }: { field: Field; params: Params; onChange: Props['onChange'] }) {
  const value = params[field.key];
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
        <NumberInput field={field} value={value as number} onChange={(v) => onChange({ [field.key]: v } as Partial<Params>)} />
      )}
    </label>
  );
}

export function ParamPanel({ params, onChange }: Props) {
  const [open, setOpen] = useState<Record<string, boolean>>({ 'Objective & media': true, 'Coverslip & sample': true, Beam: true });
  return (
    <div className="flex flex-col gap-2">
      {FIELD_GROUPS.map((group) => {
        const isOpen = open[group.title] ?? false;
        return (
          <section key={group.title} className="rounded-lg border bg-card">
            <button
              type="button"
              className="flex w-full items-center justify-between px-3 py-2 text-left text-sm font-semibold"
              onClick={() => setOpen((o) => ({ ...o, [group.title]: !isOpen }))}
              aria-expanded={isOpen}
            >
              {group.title}
              <ChevronDown className={cn('size-4 text-muted-foreground transition-transform', isOpen && 'rotate-180')} />
            </button>
            {isOpen && (
              <div className="flex flex-col gap-3 border-t px-3 py-3">
                {group.description && <p className="text-xs text-muted-foreground">{group.description}</p>}
                {group.fields.map((f) => (
                  <FieldRow key={f.key} field={f} params={params} onChange={onChange} />
                ))}
              </div>
            )}
          </section>
        );
      })}
    </div>
  );
}
