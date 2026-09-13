'use client';

import { useEffect, useRef, useState } from 'react';
import { Bookmark, ChevronDown, Loader2, Save, Trash2 } from 'lucide-react';
import { cn } from '@/lib/cn';
import { PRESETS } from './params';
import { deleteScene, listScenes, saveScene, type SavedScene, type SceneState } from './scenes';

interface Props {
  /** Name of the scene currently loaded (a preset or a saved one). */
  currentName: string;
  /** The setup differs from the loaded scene: offer to save it. */
  dirty: boolean;
  /** A volume for this scene is being computed (or the simulator is loading). */
  busy?: boolean;
  onPreset: (index: number) => void;
  onSaved: (scene: SavedScene) => void;
  /** The state to store when the user saves. */
  getState: () => SceneState;
  /** 'popover': a dropdown for the action bar; 'inline': unfolds in place (the settings card). */
  variant?: 'popover' | 'inline';
}

/** A menu of scenes: the built-in presets with their descriptions, and the ones saved in this browser. */
export function ScenePicker({ currentName, dirty, busy = false, onPreset, onSaved, getState, variant = 'popover' }: Props) {
  const inline = variant === 'inline';
  const [open, setOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [name, setName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<SavedScene[]>([]);
  const root = useRef<HTMLDivElement>(null);

  const toggle = (next = !open) => {
    if (next) setSaved(listScenes());
    setOpen(next);
  };
  useEffect(() => {
    if (!open || inline) return;
    const close = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener('mousedown', close);
    return () => window.removeEventListener('mousedown', close);
  }, [open, inline]);

  const save = () => {
    const r = saveScene(name, getState());
    if (r.error) {
      setError(r.error);
      return;
    }
    setError(null);
    setSaving(false);
    setName('');
    setSaved(listScenes());
    if (r.scene) onSaved(r.scene);
  };

  return (
    <div ref={root} className={cn('relative', inline && 'w-full')}>
      <div className={cn('flex text-sm', inline ? 'w-full items-center gap-2' : 'inline-flex overflow-hidden rounded-md border')}>
        <button
          type="button"
          className={cn('inline-flex min-w-0 items-center gap-2 hover:bg-accent', inline ? 'flex-1 rounded-md border px-2.5 py-1.5' : 'px-3 py-1.5')}
          onClick={() => toggle()}
          aria-expanded={open}
        >
          {busy ? <Loader2 className="size-4 shrink-0 animate-spin text-primary" aria-label="computing" /> : <Bookmark className="size-4 shrink-0 text-muted-foreground" />}
          <span className={cn('truncate', inline ? 'flex-1 text-left' : 'max-w-[16rem]')}>
            {currentName}
            {dirty && <span className="text-muted-foreground"> · edited</span>}
          </span>
          <ChevronDown className={cn('size-4 shrink-0 text-muted-foreground transition-transform', open && 'rotate-180')} />
        </button>
        {dirty && (
          <button
            type="button"
            className={cn('inline-flex items-center gap-1.5 hover:bg-accent', inline ? 'rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1.5 text-primary' : 'border-l px-3 py-1.5')}
            onClick={() => {
              setSaving((v) => !v);
              toggle(true);
            }}
            title="Save the current setup under a name in this browser"
          >
            <Save className="size-4" />
            Save
          </button>
        )}
      </div>
      {open && (
        <div
          className={cn(
            'text-sm',
            inline ? 'mt-2 rounded-lg border bg-background p-2' : 'absolute bottom-full left-0 z-20 mb-2 w-[min(38rem,90vw)] rounded-xl border bg-popover p-3 shadow-lg',
          )}
        >
          {saving && (
            <form
              className="mb-3 flex flex-wrap items-center gap-2 rounded-lg border bg-background p-2"
              onSubmit={(e) => {
                e.preventDefault();
                save();
              }}
            >
              <input
                autoFocus
                className="min-w-0 flex-1 rounded-md border bg-background px-2 py-1 text-sm outline-none focus:ring-2 focus:ring-ring"
                placeholder="Name this scene…"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
              <button type="submit" className="rounded-md bg-primary px-3 py-1 text-sm font-medium text-primary-foreground">
                Save
              </button>
              {error && <span className="basis-full text-xs text-red-600 dark:text-red-400">{error}</span>}
            </form>
          )}
          <div className={cn('grid gap-3', !inline && 'sm:grid-cols-2')}>
            <section>
              <h3 className="mb-1.5 px-1 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">Presets</h3>
              <ul className={cn('overflow-y-auto', inline ? 'max-h-[40vh]' : 'max-h-[50vh]')}>
                {PRESETS.map((p, i) => (
                  <li key={p.name}>
                    <button
                      type="button"
                      className={cn('w-full rounded-md px-2 py-1.5 text-left hover:bg-accent', currentName === p.name && 'bg-accent/70')}
                      onClick={() => {
                        onPreset(i);
                        setOpen(false);
                      }}
                    >
                      <div className="text-sm font-medium">{p.name}</div>
                      <div className="text-xs text-muted-foreground">{p.description}</div>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
            <section>
              <h3 className="mb-1.5 px-1 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">Saved in this browser</h3>
              {saved.length === 0 ? (
                <p className="px-2 py-1.5 text-xs text-muted-foreground">Nothing saved yet. Set up the microscope and press Save.</p>
              ) : (
                <ul className="max-h-[50vh] overflow-y-auto">
                  {saved.map((s) => (
                    <li key={s.id} className="flex items-start gap-1">
                      <button
                        type="button"
                        className={cn('min-w-0 flex-1 rounded-md px-2 py-1.5 text-left hover:bg-accent', currentName === s.name && 'bg-accent/70')}
                        onClick={() => {
                          onSaved(s);
                          setOpen(false);
                        }}
                      >
                        <div className="truncate text-sm font-medium">{s.name}</div>
                        <div className="text-xs text-muted-foreground">
                          {new Date(s.savedAt).toLocaleString()} · NA {s.params.NA}, λ {(s.params.Wavelength * 1000).toFixed(0)} nm
                          {s.params.SLM ? ', SLM' : ''}
                          {s.imaging.mode === 'sample' ? ', sample' : ''}
                        </div>
                      </button>
                      <button
                        type="button"
                        className="mt-1 rounded-md p-1 text-muted-foreground hover:bg-accent hover:text-red-600"
                        title="Delete"
                        onClick={() => {
                          deleteScene(s.id);
                          setSaved(listScenes());
                        }}
                      >
                        <Trash2 className="size-3.5" />
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </div>
        </div>
      )}
    </div>
  );
}
