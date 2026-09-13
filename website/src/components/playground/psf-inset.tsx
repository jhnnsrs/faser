'use client';

import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/cn';
import { COLORMAPS, type ColormapName } from './colormaps';
import { SliceCanvas } from './slice-canvas';
import { SliceViews } from './slice-views';
import { colormapCss } from './colormaps';
import type { PsfResult } from './use-psf-worker';
import type { Volume } from './volume';
import { VolumeViewer, type RenderSettings } from './volume-viewer';

interface Props {
  result: PsfResult | null;
  volume: Volume | null;
  quality: 'preview' | 'final';
  render: RenderSettings;
  onRender: (patch: Partial<RenderSettings>) => void;
  /** The convolved image of the simulated sample (imaging simulation open); shown instead of the PSF. */
  image?: Volume | null;
}

/**
 * The zoom-in on the focus: the PSF volume and its XY / XZ cross-sections,
 * floating next to the microscope and tied to the focus by a leader line.
 */
export function PsfInset({ result, volume, quality, render, onRender, image }: Props) {
  const [showControls, setShowControls] = useState(false);
  const [view, setView] = useState<'psf' | 'image'>('image');
  const showImage = !!image && view === 'image';
  const shown = showImage ? image : volume;
  const imageRender: RenderSettings = { ...render, mode: 'composite', threshold: Math.max(render.threshold, 0.05), opacity: Math.max(render.opacity, 0.9) };
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-2 border-b px-3 py-2 text-xs">
        {image ? (
          <div className="inline-flex overflow-hidden rounded-md border">
            {(['image', 'psf'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setView(m)}
                className={cn('px-2 py-0.5', view === m ? 'bg-primary text-primary-foreground' : 'text-muted-foreground')}
              >
                {m === 'image' ? 'Image of the sample' : 'PSF'}
              </button>
            ))}
          </div>
        ) : (
          <span className="font-semibold">PSF at the focus</span>
        )}
        {!showImage && result && quality === 'preview' && (
          <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[11px] text-amber-700 dark:text-amber-300">preview</span>
        )}
        <span className="ml-auto text-muted-foreground">{shown ? `${shown.nx}×${shown.ny}×${shown.nz}` : ''}</span>
      </div>
      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
        <div className="relative h-[210px] shrink-0 overflow-hidden rounded-lg border bg-[#0b0b10]">
          <VolumeViewer volume={shown} settings={showImage ? imageRender : render} />
        </div>
        {showImage && image ? (
          <div className="grid grid-cols-2 gap-3 text-xs text-muted-foreground">
            <figure className="min-w-0">
              <SliceCanvas volume={image} plane="xy" index={Math.floor(image.nz / 2)} colormap={render.colormap} mapping={render.log ? { kind: 'log', decades: render.logDecades } : { kind: 'linear', gamma: render.gamma }} />
              <figcaption className="mt-1">XY at the centre, {image.sizeX.toFixed(1)} µm wide</figcaption>
            </figure>
            <figure className="min-w-0">
              <SliceCanvas volume={image} plane="xz" index={Math.floor(image.ny / 2)} colormap={render.colormap} mapping={render.log ? { kind: 'log', decades: render.logDecades } : { kind: 'linear', gamma: render.gamma }} />
              <figcaption className="mt-1">XZ through the centre, {image.sizeZ.toFixed(1)} µm tall</figcaption>
            </figure>
            <div className="col-span-2 flex items-center gap-2">
              <span>0</span>
              <div className="h-2 flex-1 rounded" style={{ background: colormapCss(render.colormap) }} />
              <span>max</span>
            </div>
          </div>
        ) : result ? (
          <SliceViews result={result} colormap={render.colormap} log={render.log} logDecades={render.logDecades} gamma={render.gamma} />
        ) : (
          <div className="flex h-24 items-center justify-center rounded-lg border text-xs text-muted-foreground">No volume yet</div>
        )}
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px]">
          {!showImage && (
          <div className="inline-flex overflow-hidden rounded-md border">
            {(['mip', 'composite'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => onRender({ mode: m })}
                className={cn('px-2 py-0.5', render.mode === m ? 'bg-primary text-primary-foreground' : 'text-muted-foreground')}
              >
                {m === 'mip' ? 'Max' : 'Composite'}
              </button>
            ))}
          </div>
          )}
          <select className="rounded-md border bg-background px-1.5 py-0.5" value={render.colormap} onChange={(e) => onRender({ colormap: e.target.value as ColormapName })}>
            {COLORMAPS.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <label className="inline-flex items-center gap-1">
            <input type="checkbox" checked={render.log} onChange={(e) => onRender({ log: e.target.checked })} />
            log
          </label>
          <button type="button" className="ml-auto inline-flex items-center gap-0.5 text-muted-foreground" onClick={() => setShowControls((v) => !v)}>
            more <ChevronDown className={cn('size-3 transition-transform', showControls && 'rotate-180')} />
          </button>
        </div>
        {showControls && (
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
            {render.log ? (
              <label className="flex items-center gap-1">
                decades
                <input type="range" className="pg-range min-w-0 flex-1" min={1} max={7} step={1} value={render.logDecades} onChange={(e) => onRender({ logDecades: Number(e.target.value) })} />
                {render.logDecades}
              </label>
            ) : (
              <label className="flex items-center gap-1">
                γ
                <input type="range" className="pg-range min-w-0 flex-1" min={0.2} max={2} step={0.05} value={render.gamma} onChange={(e) => onRender({ gamma: Number(e.target.value) })} />
                {render.gamma.toFixed(2)}
              </label>
            )}
            <label className="flex items-center gap-1">
              cut
              <input type="range" className="pg-range min-w-0 flex-1" min={0} max={0.9} step={0.01} value={render.threshold} onChange={(e) => onRender({ threshold: Number(e.target.value) })} />
            </label>
            {render.mode === 'composite' && (
              <label className="flex items-center gap-1">
                opacity
                <input type="range" className="pg-range min-w-0 flex-1" min={0.05} max={2} step={0.05} value={render.opacity} onChange={(e) => onRender({ opacity: Number(e.target.value) })} />
              </label>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
