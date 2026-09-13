'use client';

import { useEffect, useRef, useState } from 'react';
import { ChevronDown, Download } from 'lucide-react';
import { cn } from '@/lib/cn';
import type { ColormapName } from './colormaps';
import { paintSlice, type SliceMapping } from './slice-canvas';
import { argmax3, type Volume } from './volume';

export interface ExportImage {
  volume: Volume;
  /** XY plane through the brightest voxel (PSF) or the centre (sample image). */
  brightestPlane: boolean;
  colormap: ColormapName;
  mapping: SliceMapping;
}

interface Props {
  /** The volume shown right now, for the picture exports. */
  image: ExportImage | null;
  onTiff: () => void;
  onConfig: () => void;
  download: (blob: Blob, name: string) => void;
  disabledVolume: boolean;
  /** Open the list above the button (for a footer). */
  openUpward?: boolean;
}

/** The XY and XZ slices side by side, upscaled, as a PNG or JPEG blob. */
async function sliceImage(img: ExportImage, type: 'image/png' | 'image/jpeg'): Promise<Blob> {
  const { volume: v } = img;
  const zi = img.brightestPlane ? argmax3(v)[0] : Math.floor(v.nz / 2);
  const yi = Math.floor(v.ny / 2);
  const xy = document.createElement('canvas');
  const xz = document.createElement('canvas');
  paintSlice(xy, { volume: v, plane: 'xy', index: zi, colormap: img.colormap, mapping: img.mapping });
  paintSlice(xz, { volume: v, plane: 'xz', index: yi, colormap: img.colormap, mapping: img.mapping });
  // physical scale: same µm per pixel for both panels, at least 4 px per voxel
  const scale = Math.max(4, Math.ceil(600 / v.nx));
  const pxPerUm = (v.nx * scale) / v.sizeX;
  const wXY = Math.round(v.sizeX * pxPerUm);
  const hXY = Math.round(v.sizeY * pxPerUm);
  const hXZ = Math.round(v.sizeZ * pxPerUm);
  const gap = 24;
  const out = document.createElement('canvas');
  out.width = wXY * 2 + gap * 3;
  out.height = Math.max(hXY, hXZ) + gap * 2 + 40;
  const ctx = out.getContext('2d');
  if (!ctx) throw new Error('canvas not available');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.imageSmoothingEnabled = v.nx >= 48;
  ctx.drawImage(xy, gap, gap, wXY, hXY);
  ctx.drawImage(xz, gap * 2 + wXY, gap, wXY, hXZ);
  ctx.fillStyle = '#ddd';
  ctx.font = '16px system-ui, sans-serif';
  ctx.fillText(`XY, ${v.sizeX.toFixed(2)} µm wide`, gap, out.height - 14);
  ctx.fillText(`XZ (z up), ${v.sizeZ.toFixed(2)} µm tall`, gap * 2 + wXY, out.height - 14);
  return new Promise((resolve, reject) => out.toBlob((b) => (b ? resolve(b) : reject(new Error('could not encode the image'))), type, 0.92));
}

/** "Export" dropdown: the volume as TIFF, the slices as PNG / JPEG, the parameters as JSON. */
export function ExportMenu({ image, onTiff, onConfig, download, disabledVolume, openUpward = false }: Props) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener('mousedown', close);
    return () => window.removeEventListener('mousedown', close);
  }, [open]);

  const item = (label: string, hint: string, onClick: () => void, disabled = false) => (
    <button
      type="button"
      disabled={disabled}
      className="flex w-full flex-col items-start rounded-md px-2 py-1.5 text-left hover:bg-accent disabled:opacity-50"
      onClick={() => {
        setOpen(false);
        onClick();
      }}
    >
      <span className="text-sm">{label}</span>
      <span className="text-xs text-muted-foreground">{hint}</span>
    </button>
  );

  return (
    <div ref={root} className="relative">
      <button type="button" className="inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <Download className="size-4" />
        Export
        <ChevronDown className={cn('size-3.5 text-muted-foreground transition-transform', open && 'rotate-180')} />
      </button>
      {open && (
        <div className={cn('absolute z-20 w-64 rounded-lg border bg-popover p-1.5 shadow-lg', openUpward ? 'bottom-full right-0 mb-1' : 'left-0 top-full mt-1')}>
          {item('Volume as TIFF', '32-bit multi-page, with voxel size; opens in Fiji and napari', onTiff, disabledVolume)}
          {item('Slices as PNG', 'XY and XZ through the focus, lossless', () => image && sliceImage(image, 'image/png').then((b) => download(b, 'psf_slices.png')), !image)}
          {item('Slices as JPEG', 'The same picture, smaller file', () => image && sliceImage(image, 'image/jpeg').then((b) => download(b, 'psf_slices.jpg')), !image)}
          {item('Config as JSON', 'psf_config.json for the CLI and the napari plugin', onConfig)}
        </div>
      )}
    </div>
  );
}
