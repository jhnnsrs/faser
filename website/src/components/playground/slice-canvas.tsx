'use client';

import { useEffect, useRef } from 'react';
import { colormapBytes, divergingBytes, type ColormapName } from './colormaps';
import type { Volume } from './volume';

export type SliceMapping =
  | { kind: 'linear'; gamma: number; max?: number }
  | { kind: 'log'; decades: number; max?: number }
  | { kind: 'diverging'; scale: number };

export interface SliceCanvasProps {
  volume: Volume;
  /** 'xy' at z = index, 'xz' at y = index */
  plane: 'xy' | 'xz';
  index: number;
  colormap: ColormapName;
  mapping: SliceMapping;
  className?: string;
  /** Cap the rendered height (px); the width follows the physical aspect ratio. */
  maxHeight?: number;
}

function mapper(m: SliceMapping, volumeMax: number): (v: number) => number {
  switch (m.kind) {
    case 'linear': {
      const max = m.max ?? volumeMax;
      const inv = max > 0 ? 1 / max : 0;
      return (v) => Math.pow(Math.max(v * inv, 0), m.gamma);
    }
    case 'log': {
      const max = m.max ?? volumeMax;
      const inv = max > 0 ? 1 / max : 0;
      const logMin = Math.pow(10, -m.decades);
      return (v) => (Math.log(Math.max(v * inv, logMin)) - Math.log(logMin)) / -Math.log(logMin);
    }
    case 'diverging': {
      const s = m.scale > 0 ? 1 / m.scale : 0;
      return (v) => 0.5 + 0.5 * Math.max(-1, Math.min(1, v * s));
    }
  }
}

/** Draw one plane of a volume into a canvas element (image rows top-down, +y / +z up). */
export function paintSlice(canvas: HTMLCanvasElement, props: Omit<SliceCanvasProps, 'className'>) {
  const { volume: v, plane, index, colormap, mapping } = props;
  const width = v.nx;
  const height = plane === 'xy' ? v.ny : v.nz;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const lut = mapping.kind === 'diverging' ? divergingBytes() : colormapBytes(colormap);
  const map = mapper(mapping, v.max);
  const img = ctx.createImageData(width, height);
  const zi = Math.max(0, Math.min(v.nz - 1, index));
  const yi = Math.max(0, Math.min(v.ny - 1, index));
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      const value =
        plane === 'xy'
          ? v.data[(zi * v.ny + (v.ny - 1 - j)) * v.nx + i]
          : v.data[((v.nz - 1 - j) * v.ny + yi) * v.nx + i];
      const c = Math.max(0, Math.min(255, Math.round(map(value) * 255))) * 4;
      const k = (j * width + i) * 4;
      img.data[k] = lut[c];
      img.data[k + 1] = lut[c + 1];
      img.data[k + 2] = lut[c + 2];
      img.data[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

export function SliceCanvas({ className, maxHeight, ...props }: SliceCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { volume, plane, index, colormap, mapping } = props;
  useEffect(() => {
    if (ref.current) paintSlice(ref.current, { volume, plane, index, colormap, mapping });
  }, [volume, plane, index, colormap, mapping]);
  const ratio = plane === 'xy' ? volume.sizeX / volume.sizeY : volume.sizeX / volume.sizeZ;
  const style: React.CSSProperties = { aspectRatio: String(ratio) };
  if (maxHeight) {
    style.maxHeight = maxHeight;
    style.width = `min(100%, ${(maxHeight * ratio).toFixed(1)}px)`;
  }
  return (
    <div className={className ?? 'overflow-hidden rounded-md border bg-black'} style={style}>
      <canvas
        ref={ref}
        className="h-full w-full"
        style={{ imageRendering: volume.nx < 48 ? 'pixelated' : 'auto' }}
      />
    </div>
  );
}
