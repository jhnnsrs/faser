'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { asset } from '@/lib/base-path';
import type { Derived, Params } from './params';
import type { SampleSpec, Volume } from './volume';

export interface PsfResult {
  /** Intensities in (z, y, x) C-order. */
  data: Float32Array;
  nz: number;
  ny: number;
  nx: number;
  /** Maximum before normalization (1 when Normalize is YES). */
  max: number;
  derived: Derived;
  /** Compute time in the worker, ms. */
  ms: number;
  params: Params;
  /** true for the scalar Debye model, false for the vectorial one. */
  scalar: boolean;
}

interface RawVolume {
  data: Float32Array;
  nz: number;
  ny: number;
  nx: number;
  max: number;
  ms: number;
}

type Pending =
  | { type: 'generate'; resolve: (r: PsfResult) => void; reject: (e: Error) => void; params: Params; scalar: boolean }
  | { type: 'derive'; resolve: (d: Derived) => void; reject: (e: Error) => void }
  | { type: 'volume'; resolve: (v: RawVolume) => void; reject: (e: Error) => void };

export type WorkerStatus = 'loading' | 'ready' | 'error';

/** The SLM pixel phases can be large; they are sent as a typed array, not as JSON. */
function withoutSlmPhase(p: Params): Params {
  return p.SLM ? { ...p, SLM: { ...p.SLM, phase: [] } } : p;
}

function slmPhaseArray(p: Params): Float32Array | null {
  return p.SLM ? Float32Array.from(p.SLM.phase) : null;
}

/**
 * Runs the WebAssembly simulator in a Web Worker (public/psf-worker.js) so
 * the page and the 3D views stay responsive while a volume is computed.
 */
export function usePsfWorker() {
  const workerRef = useRef<Worker | null>(null);
  const pending = useRef(new Map<number, Pending>());
  const nextId = useRef(1);
  const [status, setStatus] = useState<WorkerStatus>('loading');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let worker: Worker;
    try {
      worker = new Worker(asset('/psf-worker.js'), { type: 'module' });
    } catch (e) {
      const message = `Web Workers are not available: ${e instanceof Error ? e.message : String(e)}`;
      queueMicrotask(() => {
        setStatus('error');
        setError(message);
      });
      return;
    }
    workerRef.current = worker;
    const map = pending.current;

    worker.onmessage = (event: MessageEvent) => {
      const msg = event.data;
      if (msg.type === 'ready') {
        setStatus('ready');
        return;
      }
      if (msg.id == null) {
        if (msg.type === 'error') {
          setStatus('error');
          setError(msg.message);
        }
        return;
      }
      const p = map.get(msg.id);
      if (!p) return;
      map.delete(msg.id);
      if (msg.type === 'error') {
        p.reject(new Error(msg.message));
      } else if (msg.type === 'result' && p.type === 'generate') {
        p.resolve({
          data: msg.data,
          nz: msg.nz,
          ny: msg.ny,
          nx: msg.nx,
          max: msg.max,
          derived: msg.derived,
          ms: msg.ms,
          params: p.params,
          scalar: p.scalar,
        });
      } else if (msg.type === 'derived' && p.type === 'derive') {
        p.resolve(msg.derived);
      } else if (msg.type === 'volume' && p.type === 'volume') {
        p.resolve({ data: msg.data, nz: msg.nz, ny: msg.ny, nx: msg.nx, max: msg.max, ms: msg.ms });
      } else {
        p.reject(new Error(`unexpected worker reply ${msg.type}`));
      }
    };
    worker.onerror = (e) => {
      setStatus('error');
      setError(e.message || 'the simulator worker crashed');
    };

    return () => {
      worker.terminate();
      workerRef.current = null;
      for (const p of map.values()) p.reject(new Error('worker terminated'));
      map.clear();
    };
  }, []);

  const post = useCallback((message: Record<string, unknown>, entry: Pending, transfer: Transferable[] = []) => {
    const worker = workerRef.current;
    if (!worker) {
      entry.reject(new Error('simulator not loaded'));
      return;
    }
    const id = nextId.current++;
    pending.current.set(id, entry);
    worker.postMessage({ id, ...message }, transfer);
  }, []);

  const generate = useCallback(
    (params: Params, scalar = false) =>
      new Promise<PsfResult>((resolve, reject) =>
        post(
          { type: 'generate', params: withoutSlmPhase(params), scalar, slmPhase: slmPhaseArray(params) },
          { type: 'generate', resolve, reject, params, scalar },
        ),
      ),
    [post],
  );

  const derive = useCallback(
    (params: Params) =>
      new Promise<Derived>((resolve, reject) =>
        post({ type: 'derive', params: withoutSlmPhase(params) }, { type: 'derive', resolve, reject }),
      ),
    [post],
  );

  /** Synthetic ground-truth volume for the given spec. */
  const sample = useCallback(
    (spec: SampleSpec) =>
      new Promise<Volume>((resolve, reject) =>
        post(
          { type: 'sample', spec },
          {
            type: 'volume',
            resolve: (v) =>
              resolve({
                data: v.data,
                nx: v.nx,
                ny: v.ny,
                nz: v.nz,
                max: v.max,
                sizeX: v.nx * spec.dx,
                sizeY: v.ny * spec.dx,
                sizeZ: v.nz * spec.dz,
                ms: v.ms,
              }),
            reject,
          },
        ),
      ),
    [post],
  );

  /** Image `sampleVolume` through `psf` (3-D convolution + optional shot noise). */
  const convolve = useCallback(
    (sampleVolume: Volume, psf: Volume, photons: number, seed: number) =>
      new Promise<Volume>((resolve, reject) =>
        post(
          {
            type: 'convolve',
            // copies: both volumes stay usable on the page
            sample: { data: sampleVolume.data.slice(), nz: sampleVolume.nz, ny: sampleVolume.ny, nx: sampleVolume.nx },
            psf: { data: psf.data.slice(), nz: psf.nz, ny: psf.ny, nx: psf.nx },
            photons,
            seed,
          },
          {
            type: 'volume',
            resolve: (v) =>
              resolve({
                data: v.data,
                nx: v.nx,
                ny: v.ny,
                nz: v.nz,
                max: v.max,
                sizeX: sampleVolume.sizeX,
                sizeY: sampleVolume.sizeY,
                sizeZ: sampleVolume.sizeZ,
                ms: v.ms,
              }),
            reject,
          },
        ),
      ),
    [post],
  );

  return { status, error, generate, derive, sample, convolve };
}
