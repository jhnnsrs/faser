// Web Worker that runs the faser simulator (WebAssembly, see rust/wasm).
// Kept as a plain module script in public/ so it needs no bundler support:
// the page starts it with `new Worker(asset('/psf-worker.js'), { type: 'module' })`
// and the glue code in ./wasm/ is produced by `pnpm build:wasm`.
//
// Protocol (all requests carry an `id` chosen by the page):
//   -> { id, type: 'generate', params, scalar }   params: psf_config.json object
//   <- { id, type: 'result', data: Float32Array, nz, ny, nx, max, derived, ms }
//   -> { id, type: 'derive', params }
//   <- { id, type: 'derived', derived }
//   -> { id, type: 'sample', spec }               spec: SampleSpec object
//   <- { id, type: 'volume', data, nz, ny, nx, max, ms }
//   -> { id, type: 'convolve', sample: {data,nz,ny,nx}, psf: {data,nz,ny,nx}, photons, seed }
//   <- { id, type: 'volume', data, nz, ny, nx, max, ms }
//   <- { id, type: 'error', message }
//   <- { type: 'ready', defaults }                once the module is instantiated

import init, { convolve, default_params, derive, generate_psf, generate_sample } from './wasm/faser_wasm.js';

const ready = init().then(() => {
  postMessage({ type: 'ready', defaults: JSON.parse(default_params()) });
});

ready.catch((e) => postMessage({ type: 'error', id: null, message: `failed to load simulator: ${e}` }));

function volumeMessage(id, volume, t0) {
  const data = volume.data; // copied out of wasm memory
  const message = { id, type: 'volume', data, nz: volume.nz, ny: volume.ny, nx: volume.nx, max: volume.max, ms: performance.now() - t0 };
  volume.free();
  return message;
}

self.onmessage = async (event) => {
  const { id, type } = event.data;
  try {
    await ready;
    if (type === 'generate') {
      const { params, scalar } = event.data;
      const t0 = performance.now();
      const volume = generate_psf(JSON.stringify(params), Boolean(scalar));
      const data = volume.data;
      const message = {
        id,
        type: 'result',
        data,
        nz: volume.nz,
        ny: volume.ny,
        nx: volume.nx,
        max: volume.max,
        derived: JSON.parse(volume.derived),
        ms: performance.now() - t0,
      };
      volume.free();
      postMessage(message, [data.buffer]);
    } else if (type === 'derive') {
      postMessage({ id, type: 'derived', derived: JSON.parse(derive(JSON.stringify(event.data.params))) });
    } else if (type === 'sample') {
      const t0 = performance.now();
      const message = volumeMessage(id, generate_sample(JSON.stringify(event.data.spec)), t0);
      postMessage(message, [message.data.buffer]);
    } else if (type === 'convolve') {
      const { sample, psf, photons, seed } = event.data;
      const t0 = performance.now();
      const out = convolve(sample.data, sample.nz, sample.ny, sample.nx, psf.data, psf.nz, psf.ny, psf.nx, photons, BigInt(seed));
      const message = volumeMessage(id, out, t0);
      postMessage(message, [message.data.buffer]);
    }
  } catch (e) {
    postMessage({ id, type: 'error', message: e instanceof Error ? e.message : String(e) });
  }
};
