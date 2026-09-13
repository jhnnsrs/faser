'use client';

import dynamic from 'next/dynamic';

// The playground uses WebGL, Web Workers and WebAssembly: browser only.
const Playground = dynamic(() => import('./playground').then((m) => m.Playground), {
  ssr: false,
  loading: () => (
    <div className="flex flex-1 items-center justify-center p-16 text-sm text-muted-foreground">Loading the playground…</div>
  ),
});

export function PlaygroundLoader() {
  return <Playground />;
}
