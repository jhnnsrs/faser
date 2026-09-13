import type { Metadata } from 'next';
import { PlaygroundLoader } from '@/components/playground/loader';

export const metadata: Metadata = {
  title: 'Playground',
  description: 'Generate and explore vectorial PSFs in your browser with the faser simulator compiled to WebAssembly.',
};

export default function PlaygroundPage() {
  return (
    <main className="flex flex-1 flex-col">
      <PlaygroundLoader />
    </main>
  );
}
