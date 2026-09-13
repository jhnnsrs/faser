import Link from 'next/link';
import Image from 'next/image';
import { ArrowRight, Atom, Cpu, FlaskConical, GitFork, Microscope, Terminal } from 'lucide-react';
import { gitConfig } from '@/lib/shared';
import hero from '@/assets/hero.png';

const features = [
  {
    icon: Atom,
    title: 'Vectorial diffraction theory',
    text: 'Richards–Wolf / Török integration for high-NA focusing through immersion, coverslip and sample, including index mismatch, coverslip tilt and cranial windows.',
  },
  {
    icon: Microscope,
    title: 'Beam shaping and aberrations',
    text: 'Gaussian, donut, bottle and mixed STED beams, arbitrary polarization states and Zernike aberrations, all on the same pupil.',
  },
  {
    icon: Cpu,
    title: 'Fast native simulator',
    text: 'The Debye integral runs in a multi-core Rust backend on the desktop and as WebAssembly right here in your browser.',
  },
  {
    icon: Terminal,
    title: 'Napari, CLI and Python',
    text: 'One config model everywhere: explore interactively in napari, script it from the terminal or call it from your own code.',
  },
];

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col px-6 pb-24 pt-6 sm:px-6 lg:px-8">
      {/* ───────────────────────── Hero ───────────────────────── */}
      <section className="relative isolate w-full overflow-hidden rounded-3xl border border-white/10 bg-[#0a0a0c] text-white">
        <div aria-hidden className="pointer-events-none absolute inset-0 -z-10">
          <div className="absolute -left-24 -top-24 h-[34rem] w-[34rem] rounded-full bg-primary/30 blur-[130px]" />
          <div className="absolute -right-16 top-1/4 h-[28rem] w-[28rem] rounded-full bg-primary/20 blur-[130px]" />
        </div>
        <div aria-hidden className="bg-grain pointer-events-none absolute inset-0 -z-10 opacity-[0.12] mix-blend-overlay" />

        <div className="relative flex flex-col lg:flex-row lg:items-center">
          <div className="relative z-10 max-w-2xl shrink-0 px-6 pt-14 sm:px-12 sm:pt-20 lg:w-[52%] lg:px-16 lg:py-24">
            <span className="inline-flex items-center gap-2 rounded-full border border-primary/40 bg-primary/10 px-3.5 py-1.5 text-sm font-medium text-primary backdrop-blur">
              Vectorial PSF simulation
            </span>
            <h1 className="mt-7 max-w-3xl text-5xl font-bold leading-[1.05] tracking-tight sm:text-6xl xl:text-7xl">
              See your focus
              <br />
              <span className="text-primary">before you image.</span>
            </h1>
            <p className="mt-6 max-w-xl text-lg text-white/60">
              faser simulates the excitation point spread function of high-NA microscopes with a
              full vectorial model: refractive-index mismatch, coverslip tilt, cranial windows, STED
              phase masks, polarization and aberrations.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                href="/playground"
                className="inline-flex items-center gap-2 rounded-full bg-primary px-7 py-3 text-base font-medium text-primary-foreground transition hover:opacity-90"
              >
                <FlaskConical className="size-4" />
                Open the playground
                <ArrowRight className="size-4" />
              </Link>
              <Link
                href="/docs"
                className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/10 px-7 py-3 text-base font-medium text-white transition hover:bg-white/20"
              >
                Read the docs
              </Link>
              <a
                href={`https://github.com/${gitConfig.user}/${gitConfig.repo}`}
                target="_blank"
                rel="noreferrer noopener"
                className="inline-flex items-center gap-2 rounded-full px-4 py-3 text-base text-white/70 transition hover:text-white"
              >
                <GitFork className="size-4" />
                Source
              </a>
            </div>
          </div>

          <div className="relative z-0 mt-10 px-3 pb-3 sm:px-10 lg:my-0 lg:flex lg:min-w-0 lg:flex-1 lg:justify-center lg:px-10 lg:py-12">
            <Image
              src={hero}
              alt="A simulated point spread function rendered in napari"
              priority
              className="w-full max-w-[720px] rounded-2xl border border-white/10 shadow-2xl"
            />
          </div>
        </div>
      </section>

      {/* ──────────────────────── Features ─────────────────────── */}
      <section className="mx-auto mt-16 grid w-full max-w-6xl gap-4 sm:grid-cols-2">
        {features.map(({ icon: Icon, title, text }) => (
          <div key={title} className="rounded-2xl border bg-card p-6">
            <div className="flex items-center gap-3">
              <span className="inline-flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
                <Icon className="size-5" />
              </span>
              <h2 className="text-lg font-semibold">{title}</h2>
            </div>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{text}</p>
          </div>
        ))}
      </section>

      <section className="mx-auto mt-16 w-full max-w-3xl text-center text-sm text-muted-foreground">
        faser is developed by Johannes Roos and Stéphane Bancelin and released under the MIT
        license. Install it with{' '}
        <code className="rounded bg-muted px-1.5 py-0.5 text-foreground">pip install faser</code> or
        as a napari plugin.
      </section>
    </main>
  );
}
