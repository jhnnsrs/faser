# faser website

The documentation site and in-browser playground for **faser**, published to
[jhnnsrs.github.io/faser](https://jhnnsrs.github.io/faser). It is a
[Next.js](https://nextjs.org) app built with [Fumadocs](https://fumadocs.dev) and
exported as a static site to GitHub Pages.

The playground runs the faser simulator itself: the Rust core in `../rust/core` is
compiled to WebAssembly (`../rust/wasm`) and driven from a Web Worker, the volume is
ray-marched with three.js, and an idealized microscope model follows the parameters.

## Quick start

Requires **Node 22** (see [`.nvmrc`](.nvmrc)), **pnpm**, a Rust toolchain with the
`wasm32-unknown-unknown` target and [`wasm-pack`](https://rustwasm.github.io/wasm-pack/):

```bash
rustup target add wasm32-unknown-unknown
curl -sSfL https://rustwasm.github.io/wasm-pack/installer/init.sh | sh

pnpm install
pnpm build:wasm   # -> public/wasm/ (gitignored), rerun after changing rust/core
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). Content and components hot-reload
as you edit; the wasm module does not, rerun `pnpm build:wasm`.

## Scripts

| Command            | Description                                                            |
| ------------------ | ---------------------------------------------------------------------- |
| `pnpm build:wasm`  | Compile the simulator to WebAssembly into `public/wasm/`.              |
| `pnpm dev`         | Start the dev server with hot reload.                                  |
| `pnpm build`       | Build the static export into `out/`. Set `PAGES_BASE_PATH=/faser` for the GitHub Pages layout. |
| `pnpm start`       | Serve the built `out/` directory locally.                              |
| `pnpm types:check` | Regenerate MDX/route types and run `tsc --noEmit`.                     |
| `pnpm lint`        | Run ESLint.                                                            |

## Writing content

All documentation lives in [`content/docs`](content/docs) as MDX. Each folder has a
`meta.json` that controls sidebar ordering and labels. Frontmatter needs a `title`
and a `description`. `Callout`, `Steps`/`Step`, `Files`/`File`/`Folder` and
`Tabs`/`Tab` are available in every page without importing them; LaTeX works with
`$...$` and `$$...$$` (remark-math + KaTeX). Images go to `public/img` and are
referenced as `/img/name.png`.

## Project layout

```
content/docs/                 MDX documentation content
public/psf-worker.js          Web Worker that runs the wasm simulator
public/wasm/                  wasm-pack output (generated, gitignored)
src/app/(home)/               Landing page and the /playground route
src/app/docs/                 Documentation layout and dynamic pages
src/app/api/search/           Static Orama search index
src/app/llms*.txt, llms.mdx   Markdown exports of the docs for LLMs
src/components/playground/    Parameter panel, worker hook, volume renderer,
                              microscope model, slice views, TIFF export,
                              vectorial-vs-scalar comparison, imaging simulation
src/components/site/          Provider, search dialog, MDX components, logo
src/lib/                      Source loader, layout options, base-path helper
```

## Deployment

`.github/workflows/deploy.yml` installs Rust + wasm-pack, builds the wasm, builds the
site with `PAGES_BASE_PATH=/faser` and uploads `out/` to GitHub Pages on every push
to `main`.
