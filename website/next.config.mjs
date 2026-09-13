import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

// Set by the GitHub Pages workflow ("/faser": the site lives at
// jhnnsrs.github.io/faser). Empty by default so local dev / `serve out` stay clean.
const basePath = process.env.PAGES_BASE_PATH || '';

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  // Folders with index.html so both /playground and /playground/ resolve on GitHub Pages.
  trailingSlash: true,
  basePath,
  // Expose the base path to client components so they can prefix raw asset
  // references (the wasm worker, images in raw HTML; see src/lib/base-path.ts).
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
  reactStrictMode: true,
  images: { unoptimized: true },
  // Hosts allowed to reach the dev server's HMR / _next resources besides
  // localhost. Next refuses a bare '*' wildcard, so list the machine's names,
  // LAN/Tailscale IPs and a subdomain wildcard for tunnels.
  // Dev-only; ignored by `next build`.
  allowedDevOrigins: [
    'jhnnsrs-lab',
    'jhnnsrs-server',
    '*.jhnnsrs-lab',
    '*.jhnnsrs-server',
    '*.local',
    '*.ts.net',
    '140.78.80.150',
    '100.116.108.106',
    '192.168.*.*',
    '172.*.*.*',
  ],
};

export default withMDX(config);
