// @ts-check
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';

import tailwind from '@astrojs/tailwind';

// https://astro.build/config
export default defineConfig({
    site: 'https://jhnnsrs.github.io',
    base: 'faser',
    integrations: [starlight({
        title: 'Faser',
        social: {
            github: 'https://github.com/jhnnsrs/faser',
        },
        customCss: [
            // Path to your Tailwind base styles:
            './src/tailwind.css',
          ],
        sidebar: [
            {
                label: 'Guides',
                items: [
                    // Each item here is one entry in the navigation menu.
                    { label: 'First Steps', slug: 'guides/introduction' },
                ],   
            },
            {
                label: 'Installation',
                autogenerate: { directory: 'installation' },
            },
            {
                label: 'Reference',
                autogenerate: { directory: 'reference' },
            },
        ],
		}), tailwind(
            {
                applyBaseStyles: false,
            }
        )],
});