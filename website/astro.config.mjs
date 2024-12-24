// @ts-check
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';

import tailwind from '@astrojs/tailwind';

// https://astro.build/config
export default defineConfig({
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