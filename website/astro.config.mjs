// @ts-check
import starlight from '@astrojs/starlight';
import tailwind from '@astrojs/tailwind';
import { defineConfig } from 'astro/config';
import rehypeMathjax from 'rehype-mathjax';
import remarkMath from 'remark-math';

// https://astro.build/config
export default defineConfig({
    site: 'https://jhnnsrs.github.io',
    base: 'faser',
    integrations: [starlight({
        title: 'faser',
        social: {
            github: 'https://github.com/jhnnsrs/faser',
        },
        editLink: {
            baseUrl: "https://github.com/jhnnsrs/faser/edit/master/website/",
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
                    { label: 'What is faser?', slug: 'guides/introduction' },
                    { label: "Theory", slug: "guides/concepts"}
                ],   
            },
            {
                label: 'Installation',
                autogenerate: { directory: 'installation' },
            },
            {
                label: 'Usage',
                autogenerate: { directory: 'usage' },
            },
        ],
		}), tailwind(
            {
                applyBaseStyles: false,
            }
        )],
        markdown: {remarkPlugins: [remarkMath],
            rehypePlugins: [rehypeMathjax],
        },
});