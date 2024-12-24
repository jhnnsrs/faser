import starlightPlugin from '@astrojs/starlight-tailwind';

// Generated color palettes
const accent = { 200: '#eeb1e8', 600: '#960092', 900: '#570054', 950: '#3c0a3a' };
const gray = { 100: '#f6f6f6', 200: '#eeeeee', 300: '#c2c2c2', 400: '#8b8b8b', 500: '#585858', 700: '#000000', 800: '#000000', 900: '#000000' };

/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	theme: {
		extend: {
			colors: { accent, gray },
		},
	},
	plugins: [starlightPlugin()],
};