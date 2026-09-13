import defaultMdxComponents from 'fumadocs-ui/mdx';
import { ImageZoom } from 'fumadocs-ui/components/image-zoom';
import { Callout } from 'fumadocs-ui/components/callout';
import { Step, Steps } from 'fumadocs-ui/components/steps';
import { File, Files, Folder } from 'fumadocs-ui/components/files';
import { Tab, Tabs } from 'fumadocs-ui/components/tabs';
import type { MDXComponents } from 'mdx/types';
import type { ComponentProps } from 'react';

/**
 * Markdown images (`![alt](/img/...)`) are bundled by fumadocs-mdx's
 * remark-image and open in a lightbox on click.
 */
function DocImage(props: ComponentProps<'img'>) {
  return <ImageZoom {...(props as ComponentProps<typeof ImageZoom>)} className="w-full rounded-lg" />;
}

/**
 * Components available in every MDX page without importing them:
 * Callout, Steps/Step, Files/File/Folder, Tabs/Tab.
 */
export function getMDXComponents(components?: MDXComponents): MDXComponents {
  return {
    ...defaultMdxComponents,
    img: DocImage,
    Callout,
    Steps,
    Step,
    Files,
    File,
    Folder,
    Tabs,
    Tab,
    ...components,
  } as MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
