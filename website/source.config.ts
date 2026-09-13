import { defineConfig, defineDocs } from 'fumadocs-mdx/config';
import { metaSchema, pageSchema } from 'fumadocs-core/source/schema';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// Prefix absolute asset paths in RAW HTML written inside MDX (e.g.
// <img src="/img/x.png">) with the deploy base path. Markdown ![](...) images
// are bundled by fumadocs and skipped here. No-op when PAGES_BASE_PATH is unset.
function rehypeBasePath() {
  const basePath = process.env.PAGES_BASE_PATH || '';
  const attrs: Record<string, string[]> = {
    img: ['src'],
    source: ['src'],
    video: ['src', 'poster'],
    audio: ['src'],
    a: ['href'],
  };
  const prefix = (val: unknown): unknown => {
    if (
      typeof val !== 'string' ||
      !val.startsWith('/') ||
      val.startsWith('//') ||
      val.startsWith(`${basePath}/`)
    ) {
      return val;
    }
    return `${basePath}${val}`;
  };
  const walk = (node: any) => {
    if (!node || typeof node !== 'object') return;
    if (node.type === 'element' && node.properties) {
      const keys = attrs[node.tagName];
      if (keys) {
        for (const k of keys) {
          if (node.properties[k]) node.properties[k] = prefix(node.properties[k]);
        }
      }
    }
    if (
      (node.type === 'mdxJsxFlowElement' || node.type === 'mdxJsxTextElement') &&
      Array.isArray(node.attributes)
    ) {
      const keys = attrs[node.name];
      if (keys) {
        for (const attr of node.attributes) {
          if (attr?.type === 'mdxJsxAttribute' && keys.includes(attr.name)) {
            attr.value = prefix(attr.value);
          }
        }
      }
    }
    if (Array.isArray(node.children)) node.children.forEach(walk);
  };
  return (tree: any) => {
    if (basePath) walk(tree);
    return tree;
  };
}

export const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    schema: pageSchema,
    postprocess: {
      includeProcessedMarkdown: true,
    },
  },
  meta: {
    schema: metaSchema,
  },
});

export default defineConfig({
  mdxOptions: {
    // LaTeX in the theory page: $...$ / $$...$$ via remark-math + KaTeX.
    remarkPlugins: [remarkMath],
    rehypePlugins: (plugins) => [rehypeKatex, ...plugins, rehypeBasePath],
  },
});
