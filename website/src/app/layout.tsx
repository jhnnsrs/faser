import { Inter } from 'next/font/google';
import type { Metadata } from 'next';
import { Provider } from '@/components/site';
import { appName } from '@/lib/shared';
import 'katex/dist/katex.min.css';
import './global.css';

const inter = Inter({
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: {
    template: `%s | ${appName}`,
    default: appName,
  },
  description: 'faser: a vectorial point spread function simulator for high-NA microscopy.',
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
