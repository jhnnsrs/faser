import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { BookOpen, FlaskConical } from 'lucide-react';
import { Logo } from '@/components/site';
import { appName, gitConfig } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <>
          <Logo className="size-6" />
          <span className="font-semibold">{appName}</span>
        </>
      ),
    },
    links: [
      {
        text: 'Docs',
        url: '/docs',
        active: 'nested-url',
        icon: <BookOpen />,
      },
      {
        text: 'Playground',
        url: '/playground',
        active: 'nested-url',
        icon: <FlaskConical />,
      },
    ],
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
