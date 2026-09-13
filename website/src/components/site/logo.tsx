import type { ComponentProps } from 'react';

/**
 * The faser mark: a focused beam. Two cones meeting at the focal point, with
 * the focal spot drawn in the brand colour so it re-tints with the theme.
 */
export function Logo({ className, ...props }: ComponentProps<'svg'>) {
  return (
    <svg
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
      className={className}
      {...props}
    >
      <path
        d="M12 8h40L34 32l18 24H12l18-24L12 8Z"
        fill="var(--brand-logo-mid)"
        opacity="0.9"
      />
      <path d="M20 8h24L32 24 20 8Z" fill="var(--brand-logo-light)" />
      <path d="M20 56h24L32 40 20 56Z" fill="var(--brand-logo-dark)" />
      <circle cx="32" cy="32" r="5" fill="#fff" />
      <circle cx="32" cy="32" r="2.5" fill="var(--brand-logo-light)" />
    </svg>
  );
}
