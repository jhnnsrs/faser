/** The Zernike modes of the simulator (rust/core/src/lib.rs `zernike`), shared by the pupil drawing and the SLM layer. */
import type { ZernikeCoeffs } from './params';

/** Zernike phase (rad) at unit-pupil (x, y) for a set of coefficients, same polynomials as the simulator. */
export function zernikePhase(p: ZernikeCoeffs, x: number, y: number): number {
  const rho = Math.hypot(x, y);
  const phi = Math.atan2(y, x);
  const rho2 = rho * rho;
  const rho3 = rho2 * rho;
  const rho4 = rho2 * rho2;
  const rho6 = rho4 * rho2;
  const s6 = Math.sqrt(6);
  const s8 = Math.sqrt(8);
  return (
    p.a0 +
    p.a1 * 2 * rho * Math.sin(phi) +
    p.a2 * 2 * rho * Math.cos(phi) +
    p.a3 * s6 * rho2 * Math.sin(2 * phi) +
    p.a4 * Math.sqrt(3) * (2 * rho2 - 1) +
    p.a5 * s6 * rho2 * Math.cos(2 * phi) +
    p.a6 * s8 * rho3 * Math.sin(3 * phi) +
    p.a7 * s8 * (3 * rho3 - 2 * rho) * Math.sin(phi) +
    p.a8 * s8 * (3 * rho3 - 2 * rho) * Math.cos(phi) +
    p.a9 * s8 * rho3 * Math.cos(3 * phi) +
    p.a12 * Math.sqrt(5) * (6 * rho4 - 6 * rho2 + 1) +
    p.a24 * Math.sqrt(7) * (20 * rho6 - 30 * rho4 + 12 * rho2 - 1)
  );
}


/** Peak-to-valley bound of the Zernike phase over the pupil (rad), used to size the θ quadrature. */
export function zernikeRange(c: ZernikeCoeffs): number {
  return (
    4 *
    (Math.abs(c.a1) + Math.abs(c.a2) + Math.abs(c.a3) + Math.abs(c.a4) + Math.abs(c.a5) + Math.abs(c.a6) +
      Math.abs(c.a7) + Math.abs(c.a8) + Math.abs(c.a9) + Math.abs(c.a12) + Math.abs(c.a24))
  );
}
