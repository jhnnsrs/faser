import type { Derived, Params } from './params';

/**
 * Draw the back pupil as the simulator sees it: the Gaussian amplitude, the
 * STED phase mask and the Zernike aberration phase, plus the incident
 * polarization. Phase is mapped to hue, amplitude to brightness, so a flat
 * Gaussian beam is a uniformly coloured disc and a vortex is a hue wheel.
 * Mirrors `amplitude`, `phase_mask` and `zernike` in rust/core/src/lib.rs.
 */
export function drawPupil(canvas: HTMLCanvasElement, p: Params, d: Derived, size = 256) {
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const img = ctx.createImageData(size, size);
  const px = img.data;
  const r0 = d.r0;
  const rEff = d.r0_eff / r0; // window-clipped aperture on the unit pupil
  const off = 1 / p.Nxy; // offsets are in units of r0 / Nxy (see theta_ring)
  const half = size / 2;

  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      const x = (i + 0.5 - half) / (half * 0.96); // unit-pupil coordinates
      const y = -(j + 0.5 - half) / (half * 0.96);
      const rr = Math.hypot(x, y);
      const k = (j * size + i) * 4;
      if (rr > 1) {
        px[k + 3] = 0;
        continue;
      }
      // amplitude (µm on the pupil)
      const ax = (x - off * p.Ampl_offset_x) * r0;
      const ay = (y - off * p.Ampl_offset_y) * r0;
      const amp = Math.exp(-(ax * ax + ay * ay) / (p.Waist * p.Waist));

      // phase mask
      let phase = 0;
      const mx = x - off * p.Mask_offset_x;
      const my = y - off * p.Mask_offset_y;
      const mr = Math.hypot(mx, my);
      const mphi = Math.atan2(my, mx);
      const mode = p.Mode;
      if (mode === 'DONUT' || mode === 'DONUT BOTTLE') phase += p.VC * mphi;
      if ((mode === 'BOTTLE' || mode === 'DONUT BOTTLE') && mr <= p.Ring_Radius) phase += p.RC * Math.PI;

      // zernike
      const zx = x - off * p.Aberration_offset_x;
      const zy = y - off * p.Aberration_offset_y;
      phase += zernike(zx, zy, p);

      const outside = rr > rEff;
      const [r, g, b] = hsl(((phase / (2 * Math.PI)) % 1 + 1) % 1, outside ? 0.1 : 0.85, 0.12 + 0.5 * amp);
      px[k] = r;
      px[k + 1] = g;
      px[k + 2] = b;
      px[k + 3] = outside ? 90 : 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  // Aperture ring
  ctx.strokeStyle = 'rgba(255,255,255,0.7)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(half, half, half * 0.96, 0, Math.PI * 2);
  ctx.stroke();
  if (rEff < 0.999) {
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(half, half, half * 0.96 * rEff, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Polarization
  ctx.strokeStyle = 'rgba(255,255,255,0.95)';
  ctx.fillStyle = 'rgba(255,255,255,0.95)';
  ctx.lineWidth = 2.5;
  if (p.Polarization === 1) {
    const psi = (p.Psi * Math.PI) / 180;
    const eps = (p.Epsilon * Math.PI) / 180;
    const a = half * 0.28;
    const b = a * Math.abs(Math.sin(eps)) / Math.max(Math.cos(eps), 1e-3);
    ctx.save();
    ctx.translate(half, half);
    ctx.rotate(-psi);
    ctx.beginPath();
    ctx.ellipse(0, 0, a, Math.min(b, a), 0, 0, Math.PI * 2);
    ctx.stroke();
    // handedness arrow
    if (Math.abs(eps) > 0.02) {
      const s = Math.sign(eps);
      arrow(ctx, a, 0, a, -s * 10, 7);
    } else {
      arrow(ctx, a * 0.5, 0, a, 0, 7);
      arrow(ctx, -a * 0.5, 0, -a, 0, 7);
    }
    ctx.restore();
  } else {
    // radial (2) or azimuthal (3): eight short arrows on a ring
    const rr = half * 0.6;
    for (let n = 0; n < 8; n++) {
      const t = (n / 8) * Math.PI * 2;
      const cx = half + rr * Math.cos(t);
      const cy = half - rr * Math.sin(t);
      const dx = p.Polarization === 2 ? Math.cos(t) : -Math.sin(t);
      const dy = p.Polarization === 2 ? -Math.sin(t) : -Math.cos(t);
      const L = 14;
      arrow(ctx, cx - dx * L, cy - dy * L, cx + dx * L, cy + dy * L, 6);
    }
  }
}

function arrow(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, head: number) {
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  const ang = Math.atan2(y1 - y0, x1 - x0);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * Math.cos(ang - 0.5), y1 - head * Math.sin(ang - 0.5));
  ctx.lineTo(x1 - head * Math.cos(ang + 0.5), y1 - head * Math.sin(ang + 0.5));
  ctx.closePath();
  ctx.fill();
}

/** Zernike phase on the unit pupil, same polynomials as the simulator. */
export function zernike(x: number, y: number, p: Params): number {
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

function hsl(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = h * 6;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r = 0, g = 0, b = 0;
  if (hp < 1) [r, g, b] = [c, x, 0];
  else if (hp < 2) [r, g, b] = [x, c, 0];
  else if (hp < 3) [r, g, b] = [0, c, x];
  else if (hp < 4) [r, g, b] = [0, x, c];
  else if (hp < 5) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  const m = l - c / 2;
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

/** Approximate sRGB colour of monochromatic light (nm), for tinting the beam. */
export function wavelengthToRgb(nm: number): [number, number, number] {
  let r = 0, g = 0, b = 0;
  if (nm < 380) [r, g, b] = [0.5, 0, 1];
  else if (nm < 440) [r, g, b] = [-(nm - 440) / 60, 0, 1];
  else if (nm < 490) [r, g, b] = [0, (nm - 440) / 50, 1];
  else if (nm < 510) [r, g, b] = [0, 1, -(nm - 510) / 20];
  else if (nm < 580) [r, g, b] = [(nm - 510) / 70, 1, 0];
  else if (nm < 645) [r, g, b] = [1, -(nm - 645) / 65, 0];
  else if (nm <= 780) [r, g, b] = [1, 0, 0];
  else [r, g, b] = [0.8, 0.1, 0.1]; // near infrared: draw as deep red
  const f = nm > 700 ? Math.max(0.35, 1 - (nm - 700) / 200) : nm < 420 ? 0.3 + (0.7 * (nm - 380)) / 40 : 1;
  const gamma = 0.8;
  return [Math.pow(r * f, gamma), Math.pow(g * f, gamma), Math.pow(b * f, gamma)];
}
