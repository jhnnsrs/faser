'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import type { ComponentRef } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber';
import { Html, Line, OrbitControls, useCursor } from '@react-three/drei';
import { cn } from '@/lib/cn';
import type { ComponentId, Derived, Params } from './params';
import { drawPhasePlate, drawPolarizationPlate, drawPupil, wavelengthToRgb } from './pupil';
import { drawSlmPanel } from './slm';

/**
 * An idealized microscope drawn from the current parameters, with every
 * part clickable. Not to scale (the focal length is thousands of times the
 * coverslip thickness); instead:
 *   1 scene unit = 200 µm for the coverslip, the imaging depth and the focus
 *   shift, while the immersion gap and the objective have fixed schematic
 *   sizes. Angles are real: the cone half-angles are alpha (immersion),
 *   alpha2 (coverslip) and alpha3 (sample) from the simulator, so the front
 *   aperture visibly widens with NA, and index mismatch bends the cone.
 *
 * Frame: y is the optical axis pointing up (towards the objective); y = 0 is
 * the coverslip/sample interface. The output volume of the simulation follows
 * the coverslip frame, so a coverslip tilt is drawn as the objective (and its
 * beam) being tilted. The illumination optics (laser, wave plates, phase
 * plate, SLM) are stacked on the optical axis above the back pupil, the
 * beam coming straight down into the objective.
 *
 * Labels are shown for the hovered / selected part only, or for all parts
 * with `labels = 'always'`.
 *
 * Controls: dragging moves the view up and down the optical axis, shift +
 * drag (or the right button) rotates about the axis, the wheel zooms. The
 * orbit target is kept on the optical axis so the microscope stays centred.
 */

const UNIT = 200; // µm per scene unit
const R_LENS = 2.6; // drawn radius of the front lens (units)
const MAX_DEPTH = 3; // imaging depth is compressed smoothly to at most this many units
const BODY = 2.6; // objective body height (units)
const BELOW = 4; // sample block depth (units)
// Heights of the illumination optics above the back pupil (units); the beam
// comes straight down the optical axis from the laser at the top.
const STACK = { slm: 1.5, phaseplate: 2.9, polarization: 4.3, laser: 5.9 };
const LASER_LENGTH = 1.8;
const METAL = '#5b5b68';
const METAL_DARK = '#3a3a45';

export type LabelMode = 'hover' | 'always';

export interface MicroscopeProps {
  params: Params;
  derived: Derived | null;
  labels: LabelMode;
  /** The SLM displays a Zernike layer on top of its pattern. */
  slmZernike?: boolean;
  /** Reports where the focus lands on the canvas (px from its top-left), for the zoom-in leader line. */
  onFocusScreen?: (pt: { x: number; y: number } | null) => void;
  selected: ComponentId | null;
  hovered: ComponentId | null;
  onSelect: (id: ComponentId | null) => void;
  onHover: (id: ComponentId | null) => void;
}

/** Highlight colour of a part (brand magenta). */
const HI = new THREE.Color('#e04fb8');

type PartProps = {
  id: ComponentId;
  hovered: ComponentId | null;
  selected: ComponentId | null;
  onSelect: (id: ComponentId | null) => void;
  onHover: (id: ComponentId | null) => void;
  children: React.ReactNode;
};

/** Makes its children a clickable, hoverable part of the microscope. */
function Part({ id, hovered, selected, onSelect, onHover, children }: PartProps) {
  return (
    <group
      onPointerOver={(e) => {
        e.stopPropagation();
        onHover(id);
      }}
      onPointerOut={(e) => {
        e.stopPropagation();
        onHover(null);
      }}
      onClick={(e: ThreeEvent<MouseEvent>) => {
        e.stopPropagation();
        if (e.delta > 5) return; // an orbit drag, not a click
        onSelect(selected === id ? null : id);
      }}
    >
      {children}
    </group>
  );
}

function useTone(id: ComponentId, hovered: ComponentId | null, selected: ComponentId | null) {
  const on = selected === id;
  const hov = hovered === id;
  return {
    on,
    hov,
    emissive: on ? HI : hov ? HI : new THREE.Color('#000000'),
    emissiveIntensity: on ? 0.32 : hov ? 0.2 : 0,
    boost: on ? 0.25 : hov ? 0.12 : 0,
  };
}

function Label({
  position,
  children,
  muted,
  active,
  onClick,
}: {
  position: [number, number, number];
  children: React.ReactNode;
  muted?: boolean;
  active?: boolean;
  onClick?: () => void;
}) {
  return (
    <Html position={position} center zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}>
      <button
        type="button"
        onClick={onClick}
        className={cn('pg-label', muted && 'pg-label-muted', active && 'pg-label-active', onClick && 'pg-label-button')}
        style={{ pointerEvents: onClick ? 'auto' : 'none' }}
      >
        {children}
      </button>
    </Html>
  );
}

function Frustum({
  y0,
  y1,
  r0,
  r1,
  color,
  opacity,
  wire,
}: {
  y0: number;
  y1: number;
  r0: number;
  r1: number;
  color: THREE.Color;
  opacity: number;
  wire?: boolean;
}) {
  const h = Math.max(y1 - y0, 1e-3);
  return (
    <mesh position={[0, (y0 + y1) / 2, 0]}>
      <cylinderGeometry args={[Math.max(r1, 1e-3), Math.max(r0, 1e-3), h, 48, 1, true]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} side={THREE.DoubleSide} depthWrite={false} wireframe={wire} />
    </mesh>
  );
}

/**
 * A canvas-backed texture that is redrawn by `draw` whenever `deps` change.
 * Returns the canvas (to create the texture from) and the ref to attach to it.
 */
function useCanvasTexture(draw: (canvas: HTMLCanvasElement) => void, deps: unknown[]): [HTMLCanvasElement, React.RefObject<THREE.CanvasTexture | null>] {
  const [canvas] = useState(() => document.createElement('canvas'));
  const ref = useRef<THREE.CanvasTexture>(null);
  useEffect(() => {
    draw(canvas);
    if (ref.current) ref.current.needsUpdate = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return [canvas, ref];
}

/** A thin optical element (disc) lying in the beam at height y, held by an arm from the rail. */
function StackDisc({
  y,
  radius,
  rail,
  canvas,
  texRef,
  tone,
}: {
  y: number;
  radius: number;
  rail: number;
  canvas: HTMLCanvasElement;
  texRef: React.RefObject<THREE.CanvasTexture | null>;
  tone: ReturnType<typeof useTone>;
}) {
  return (
    <group position={[0, y, 0]}>
      {/* mount ring and arm to the rail */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.04, 0]}>
        <ringGeometry args={[radius, radius + 0.2, 48]} />
        <meshStandardMaterial color={METAL} metalness={0.7} roughness={0.35} side={THREE.DoubleSide} emissive={tone.emissive} emissiveIntensity={tone.emissiveIntensity} />
      </mesh>
      <mesh position={[(rail + radius) / 2, -0.04, 0]}>
        <boxGeometry args={[Math.max(rail - radius, 0.1), 0.1, 0.16]} />
        <meshStandardMaterial color={METAL} metalness={0.7} roughness={0.35} />
      </mesh>
      {/* the element itself, textured on both faces */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[radius, 64]} />
        <meshBasicMaterial transparent toneMapped={false} side={THREE.DoubleSide} depthWrite={false}>
          <canvasTexture ref={texRef} attach="map" args={[canvas]} colorSpace={THREE.SRGBColorSpace} />
        </meshBasicMaterial>
      </mesh>
      {(tone.on || tone.hov) && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
          <ringGeometry args={[radius + 0.02, radius + 0.14, 64]} />
          <meshBasicMaterial color={HI} transparent opacity={tone.on ? 0.9 : 0.5} side={THREE.DoubleSide} />
        </mesh>
      )}
    </group>
  );
}

function Scene({ params: p, derived: d, labels, selected, hovered, onSelect, onHover, resetKey, slmZernike, onFocusScreen }: MicroscopeProps & { resetKey: number }) {
  useCursor(hovered != null, 'pointer', 'auto');
  const show = (id: ComponentId) => labels === 'always' || hovered === id || selected === id;
  const beamColor = useMemo(() => {
    const [r, g, b] = wavelengthToRgb(p.Wavelength * 1000);
    return new THREE.Color(r, g, b);
  }, [p.Wavelength]);

  // Optional elements: drawn only when they are part of the setup. When one
  // is absent but selected / hovered (via the inspector chips), a ghost
  // outline shows where it would sit.
  const hasSlm = p.SLM != null;
  const hasPlate = p.Mode !== 'GAUSSIAN' && p.Mode !== 'LOADED';
  const hasCoverslip = p.Thickness > 0;
  const hasWindow = p.Window === 'CUSTOM';
  const ghost = (id: ComponentId) => hovered === id || selected === id;

  // Geometry (scene units)
  const t = hasCoverslip ? Math.max(p.Thickness / UNIT, 0.04) : 0;
  const compress = (um: number) => MAX_DEPTH * Math.tanh(um / UNIT / MAX_DEPTH);
  const depth = compress(p.Depth);
  const dfoc = compress(p.Depth + (d?.dfoc ?? 0)) - depth;
  const yFocus = -(depth + dfoc);
  const yNominal = -depth;
  const below = Math.max(BELOW, depth + Math.abs(dfoc) + 1);

  const alpha = d?.alpha ?? Math.asin(Math.min(p.NA / p.n1, 1));
  const alphaEff = d?.alpha_eff ?? alpha;
  const alpha2 = d?.alpha2_eff ?? Math.asin(Math.min((p.n1 / p.n2) * Math.sin(alphaEff), 1));
  const alpha3 = d?.alpha3_eff ?? Math.asin(Math.min((p.n1 / p.n3) * Math.sin(alphaEff), 1));
  const tan = (a: number) => Math.tan(Math.min(a, 1.31));

  const r3 = (0 - yFocus) * tan(alpha3);
  const r2 = r3 + t * tan(alpha2);
  const GAP = Math.min(5, Math.max(0.6, (R_LENS - r2) / Math.max(tan(alphaEff), 0.05)));
  const rLens = r2 + GAP * tan(alphaEff);
  const rLensFull = r2 + GAP * tan(alpha);
  const clipped = alphaEff < alpha - 1e-6;

  const yLens = t + GAP;
  const rBody = Math.max(rLens, R_LENS) + 0.3;
  const rPupil = rBody * 0.8;
  const yPupil = yLens + BODY;
  const yTop = yPupil + STACK.laser + LASER_LENGTH;
  const rBeam = rPupil * 0.8; // the collimated beam filling the pupil
  const rail = rBeam + 1.0; // x of the vertical rail holding the optics

  const tilt = ((p.Tilt ?? 0) * Math.PI) / 180;
  const width = Math.max(11, 2 * rBody + 4);

  // Textures
  const [pupilCanvas, pupilTex] = useCanvasTexture((c) => d && drawPupil(c, p, d, 256), [p, d]);
  const [plateCanvas, plateTex] = useCanvasTexture((c) => drawPhasePlate(c, p, 128), [p.Mode, p.VC, p.RC, p.Ring_Radius, p.Mask_offset_x, p.Mask_offset_y, p.Nxy]);
  const [waveCanvas, waveTex] = useCanvasTexture((c) => drawPolarizationPlate(c, p, 128), [p.Polarization, p.Psi, p.Epsilon]);
  const [slmCanvas, slmTex] = useCanvasTexture(
    (c) => {
      if (p.SLM) drawSlmPanel(c, p.SLM, p.SLM.phase, 256);
    },
    [p.SLM],
  );

  const windowHeight = 1.5;
  const rWindow = (t + windowHeight - yFocus) * tan(alphaEff);
  const immersionColor = p.n1 > 1.45 ? '#e8c46a' : p.n1 > 1.1 ? '#6fb7ff' : '#ffffff';

  // Bounding box of the drawing, quantized so the camera only re-fits on big changes.
  const xMin = -width / 2;
  const xMax = width / 2;
  const yMin = -below;
  const yMax = yTop + 0.5;
  const halfX = Math.ceil(((xMax - xMin) / 2) * 2) / 2;
  const halfY = Math.ceil(((yMax - yMin) / 2) * 2) / 2;
  const centerX = Math.round(((xMax + xMin) / 2) * 2) / 2;
  const centerY = Math.round(((yMax + yMin) / 2) * 2) / 2;

  const part = (id: ComponentId) => ({ id, hovered, selected, onSelect, onHover });
  const tone = {
    laser: useTone('laser', hovered, selected),
    polarization: useTone('polarization', hovered, selected),
    phaseplate: useTone('phaseplate', hovered, selected),
    slm: useTone('slm', hovered, selected),
    objective: useTone('objective', hovered, selected),
    pupil: useTone('pupil', hovered, selected),
    coverslip: useTone('coverslip', hovered, selected),
    sample: useTone('sample', hovered, selected),
    window: useTone('window', hovered, selected),
    focus: useTone('focus', hovered, selected),
  };
  const isOn = (id: ComponentId) => selected === id;

  return (
    <>
      <Fit halfX={halfX} halfY={halfY} centerX={centerX} centerY={centerY} resetKey={resetKey} />
      <AxisControls centerY={centerY} />
      {onFocusScreen && <FocusProbe y={yFocus} onScreen={onFocusScreen} />}
      <ambientLight intensity={0.9} />
      <directionalLight position={[6, 10, 4]} intensity={1.4} />
      <directionalLight position={[-6, 4, -6]} intensity={0.4} />

      {/* Sample */}
      <Part {...part('sample')}>
        <mesh position={[0, -below / 2, 0]}>
          <boxGeometry args={[width, below, width * 0.7]} />
          <meshStandardMaterial
            color="#e08a9a"
            transparent
            opacity={0.16 + tone.sample.boost}
            depthWrite={false}
            emissive={tone.sample.emissive}
            emissiveIntensity={tone.sample.emissiveIntensity}
          />
        </mesh>
      </Part>
      {show('sample') && (
        <Label position={[-width / 2 + 0.2, -below + 0.5, 0]} muted active={isOn('sample')} onClick={() => onSelect('sample')}>
          sample n₃ = {p.n3.toFixed(3)}
        </Label>
      )}

      {/* Cranial window (skull) on the coverslip, only when present */}
      {hasWindow ? (
        <Part {...part('window')}>
          <group position={[0, t + windowHeight / 2, 0]}>
            <mesh>
              <cylinderGeometry args={[width * 0.45, width * 0.45, windowHeight, 64, 1, true]} />
              <meshStandardMaterial color="#d9cfae" side={THREE.BackSide} transparent opacity={0.6} emissive={tone.window.emissive} emissiveIntensity={tone.window.emissiveIntensity} />
            </mesh>
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, windowHeight / 2, 0]}>
              <ringGeometry args={[Math.max(rWindow, 0.05), width * 0.45, 64]} />
              <meshStandardMaterial color="#d9cfae" side={THREE.DoubleSide} emissive={tone.window.emissive} emissiveIntensity={tone.window.emissiveIntensity} />
            </mesh>
            <mesh>
              <cylinderGeometry args={[Math.max(rWindow, 0.05), Math.max(rWindow, 0.05), windowHeight, 64, 1, true]} />
              <meshStandardMaterial color="#c8bc95" side={THREE.DoubleSide} emissive={tone.window.emissive} emissiveIntensity={tone.window.emissiveIntensity} />
            </mesh>
          </group>
        </Part>
      ) : (
        ghost('window') && <Ghost position={[0, t + windowHeight / 2, 0]} size={[width * 0.9, windowHeight, width * 0.9]} />
      )}
      {(hasWindow ? show('window') : ghost('window')) && (
        <Label position={[width * 0.45 - 0.4, t + windowHeight + 0.3, 0]} active={isOn('window')} onClick={() => onSelect('window')}>
          {hasWindow ? `cranial window r ${p.Wind_Radius} mm, d ${p.Wind_Depth} mm → NA ${(d?.na_eff ?? p.NA).toFixed(2)}` : 'no cranial window'}
        </Label>
      )}

      {/* Coverslip (absent at zero thickness) */}
      {hasCoverslip ? (
        <Part {...part('coverslip')}>
          <mesh position={[0, t / 2, 0]}>
            <boxGeometry args={[width, t, width * 0.7]} />
            <meshPhysicalMaterial
              color="#9fd7ff"
              transparent
              opacity={0.32 + tone.coverslip.boost}
              roughness={0.1}
              depthWrite={false}
              emissive={tone.coverslip.emissive}
              emissiveIntensity={tone.coverslip.emissiveIntensity}
            />
          </mesh>
        </Part>
      ) : (
        ghost('coverslip') && <Ghost position={[0, 0.4, 0]} size={[width, 0.8, width * 0.7]} />
      )}
      {(hasCoverslip ? show('coverslip') : ghost('coverslip')) && (
        <Label position={[width / 2 - 2.5, hasCoverslip ? t / 2 : 0.4, width * 0.35]} active={isOn('coverslip')} onClick={() => onSelect('coverslip')}>
          {hasCoverslip
            ? `coverslip ${p.Thickness.toFixed(0)} µm, n₂ = ${p.n2.toFixed(3)}${Math.abs(p.Tilt) > 0.05 ? `, tilt ${p.Tilt.toFixed(1)}°` : ''}`
            : 'no coverslip (thickness 0)'}
        </Label>
      )}

      {/* Everything attached to the objective tilts with the coverslip angle,
          pivoting about the focus so the beam still converges there. */}
      <group position={[0, yFocus, 0]} rotation={[0, 0, tilt]}>
        <group position={[0, -yFocus, 0]}>
          {/* Immersion medium */}
          <Part {...part('objective')}>
            <mesh position={[0, t + GAP / 2, 0]}>
              <cylinderGeometry args={[rBody, rBody * 0.9, GAP, 48]} />
              <meshStandardMaterial color={immersionColor} transparent opacity={0.1 + tone.objective.boost * 0.5} depthWrite={false} />
            </mesh>
          </Part>
          {show('objective') && (
            <Label position={[rBody + 0.3, t + GAP / 2, 0]} muted active={isOn('objective')} onClick={() => onSelect('objective')}>
              immersion n₁ = {p.n1.toFixed(3)}
            </Label>
          )}

          {/* Focusing cone */}
          <Frustum y0={yFocus} y1={0} r0={0} r1={r3} color={beamColor} opacity={0.55} />
          <Frustum y0={0} y1={t} r0={r3} r1={r2} color={beamColor} opacity={0.5} />
          <Frustum y0={t} y1={yLens} r0={r2} r1={rLens} color={beamColor} opacity={0.4} />
          <Frustum y0={yFocus - 0.7} y1={yFocus} r0={0.7 * tan(alpha3)} r1={0} color={beamColor} opacity={0.18} />
          {clipped && <Frustum y0={t} y1={yLens} r0={r2} r1={rLensFull} color={beamColor} opacity={0.15} wire />}

          {/* Objective */}
          <group position={[0, yLens, 0]}>
            <Part {...part('objective')}>
              <mesh position={[0, 0.02, 0]} scale={[1, 0.35, 1]}>
                <sphereGeometry args={[Math.max(rLens, 0.3), 48, 24, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2]} />
                <meshPhysicalMaterial color="#bfe6ff" roughness={0.05} metalness={0.1} transparent opacity={0.9} />
              </mesh>
              <mesh position={[0, BODY / 2, 0]}>
                <cylinderGeometry args={[rBody + 0.3, rBody, BODY, 48]} />
                <meshStandardMaterial color="#2b2b33" metalness={0.7} roughness={0.35} emissive={tone.objective.emissive} emissiveIntensity={tone.objective.emissiveIntensity} />
              </mesh>
              <mesh position={[0, 0.25, 0]}>
                <cylinderGeometry args={[rBody, rBody, 0.5, 48]} />
                <meshStandardMaterial color="#c9a227" metalness={0.9} roughness={0.3} emissive={tone.objective.emissive} emissiveIntensity={tone.objective.emissiveIntensity} />
              </mesh>
              <mesh position={[0, BODY * 0.55, 0]}>
                <cylinderGeometry args={[rBody + 0.42, rBody + 0.42, 0.35, 48]} />
                <meshStandardMaterial color="#8d8d99" metalness={0.8} roughness={0.3} emissive={tone.objective.emissive} emissiveIntensity={tone.objective.emissiveIntensity} />
              </mesh>
            </Part>
            {/* back pupil */}
            <Part {...part('pupil')}>
              <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, BODY + 0.01, 0]}>
                <circleGeometry args={[rPupil, 64]} />
                <meshBasicMaterial transparent toneMapped={false}>
                  <canvasTexture ref={pupilTex} attach="map" args={[pupilCanvas]} colorSpace={THREE.SRGBColorSpace} />
                </meshBasicMaterial>
              </mesh>
              {(tone.pupil.on || tone.pupil.hov) && (
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, BODY + 0.02, 0]}>
                  <ringGeometry args={[rPupil + 0.02, rPupil + 0.16, 64]} />
                  <meshBasicMaterial color={HI} transparent opacity={tone.pupil.on ? 0.9 : 0.5} />
                </mesh>
              )}
            </Part>
            {show('objective') && (
              <>
                <Label position={[rBody + 0.6, BODY * 0.55, 0]} muted active={isOn('objective')} onClick={() => onSelect('objective')}>
                  collar {p.Collar.toFixed(0)} µm
                </Label>
                <Label position={[-rBody - 0.6, BODY * 0.25, 0]} active={isOn('objective')} onClick={() => onSelect('objective')}>
                  objective NA {p.NA.toFixed(2)}, α = {((alpha * 180) / Math.PI).toFixed(1)}°
                </Label>
              </>
            )}
            {show('pupil') && (
              <Label position={[-rPupil - 0.5, BODY + 0.35, 0]} active={isOn('pupil')} onClick={() => onSelect('pupil')}>
                back pupil{zernikeSummary(p)}
              </Label>
            )}

            {/* Illumination optics stacked on the axis above the back pupil */}
            <group position={[0, BODY, 0]}>
              {/* the collimated beam coming down from the laser */}
              <mesh position={[0, STACK.laser / 2, 0]}>
                <cylinderGeometry args={[rBeam, rBeam, STACK.laser, 40, 1, true]} />
                <meshBasicMaterial color={beamColor} transparent opacity={0.2} side={THREE.DoubleSide} depthWrite={false} />
              </mesh>
              {/* rail holding the optics */}
              <mesh position={[rail, (STACK.laser + LASER_LENGTH) / 2, 0]}>
                <boxGeometry args={[0.18, STACK.laser + LASER_LENGTH, 0.18]} />
                <meshStandardMaterial color={METAL} metalness={0.7} roughness={0.35} />
              </mesh>

              {/* SLM: a square panel, only when one is in the beam path */}
              {!hasSlm && ghost('slm') && <Ghost position={[0, STACK.slm, 0]} size={[2 * rBeam + 0.8, 0.14, 2 * rBeam + 0.8]} />}
              {hasSlm && (
              <Part {...part('slm')}>
                <group position={[0, STACK.slm, 0]}>
                  <mesh>
                    <boxGeometry args={[2 * rBeam + 0.8, 0.14, 2 * rBeam + 0.8]} />
                    <meshStandardMaterial color={METAL_DARK} metalness={0.6} roughness={0.5} emissive={tone.slm.emissive} emissiveIntensity={tone.slm.emissiveIntensity} />
                  </mesh>
                  {/* the pattern, on the face towards the laser */}
                  <mesh position={[0, 0.075, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                    <planeGeometry args={[2 * rBeam + 0.5, 2 * rBeam + 0.5]} />
                    <meshBasicMaterial toneMapped={false}>
                      <canvasTexture ref={slmTex} attach="map" args={[slmCanvas]} colorSpace={THREE.SRGBColorSpace} />
                    </meshBasicMaterial>
                  </mesh>
                  <mesh position={[(rail + rBeam + 0.4) / 2, 0, 0]}>
                    <boxGeometry args={[Math.max(rail - rBeam - 0.4, 0.1), 0.1, 0.16]} />
                    <meshStandardMaterial color={METAL} metalness={0.7} roughness={0.35} />
                  </mesh>
                  {(tone.slm.on || tone.slm.hov) && (
                    <mesh position={[0, 0.09, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                      <ringGeometry args={[rBeam + 0.42, rBeam + 0.56, 4, 1, Math.PI / 4]} />
                      <meshBasicMaterial color={HI} transparent opacity={tone.slm.on ? 0.9 : 0.5} side={THREE.DoubleSide} />
                    </mesh>
                  )}
                </group>
              </Part>
              )}

              {/* Phase plate, only when a mask is selected */}
              {!hasPlate && ghost('phaseplate') && <Ghost position={[0, STACK.phaseplate, 0]} size={[2 * rBeam + 0.4, 0.08, 2 * rBeam + 0.4]} />}
              {hasPlate && (
                <Part {...part('phaseplate')}>
                  <StackDisc y={STACK.phaseplate} radius={rBeam + 0.1} rail={rail} canvas={plateCanvas} texRef={plateTex} tone={tone.phaseplate} />
                </Part>
              )}

              {/* Polarization optics */}
              <Part {...part('polarization')}>
                <StackDisc y={STACK.polarization} radius={rBeam + 0.1} rail={rail} canvas={waveCanvas} texRef={waveTex} tone={tone.polarization} />
              </Part>

              {/* Laser, pointing down */}
              <Part {...part('laser')}>
                <group position={[0, STACK.laser, 0]}>
                  <mesh position={[0, LASER_LENGTH / 2, 0]}>
                    <cylinderGeometry args={[rBeam + 0.3, rBeam + 0.3, LASER_LENGTH, 40]} />
                    <meshStandardMaterial color={METAL} metalness={0.75} roughness={0.3} emissive={tone.laser.emissive} emissiveIntensity={tone.laser.emissiveIntensity} />
                  </mesh>
                  <mesh position={[0, -0.005, 0]} rotation={[Math.PI / 2, 0, 0]}>
                    <circleGeometry args={[rBeam, 40]} />
                    <meshBasicMaterial color={beamColor} toneMapped={false} />
                  </mesh>
                  <mesh position={[(rail + rBeam + 0.3) / 2, LASER_LENGTH / 2, 0]}>
                    <boxGeometry args={[Math.max(rail - rBeam - 0.3, 0.1), 0.3, 0.3]} />
                    <meshStandardMaterial color={METAL} metalness={0.7} roughness={0.35} />
                  </mesh>
                </group>
              </Part>

              {(hasSlm ? show('slm') : ghost('slm')) && (
                <Label position={[-rBeam - 0.6, STACK.slm, 0]} active={isOn('slm')} onClick={() => onSelect('slm')}>
                  {p.SLM ? `SLM ${p.SLM.n}² px, ${p.SLM.levels > 1 ? `${p.SLM.levels} levels` : 'continuous'}${slmZernike ? ' + zernike' : ''}` : 'no SLM (enable it here)'}
                </Label>
              )}
              {(hasPlate ? show('phaseplate') : ghost('phaseplate')) && (
                <Label position={[-rBeam - 0.6, STACK.phaseplate, 0]} active={isOn('phaseplate')} onClick={() => onSelect('phaseplate')}>
                  {hasPlate ? `phase plate: ${modeLabel(p.Mode)}` : 'no phase plate'}
                </Label>
              )}
              {show('polarization') && (
                <Label position={[-rBeam - 0.6, STACK.polarization, 0]} active={isOn('polarization')} onClick={() => onSelect('polarization')}>
                  {polarizationLabel(p)}
                </Label>
              )}
              {show('laser') && (
                <Label position={[-rBeam - 0.7, STACK.laser + LASER_LENGTH / 2, 0]} active={isOn('laser')} onClick={() => onSelect('laser')}>
                  laser λ = {(p.Wavelength * 1000).toFixed(0)} nm
                </Label>
              )}
            </group>
          </group>
        </group>
      </group>

      {/* Focus */}
      <Part {...part('focus')}>
        <mesh position={[0, yFocus, 0]}>
          <sphereGeometry args={[0.13, 24, 24]} />
          <meshStandardMaterial color={beamColor} emissive={beamColor} emissiveIntensity={1.5} />
        </mesh>
        {/* the observation volume, to (compressed) scale */}
        <mesh position={[0, yFocus, 0]}>
          <boxGeometry args={[(2 * p.L_obs_XY) / 4, (2 * p.L_obs_Z) / 4, (2 * p.L_obs_XY) / 4]} />
          <meshBasicMaterial color={tone.focus.on || tone.focus.hov ? HI : '#ffffff'} wireframe transparent opacity={tone.focus.on ? 0.9 : 0.35} />
        </mesh>
        <mesh position={[0, yFocus, 0]}>
          <sphereGeometry args={[0.45, 16, 16]} />
          <meshBasicMaterial transparent opacity={tone.focus.hov ? 0.12 : 0.001} color={HI} depthWrite={false} />
        </mesh>
      </Part>
      {Math.abs(dfoc) > 0.002 && (
        <>
          <mesh position={[0, yNominal, 0]}>
            <sphereGeometry args={[0.1, 16, 16]} />
            <meshBasicMaterial color="#ffffff" wireframe />
          </mesh>
          <Line
            points={[
              [0, yNominal, 0],
              [0, yFocus, 0],
            ]}
            color="#ffffff"
            dashed
            dashSize={0.08}
            gapSize={0.06}
            lineWidth={1}
          />
        </>
      )}
      {show('focus') && (
        <Label position={[1.6, yFocus - 0.6, 0]} active={isOn('focus')} onClick={() => onSelect('focus')}>
          focus {p.Depth.toFixed(0)} µm deep
          {d && Math.abs(d.dfoc) > 0.005 ? ` (shifted ${d.dfoc > 0 ? '+' : ''}${d.dfoc.toFixed(2)} µm)` : ''}, volume {(2 * p.L_obs_XY).toFixed(1)} × {(2 * p.L_obs_Z).toFixed(1)} µm
        </Label>
      )}

      {/* Optical axis */}
      <Line
        points={[
          [0, -below, 0],
          [0, yTop, 0],
        ]}
        color="#8a8a96"
        transparent
        opacity={0.35}
        lineWidth={1}
      />
    </>
  );
}

/** Wire box marking where an absent optional element would sit. */
function Ghost({ position, size }: { position: [number, number, number]; size: [number, number, number] }) {
  const geometry = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(...size)), [size[0], size[1], size[2]]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <lineSegments position={position} geometry={geometry}>
      <lineBasicMaterial color={HI} transparent opacity={0.8} />
    </lineSegments>
  );
}

function modeLabel(m: Params['Mode']): string {
  switch (m) {
    case 'GAUSSIAN':
      return 'none';
    case 'DONUT':
      return 'vortex';
    case 'BOTTLE':
      return 'π disc';
    case 'DONUT BOTTLE':
      return 'donut + bottle';
    case 'LOADED':
      return 'on the SLM';
  }
}

function polarizationLabel(p: Params): string {
  if (p.Polarization === 2) return 'radial polarization';
  if (p.Polarization === 3) return 'azimuthal polarization';
  const e = Math.abs(p.Epsilon);
  const kind = e < 2 ? 'linear' : e > 43 ? 'circular' : 'elliptical';
  return `${kind} polarization${kind === 'linear' ? `, ψ = ${p.Psi.toFixed(0)}°` : ''}`;
}

function zernikeSummary(p: Params): string {
  const active = (
    [
      ['a4', 'defocus'],
      ['a12', 'spherical'],
      ['a24', 'spherical 2'],
      ['a3', 'astig'],
      ['a5', 'astig'],
      ['a7', 'coma'],
      ['a8', 'coma'],
      ['a6', 'trefoil'],
      ['a9', 'trefoil'],
      ['a1', 'tilt'],
      ['a2', 'tilt'],
    ] as const
  ).filter(([k]) => Math.abs(p[k]) > 1e-6);
  if (active.length === 0) return ', no aberrations';
  const names = Array.from(new Set(active.map(([, n]) => n)));
  return `: ${names.slice(0, 3).join(', ')}${names.length > 3 ? ', …' : ''}`;
}

/**
 * Orbit controls with the left button panning (only along the optical axis:
 * the target is pinned to x = z = 0), shift + left or the right button
 * rotating, and the wheel zooming.
 */
function AxisControls({ centerY }: { centerY: number }) {
  const ref = useRef<ComponentRef<typeof OrbitControls>>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    c.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };
    c.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_ROTATE };
    const down = (e: KeyboardEvent) => {
      if (e.key === 'Shift') c.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
    };
    const up = (e: KeyboardEvent) => {
      if (e.key === 'Shift') c.mouseButtons.LEFT = THREE.MOUSE.PAN;
    };
    const blur = () => {
      c.mouseButtons.LEFT = THREE.MOUSE.PAN;
    };
    // keep the target (and so the view) on the optical axis: undo any
    // sideways component of a pan on both the target and the camera
    const onAxis = () => {
      const dx = c.target.x;
      const dz = c.target.z;
      if (dx !== 0 || dz !== 0) {
        c.target.x = 0;
        c.target.z = 0;
        c.object.position.x -= dx;
        c.object.position.z -= dz;
      }
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    window.addEventListener('blur', blur);
    c.addEventListener('change', onAxis);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
      window.removeEventListener('blur', blur);
      c.removeEventListener('change', onAxis);
    };
  }, []);
  return (
    <OrbitControls
      ref={ref}
      makeDefault
      target={[0, centerY, 0]}
      enableDamping
      dampingFactor={0.12}
      minDistance={4}
      maxDistance={150}
      maxPolarAngle={Math.PI * 0.85}
      screenSpacePanning
    />
  );
}

/** Projects the focus to canvas pixels every frame and reports it when it moves. */
function FocusProbe({ y, onScreen }: { y: number; onScreen: (pt: { x: number; y: number } | null) => void }) {
  const { camera, size } = useThree();
  const last = useRef<{ x: number; y: number } | null>(null);
  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => {
    v.set(0, y, 0).project(camera);
    const inFront = v.z < 1;
    const pt = inFront ? { x: Math.round(((v.x + 1) / 2) * size.width), y: Math.round(((1 - v.y) / 2) * size.height) } : null;
    const l = last.current;
    if ((pt === null) !== (l === null) || (pt && l && (pt.x !== l.x || pt.y !== l.y))) {
      last.current = pt;
      onScreen(pt);
    }
  });
  return null;
}

/** Places the camera so the whole schematic fits the canvas; re-runs when the extent changes or on reset. */
function Fit({ halfX, halfY, centerX, centerY, resetKey }: { halfX: number; halfY: number; centerX: number; centerY: number; resetKey: number }) {
  const { camera, size, controls } = useThree();
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    const aspect = Math.max(size.width / Math.max(size.height, 1), 0.5);
    const tanHalf = Math.tan(((cam.fov ?? 35) * Math.PI) / 360);
    // an oblique view foreshortens x and adds depth, hence the margins
    const dist = Math.max(halfY / tanHalf, (halfX * 0.8) / (tanHalf * aspect)) * 1.08 + 2;
    const dir = new THREE.Vector3(0.55, 0.3, 0.78).normalize();
    camera.position.copy(dir.multiplyScalar(dist)).add(new THREE.Vector3(centerX, centerY, 0));
    camera.lookAt(centerX, centerY, 0);
    camera.updateProjectionMatrix();
    const c = controls as ComponentRef<typeof OrbitControls> | null;
    if (c) {
      c.target.set(centerX, centerY, 0);
      c.update();
    }
  }, [camera, controls, size.width, size.height, halfX, halfY, centerX, centerY, resetKey]);
  return null;
}

export function MicroscopeScene(props: MicroscopeProps) {
  const [resetKey, setResetKey] = useState(0);
  return (
    <div className="relative h-full w-full">
      <Canvas
        gl={{ alpha: true, antialias: true }}
        camera={{ fov: 35, near: 0.1, far: 400 }}
        dpr={[1, 2]}
        onPointerMissed={() => props.onSelect(null)}
      >
        <Scene {...props} resetKey={resetKey} />
      </Canvas>
      <div className="pointer-events-none absolute bottom-2 left-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
        <span>
          <kbd className="pg-kbd">drag</kbd> move
        </span>
        <span>
          <kbd className="pg-kbd">shift</kbd> + <kbd className="pg-kbd">drag</kbd> rotate
        </span>
        <span>
          <kbd className="pg-kbd">scroll</kbd> zoom
        </span>
        <span>click a part to edit it</span>
        <button type="button" className="pointer-events-auto rounded-md border bg-background/80 px-1.5 py-0.5 hover:text-foreground" onClick={() => setResetKey((k) => k + 1)}>
          reset view
        </button>
      </div>
    </div>
  );
}
