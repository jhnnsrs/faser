'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { Canvas, useThree } from '@react-three/fiber';
import { Html, Line, OrbitControls } from '@react-three/drei';
import type { Derived, Params } from './params';
import { drawPupil, wavelengthToRgb } from './pupil';

/**
 * An idealized microscope drawn from the current parameters. Not to scale
 * (the focal length is thousands of times the coverslip thickness); instead:
 *   1 scene unit = 100 µm for the coverslip, the imaging depth and the focus
 *   shift, while the immersion gap and the objective have fixed schematic
 *   sizes. Angles are real: the cone half-angles are alpha (immersion),
 *   alpha2 (coverslip) and alpha3 (sample) from the simulator, so the front
 *   aperture visibly widens with NA, and index mismatch bends the cone.
 *
 * Frame: y is the optical axis pointing up (towards the objective); y = 0 is
 * the coverslip/sample interface. The output volume of the simulation follows
 * the coverslip frame, so a coverslip tilt is drawn as the objective (and its
 * beam) being tilted.
 */

const UNIT = 200; // µm per scene unit
const R_LENS = 2.6; // drawn radius of the front lens (units)
const MAX_DEPTH = 3; // imaging depth is compressed smoothly to at most this many units
const BODY = 2.6; // objective body height (units)
const BELOW = 4; // sample block depth (units)

export interface MicroscopeProps {
  params: Params;
  derived: Derived | null;
  showLabels: boolean;
}

function Label({ position, children, muted }: { position: [number, number, number]; children: React.ReactNode; muted?: boolean }) {
  return (
    <Html position={position} center className={muted ? 'pg-label pg-label-muted' : 'pg-label'} zIndexRange={[10, 0]}>
      {children}
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
      <meshBasicMaterial
        color={color}
        transparent
        opacity={opacity}
        side={THREE.DoubleSide}
        depthWrite={false}
        wireframe={wire}
      />
    </mesh>
  );
}

function Scene({ params: p, derived: d, showLabels }: MicroscopeProps) {
  const beamColor = useMemo(() => {
    const [r, g, b] = wavelengthToRgb(p.Wavelength * 1000);
    return new THREE.Color(r, g, b);
  }, [p.Wavelength]);

  // Geometry (scene units)
  const t = Math.max(p.Thickness / UNIT, 0.04);
  // Depth (and the focus shift) are compressed with tanh so a 1 mm deep focus
  // still fits next to a 170 µm coverslip; small depths stay linear.
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
  // Supercritical cones (alpha3 -> 90°, evanescent) are drawn at 75° at most.
  const tan = (a: number) => Math.tan(Math.min(a, 1.31));

  // Marginal-ray radii at each interface
  const r3 = (0 - yFocus) * tan(alpha3); // at the coverslip bottom
  const r2 = r3 + t * tan(alpha2); // at the coverslip top
  // Working distance: the objective sits where a front lens of fixed drawn
  // radius just captures the cone, so it is short at high NA and long at low
  // NA, like real objectives.
  const GAP = Math.min(5, Math.max(0.6, (R_LENS - r2) / Math.max(tan(alphaEff), 0.05)));
  const rLens = r2 + GAP * tan(alphaEff); // at the front lens
  // Unclipped cone (cranial window only) for the faint outline
  const rLensFull = r2 + GAP * tan(alpha);
  const clipped = alphaEff < alpha - 1e-6;

  const yLens = t + GAP;
  const rBody = Math.max(rLens, R_LENS) + 0.3;
  const rPupil = rBody * 0.8;

  // Pupil texture: drawn on a canvas whenever the parameters change.
  const [pupilCanvas] = useState(() => document.createElement('canvas'));
  const pupilTexRef = useRef<THREE.CanvasTexture>(null);
  useEffect(() => {
    if (!d) return;
    drawPupil(pupilCanvas, p, d, 256);
    if (pupilTexRef.current) pupilTexRef.current.needsUpdate = true;
  }, [p, d, pupilCanvas]);

  const tilt = ((p.Tilt ?? 0) * Math.PI) / 180;
  const width = Math.max(11, 2 * rBody + 4);

  // Cranial window (skull) slab: sits on the coverslip, its aperture grazes
  // the clipped cone so the drawing is self-consistent.
  const windowHeight = 1.5;
  const rWindow = (t + windowHeight - yFocus) * tan(alphaEff);

  const immersionColor = p.n1 > 1.45 ? '#e8c46a' : p.n1 > 1.1 ? '#6fb7ff' : '#ffffff';

  // Scene extent and vertical centre, quantized so the camera only re-fits on big changes.
  const extent = Math.ceil(Math.max(yLens + BODY + 1.5, below + 1, width / 2 + 1) / 4) * 4;
  const centerY = Math.round(((yLens + BODY - below) / 2) * 2) / 2;

  return (
    <>
      <Fit extent={extent} centerY={centerY} />
      <OrbitControls
        makeDefault
        target={[0, centerY, 0]}
        enableDamping
        dampingFactor={0.12}
        minDistance={4}
        maxDistance={150}
        maxPolarAngle={Math.PI * 0.85}
      />
      <ambientLight intensity={0.9} />
      <directionalLight position={[6, 10, 4]} intensity={1.4} />
      <directionalLight position={[-6, 4, -6]} intensity={0.4} />

      {/* Sample */}
      <mesh position={[0, -below / 2, 0]}>
        <boxGeometry args={[width, below, width * 0.7]} />
        <meshStandardMaterial color="#e08a9a" transparent opacity={0.16} depthWrite={false} />
      </mesh>
      {showLabels && (
        <Label position={[-width / 2 + 0.2, -below + 0.5, 0]} muted>
          sample n₃ = {p.n3.toFixed(3)}
        </Label>
      )}

      {/* Coverslip */}
      <mesh position={[0, t / 2, 0]}>
        <boxGeometry args={[width, t, width * 0.7]} />
        <meshPhysicalMaterial color="#9fd7ff" transparent opacity={0.32} roughness={0.1} depthWrite={false} />
      </mesh>
      {showLabels && (
        <Label position={[width / 2 - 2.5, t / 2, width * 0.35]}>
          coverslip {p.Thickness.toFixed(0)} µm, n₂ = {p.n2.toFixed(3)}
          {Math.abs(p.Tilt) > 0.05 ? `, tilt ${p.Tilt.toFixed(1)}°` : ''}
        </Label>
      )}

      {/* Cranial window (skull) */}
      {p.Window === 'CUSTOM' && (
        <group position={[0, t + windowHeight / 2, 0]}>
          <mesh>
            <cylinderGeometry args={[width * 0.45, width * 0.45, windowHeight, 64, 1, true]} />
            <meshStandardMaterial color="#d9cfae" side={THREE.BackSide} transparent opacity={0.6} />
          </mesh>
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, windowHeight / 2, 0]}>
            <ringGeometry args={[Math.max(rWindow, 0.05), width * 0.45, 64]} />
            <meshStandardMaterial color="#d9cfae" side={THREE.DoubleSide} />
          </mesh>
          <mesh>
            <cylinderGeometry args={[Math.max(rWindow, 0.05), Math.max(rWindow, 0.05), windowHeight, 64, 1, true]} />
            <meshStandardMaterial color="#c8bc95" side={THREE.DoubleSide} />
          </mesh>
          {showLabels && (
            <Label position={[width * 0.45 - 0.4, windowHeight / 2 + 0.3, 0]}>
              cranial window r = {p.Wind_Radius} mm, d = {p.Wind_Depth} mm → NA<sub>eff</sub> {(d?.na_eff ?? p.NA).toFixed(2)}
            </Label>
          )}
        </group>
      )}

      {/* Everything attached to the objective tilts with the coverslip angle,
          pivoting about the focus so the beam still converges there. */}
      <group position={[0, yFocus, 0]} rotation={[0, 0, tilt]}>
        <group position={[0, -yFocus, 0]}>
          {/* Immersion medium */}
          <mesh position={[0, t + GAP / 2, 0]}>
            <cylinderGeometry args={[rBody, rBody * 0.9, GAP, 48]} />
            <meshStandardMaterial color={immersionColor} transparent opacity={0.1} depthWrite={false} />
          </mesh>
          {showLabels && (
            <Label position={[rBody + 0.3, t + GAP / 2, 0]} muted>
              immersion n₁ = {p.n1.toFixed(3)}
            </Label>
          )}

          {/* Focusing cone: sample, coverslip, immersion, plus a short
              diverging tail beyond the focus. */}
          <Frustum y0={yFocus} y1={0} r0={0} r1={r3} color={beamColor} opacity={0.55} />
          <Frustum y0={0} y1={t} r0={r3} r1={r2} color={beamColor} opacity={0.5} />
          <Frustum y0={t} y1={yLens} r0={r2} r1={rLens} color={beamColor} opacity={0.4} />
          <Frustum y0={yFocus - 0.7} y1={yFocus} r0={0.7 * tan(alpha3)} r1={0} color={beamColor} opacity={0.18} />
          {clipped && <Frustum y0={t} y1={yLens} r0={r2} r1={rLensFull} color={beamColor} opacity={0.15} wire />}

          {/* Objective */}
          <group position={[0, yLens, 0]}>
            {/* front lens */}
            <mesh position={[0, 0.02, 0]} scale={[1, 0.35, 1]}>
              <sphereGeometry args={[Math.max(rLens, 0.3), 48, 24, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2]} />
              <meshPhysicalMaterial color="#bfe6ff" roughness={0.05} metalness={0.1} transparent opacity={0.9} />
            </mesh>
            {/* body */}
            <mesh position={[0, BODY / 2, 0]}>
              <cylinderGeometry args={[rBody + 0.3, rBody, BODY, 48]} />
              <meshStandardMaterial color="#2b2b33" metalness={0.7} roughness={0.35} />
            </mesh>
            <mesh position={[0, 0.25, 0]}>
              <cylinderGeometry args={[rBody, rBody, 0.5, 48]} />
              <meshStandardMaterial color="#c9a227" metalness={0.9} roughness={0.3} />
            </mesh>
            {/* correction collar ring */}
            <mesh position={[0, BODY * 0.55, 0]}>
              <cylinderGeometry args={[rBody + 0.42, rBody + 0.42, 0.35, 48]} />
              <meshStandardMaterial color="#8d8d99" metalness={0.8} roughness={0.3} />
            </mesh>
            {/* back pupil */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, BODY + 0.01, 0]}>
              <circleGeometry args={[rPupil, 64]} />
              <meshBasicMaterial transparent toneMapped={false}>
                <canvasTexture ref={pupilTexRef} attach="map" args={[pupilCanvas]} colorSpace={THREE.SRGBColorSpace} />
              </meshBasicMaterial>
            </mesh>
            {showLabels && (
              <>
                <Label position={[rBody + 0.6, BODY * 0.55, 0]} muted>
                  collar {p.Collar.toFixed(0)} µm
                </Label>
                <Label position={[0, BODY + 0.5, 0]}>
                  back pupil: amplitude · phase mask · aberrations, {p.Mode.toLowerCase()}
                </Label>
                <Label position={[-rBody - 0.6, BODY * 0.25, 0]}>
                  objective NA {p.NA.toFixed(2)}, α = {((alpha * 180) / Math.PI).toFixed(1)}°
                </Label>
                <Label position={[-rLens - 0.4, -GAP / 2, 0]} muted>
                  λ = {(p.Wavelength * 1000).toFixed(0)} nm
                </Label>
              </>
            )}
          </group>
        </group>
      </group>

      {/* Focus markers */}
      <mesh position={[0, yFocus, 0]}>
        <sphereGeometry args={[0.13, 24, 24]} />
        <meshStandardMaterial color={beamColor} emissive={beamColor} emissiveIntensity={1.5} />
      </mesh>
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
      {showLabels && (
        <Label position={[1.4, yFocus - 0.6, 0]}>
          focus {p.Depth.toFixed(0)} µm below the coverslip
          {d && Math.abs(d.dfoc) > 0.005 ? ` (shifted ${d.dfoc > 0 ? '+' : ''}${d.dfoc.toFixed(2)} µm)` : ''}
        </Label>
      )}

      {/* Optical axis */}
      <Line
        points={[
          [0, -below, 0],
          [0, yLens + BODY + 1, 0],
        ]}
        color="#8a8a96"
        transparent
        opacity={0.35}
        lineWidth={1}
      />
    </>
  );
}

/** Places the camera so the whole schematic fits; re-runs when the extent changes. */
function Fit({ extent, centerY }: { extent: number; centerY: number }) {
  const { camera } = useThree();
  useEffect(() => {
    const dir = new THREE.Vector3(0.7, 0.3, 0.64).normalize();
    camera.position.copy(dir.multiplyScalar(extent * 2.0)).add(new THREE.Vector3(0, centerY, 0));
    camera.lookAt(0, centerY, 0);
    camera.updateProjectionMatrix();
  }, [camera, extent, centerY]);
  return null;
}

export function MicroscopeScene(props: MicroscopeProps) {
  return (
    <Canvas gl={{ alpha: true, antialias: true }} camera={{ fov: 35, near: 0.1, far: 400 }} dpr={[1, 2]}>
      <Scene {...props} />
    </Canvas>
  );
}
