'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Html, OrbitControls } from '@react-three/drei';
import { colormapBytes, type ColormapName } from './colormaps';
import type { Volume } from './volume';

export interface RenderSettings {
  mode: 'mip' | 'composite';
  colormap: ColormapName;
  log: boolean;
  /** Decades shown below 1 in log mode. */
  logDecades: number;
  gamma: number;
  threshold: number;
  opacity: number;
  steps: number;
  showBox: boolean;
  /** Axis-length labels on the box (off in compact views). */
  boxLabels?: boolean;
}

export const DEFAULT_RENDER: RenderSettings = {
  mode: 'mip',
  colormap: 'inferno',
  log: false,
  logDecades: 4,
  gamma: 1,
  threshold: 0.02,
  opacity: 0.6,
  steps: 256,
  showBox: true,
};

const vertexShader = /* glsl */ `
uniform mat4 uInvModel;
out vec3 vOrigin;
out vec3 vDirection;
void main() {
  vOrigin = (uInvModel * vec4(cameraPosition, 1.0)).xyz;
  vDirection = position - vOrigin;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const fragmentShader = /* glsl */ `
precision highp float;
precision highp sampler3D;
uniform sampler3D uVolume;
uniform sampler2D uColormap;
uniform int uMode;
uniform int uLog;
uniform float uLogMin;
uniform float uSteps;
uniform float uThreshold;
uniform float uGamma;
uniform float uOpacity;
in vec3 vOrigin;
in vec3 vDirection;
out vec4 fragColor;

vec2 hitBox(vec3 orig, vec3 dir) {
  vec3 invDir = 1.0 / dir;
  vec3 tA = (vec3(-0.5) - orig) * invDir;
  vec3 tB = (vec3(0.5) - orig) * invDir;
  vec3 tMin = min(tA, tB);
  vec3 tMax = max(tA, tB);
  return vec2(max(tMin.x, max(tMin.y, tMin.z)), min(tMax.x, min(tMax.y, tMax.z)));
}

float mapValue(float v) {
  if (uLog == 1) {
    return clamp((log(max(v, uLogMin)) - log(uLogMin)) / (-log(uLogMin)), 0.0, 1.0);
  }
  return pow(clamp(v, 0.0, 1.0), uGamma);
}

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  vec3 rayDir = normalize(vDirection);
  vec2 bounds = hitBox(vOrigin, rayDir);
  if (bounds.x > bounds.y) discard;
  bounds.x = max(bounds.x, 0.0);
  float delta = 1.7320508 / uSteps;
  float t = bounds.x + delta * hash(gl_FragCoord.xy);
  vec4 acc = vec4(0.0);
  float m = 0.0;
  for (int i = 0; i < 2048; i++) {
    if (t >= bounds.y) break;
    vec3 p = vOrigin + t * rayDir + 0.5;
    float v = mapValue(texture(uVolume, p).r);
    if (uMode == 0) {
      m = max(m, v);
    } else {
      float a = clamp((v - uThreshold) / max(1.0 - uThreshold, 1e-3), 0.0, 1.0);
      a = 1.0 - exp(-a * uOpacity * delta * 60.0);
      vec3 c = texture(uColormap, vec2(v, 0.5)).rgb;
      acc.rgb += (1.0 - acc.a) * a * c;
      acc.a += (1.0 - acc.a) * a;
      if (acc.a > 0.98) break;
    }
    t += delta;
  }
  if (uMode == 0) {
    if (m <= uThreshold) discard;
    vec3 c = texture(uColormap, vec2(m, 0.5)).rgb;
    fragColor = vec4(c, smoothstep(uThreshold, uThreshold + 0.08, m));
  } else {
    if (acc.a < 0.005) discard;
    fragColor = vec4(acc.rgb / max(acc.a, 1e-4), acc.a);
  }
}
`;

function makeVolumeTexture(r: Volume) {
  // Half floats keep the low-intensity side lobes (needed for the log view)
  // and are filterable in every WebGL2 implementation.
  // Normalized to the volume maximum so the transfer function works in [0, 1].
  const inv = r.max > 0 ? 1 / r.max : 1;
  const half = new Uint16Array(r.data.length);
  for (let i = 0; i < r.data.length; i++) half[i] = THREE.DataUtils.toHalfFloat(r.data[i] * inv);
  const tex = new THREE.Data3DTexture(half, r.nx, r.ny, r.nz);
  tex.format = THREE.RedFormat;
  tex.type = THREE.HalfFloatType;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

function makeColormapTexture(name: ColormapName) {
  const tex = new THREE.DataTexture(colormapBytes(name), 256, 1, THREE.RGBAFormat);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

/** The ray-marched volume, a unit cube (longest side 1) with local z as the optical axis, rotated so z points up. */
export function VolumeMesh({ result, settings }: { result: Volume; settings: RenderSettings }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  // Created once; the material owns this object, effects update it through the ref.
  const [uniforms] = useState(() => ({
    uVolume: { value: null as THREE.Data3DTexture | null },
    uColormap: { value: null as THREE.DataTexture | null },
    uInvModel: { value: new THREE.Matrix4() },
    uMode: { value: 0 },
    uLog: { value: 0 },
    uLogMin: { value: 1e-4 },
    uSteps: { value: 256 },
    uThreshold: { value: 0.02 },
    uGamma: { value: 1 },
    uOpacity: { value: 0.6 },
  }));

  useEffect(() => {
    const mat = materialRef.current;
    if (!mat) return;
    const tex = makeVolumeTexture(result);
    mat.uniforms.uVolume.value = tex;
    return () => tex.dispose();
  }, [result]);

  useEffect(() => {
    const mat = materialRef.current;
    if (!mat) return;
    const tex = makeColormapTexture(settings.colormap);
    mat.uniforms.uColormap.value = tex;
    return () => tex.dispose();
  }, [settings.colormap]);

  useEffect(() => {
    const mat = materialRef.current;
    if (!mat) return;
    const u = mat.uniforms;
    u.uMode.value = settings.mode === 'mip' ? 0 : 1;
    u.uLog.value = settings.log ? 1 : 0;
    u.uLogMin.value = Math.pow(10, -settings.logDecades);
    u.uSteps.value = settings.steps;
    u.uThreshold.value = settings.threshold;
    u.uGamma.value = settings.gamma;
    u.uOpacity.value = settings.opacity;
  }, [settings]);

  useFrame(() => {
    const mesh = meshRef.current;
    const mat = materialRef.current;
    if (mesh && mat) {
      mesh.updateMatrixWorld();
      (mat.uniforms.uInvModel.value as THREE.Matrix4).copy(mesh.matrixWorld).invert();
    }
  });

  // Local x, y are lateral; local z is the optical axis, rotated to point up.
  const { sizeX: sx, sizeY: sy, sizeZ: sz } = result;
  const norm = Math.max(sx, sy, sz);
  const scale: [number, number, number] = [sx / norm, sy / norm, sz / norm];

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      <mesh ref={meshRef} scale={scale}>
        <boxGeometry args={[1, 1, 1]} />
        <shaderMaterial
          ref={materialRef}
          glslVersion={THREE.GLSL3}
          vertexShader={vertexShader}
          fragmentShader={fragmentShader}
          uniforms={uniforms}
          side={THREE.BackSide}
          transparent
          depthWrite={false}
        />
      </mesh>
      {settings.showBox && (
        <group scale={scale}>
          <lineSegments>
            <edgesGeometry args={[new THREE.BoxGeometry(1, 1, 1)]} />
            <lineBasicMaterial color="#8a8a96" transparent opacity={0.5} />
          </lineSegments>
          {settings.boxLabels !== false && (
            <>
              <Html position={[0.5, -0.5, -0.5]} center className="pg-label" zIndexRange={[10, 0]}>
                x {sx.toFixed(2)} µm
              </Html>
              <Html position={[-0.5, 0.5, -0.5]} center className="pg-label" zIndexRange={[10, 0]}>
                y {sy.toFixed(2)} µm
              </Html>
              <Html position={[-0.5, -0.5, 0.5]} center className="pg-label" zIndexRange={[10, 0]}>
                z {sz.toFixed(2)} µm
              </Html>
            </>
          )}
        </group>
      )}
    </group>
  );
}

function Fit() {
  const { camera } = useThree();
  useEffect(() => {
    camera.position.set(1.6, 1.1, 1.6);
    camera.lookAt(0, 0, 0);
  }, [camera]);
  return null;
}

export function VolumeViewer({ volume, settings }: { volume: Volume | null; settings: RenderSettings }) {
  return (
    <div className="relative h-full w-full">
      <Canvas
        gl={{ alpha: true, antialias: false, powerPreference: 'high-performance' }}
        camera={{ fov: 40, near: 0.05, far: 50 }}
        dpr={[1, 2]}
      >
        <Fit />
        {volume && <VolumeMesh result={volume} settings={settings} />}
        <OrbitControls makeDefault enableDamping dampingFactor={0.12} minDistance={0.8} maxDistance={8} />
      </Canvas>
      {!volume && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-white/60">
          No volume yet
        </div>
      )}
    </div>
  );
}
