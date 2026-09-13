/**
 * Named scenes saved in the browser (localStorage): the parameters, the SLM
 * design, the imaging settings and the grid mode, so a setup can be picked
 * up later. Typed arrays are stored as plain arrays.
 */
import type { Params } from './params';
import type { SlmDesign } from './slm';
import type { ImagingSettings } from './use-imaging';

const KEY = 'faser-playground-scenes';

export interface SavedScene {
  id: string;
  name: string;
  savedAt: string; // ISO
  params: Params;
  design: Omit<SlmDesign, 'image' | 'custom'> & { image: number[] | null; custom: number[] | null };
  imaging: Omit<ImagingSettings, 'image'> & { image: number[] | null };
  autoGridOn: boolean;
}

export interface SceneState {
  params: Params;
  design: SlmDesign;
  imaging: ImagingSettings;
  autoGridOn: boolean;
}

function read(): SavedScene[] {
  try {
    const raw = localStorage.getItem(KEY);
    const list = raw ? (JSON.parse(raw) as unknown) : [];
    return Array.isArray(list) ? (list as SavedScene[]) : [];
  } catch {
    return [];
  }
}

function write(list: SavedScene[]): string | null {
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
    return null;
  } catch (e) {
    return e instanceof Error && e.name === 'QuotaExceededError'
      ? 'The browser storage is full; delete a saved scene or use a smaller SLM / image.'
      : `could not save: ${e instanceof Error ? e.message : String(e)}`;
  }
}

export function listScenes(): SavedScene[] {
  return read().sort((a, b) => (a.savedAt < b.savedAt ? 1 : -1));
}

export function saveScene(name: string, state: SceneState): { scene?: SavedScene; error?: string } {
  const clean = name.trim();
  if (!clean) return { error: 'give the scene a name' };
  const scene: SavedScene = {
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    name: clean,
    savedAt: new Date().toISOString(),
    params: state.params,
    design: { ...state.design, image: state.design.image ? Array.from(state.design.image) : null, custom: state.design.custom ? Array.from(state.design.custom) : null },
    imaging: { ...state.imaging, image: state.imaging.image ? Array.from(state.imaging.image) : null },
    autoGridOn: state.autoGridOn,
  };
  // a new save under an existing name replaces it
  const list = read().filter((s) => s.name !== clean);
  list.push(scene);
  const error = write(list);
  return error ? { error } : { scene };
}

export function deleteScene(id: string): void {
  write(read().filter((s) => s.id !== id));
}

/** Back to live state (typed arrays restored). */
export function sceneState(s: SavedScene): SceneState {
  return {
    params: s.params,
    design: { ...s.design, image: s.design.image ? Float32Array.from(s.design.image) : null, custom: s.design.custom ? Float32Array.from(s.design.custom) : null },
    imaging: { ...s.imaging, image: s.imaging.image ? Float32Array.from(s.imaging.image) : null },
    autoGridOn: s.autoGridOn,
  };
}
