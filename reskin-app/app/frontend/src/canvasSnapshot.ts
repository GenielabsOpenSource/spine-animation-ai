/* Module-level handle to the live PixiJS Application + Spine instance so
 * other components can snapshot the canvas + read slot bounds.
 */
import type { Application } from 'pixi.js';
import type { Spine } from '@esotericsoftware/spine-pixi-v8';

let currentApp: Application | null = null;
let currentSpine: Spine | null = null;

export function registerCanvas(app: Application | null) {
  currentApp = app;
  (window as any).__pixiApp = app;
}

export function registerSpine(spine: Spine | null) {
  currentSpine = spine;
  (window as any).__spine = spine;
}

export function getCurrentSpine(): Spine | null {
  return currentSpine;
}

/** Snapshot the live canvas as a PNG Blob.
 *
 * Uses Pixi's `renderer.extract` (works for WebGL/WebGPU even without
 * preserveDrawingBuffer) and tightly crops to the rendered content if there's
 * a single child container — that gives us a clean PNG of just the character,
 * which is what the AI pipeline wants.
 */
export async function snapshotCanvas(): Promise<Blob | null> {
  const app = currentApp;
  if (!app) return null;
  // Prefer extracting just the first child of the stage (the character),
  // not the whole stage including its dark background. If the stage has
  // multiple children, fall back to the entire stage.
  const target = app.stage.children.length === 1 ? app.stage.children[0] : app.stage;
  const canvas = (await app.renderer.extract.canvas(target)) as HTMLCanvasElement;
  return new Promise((resolve) => {
    canvas.toBlob((b) => resolve(b), 'image/png');
  });
}
