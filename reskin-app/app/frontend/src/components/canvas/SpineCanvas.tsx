import { useEffect, useRef, useState } from 'react';
import { Application, Assets } from 'pixi.js';
import { Spine } from '@esotericsoftware/spine-pixi-v8';
import { useStore } from '../../state/store';
import { api, type Project } from '../../api/client';
import { registerCanvas, registerSpine } from '../../canvasSnapshot';
import { CanvasTools } from './CanvasTools';

/* Renders the active skin's character via the official spine-pixi-v8 runtime.
 * Default skin loads the project's main Spine.json + atlas. Generated skins
 * load `Spine-{skin}.json` + `Spine-{skin}.atlas` and select the new skin.
 */
type SlotRect = { x: number; y: number; w: number; h: number };

type AttOriginal = { x: number; y: number; rotation: number; scaleX: number; scaleY: number };

function snapshotOriginalAtt(spine: Spine): Map<string, AttOriginal> {
  const out = new Map<string, AttOriginal>();
  for (const slot of spine.skeleton.slots) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const att = slot.attachment as any;
    if (att && typeof att.x === 'number' && typeof att.y === 'number') {
      out.set(slot.data.name, {
        x: att.x,
        y: att.y,
        rotation: typeof att.rotation === 'number' ? att.rotation : 0,
        scaleX: typeof att.scaleX === 'number' ? att.scaleX : 1,
        scaleY: typeof att.scaleY === 'number' ? att.scaleY : 1,
      });
    }
  }
  return out;
}

type Transform = { x?: number; y?: number; rotation?: number; scale?: number };

function applyTransforms(
  spine: Spine,
  originals: Map<string, AttOriginal>,
  transforms: Record<string, Transform>,
): void {
  for (const slot of spine.skeleton.slots) {
    const orig = originals.get(slot.data.name);
    if (!orig) continue;
    const t = transforms[slot.data.name] ?? {};
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const att = slot.attachment as any;
    if (!att || typeof att.x !== 'number') continue;
    att.x = orig.x + (t.x ?? 0);
    att.y = orig.y + (t.y ?? 0);
    att.rotation = orig.rotation + (t.rotation ?? 0);
    const f = t.scale ?? 1;
    att.scaleX = orig.scaleX * f;
    att.scaleY = orig.scaleY * f;
    if (typeof att.updateRegion === 'function') att.updateRegion();
  }
}

function computeAllSlotRects(
  spine: Spine,
): Array<{ name: string; rect: SlotRect; drawIndex: number }> {
  const slots = spine.skeleton.slots;
  const drawOrder = spine.skeleton.drawOrder;
  const drawIndexBySlotIndex = new Map<number, number>();
  drawOrder.forEach((s, i) => drawIndexBySlotIndex.set(s.data.index, i));

  const originals = new Map<number, unknown>();
  for (const s of slots) originals.set(s.data.index, s.attachment ?? null);

  const out: Array<{ name: string; rect: SlotRect; drawIndex: number }> = [];
  try {
    for (const target of slots) {
      const orig = originals.get(target.data.index);
      if (!orig) continue;
      for (const s of slots) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        s.setAttachment(s.data.index === target.data.index ? (orig as any) : null);
      }
      spine.update(0);
      const b = spine.getBounds();
      if (b.width > 0 && b.height > 0) {
        out.push({
          name: target.data.name,
          rect: { x: b.x, y: b.y, w: b.width, h: b.height },
          drawIndex: drawIndexBySlotIndex.get(target.data.index) ?? 0,
        });
      }
    }
  } finally {
    for (const s of slots) {
      const o = originals.get(s.data.index);
      if (o !== undefined) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        s.setAttachment(o as any);
      }
    }
    spine.update(0);
  }
  // Sort by drawIndex DESC so iteration picks the topmost slot first.
  out.sort((a, b) => b.drawIndex - a.drawIndex);
  return out;
}

export function SpineCanvas() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const appRef = useRef<Application | null>(null);
  const spineRef = useRef<Spine | null>(null);
  const defaultAttachmentsRef = useRef<Map<number, string | null>>(new Map());
  const slotRectsRef = useRef<Array<{ name: string; rect: SlotRect; drawIndex: number }>>([]);
  // Per-slot original (x, y) snapshot from the loaded JSON. Drag deltas are
  // applied on top of these so reset can return to disk-default.
  const originalAttachmentXYRef = useRef<Map<string, AttOriginal>>(new Map());
  // Drag state survives re-renders via a ref — `selectSlot()` inside onDown
  // triggers React updates that re-bind the pointer listeners; the new
  // listeners read the same ref so onMove/onUp still find the active drag.
  type DragState =
    | {
        kind: 'hand';
        slot: string;
        bone: { localToWorld: (v: { x: number; y: number }) => void; worldToLocal: (v: { x: number; y: number }) => void };
        startWorld: { x: number; y: number };
        startClient: { x: number; y: number };
      }
    | {
        kind: 'scale';
        slot: string;
        anchorClient: { x: number; y: number };  // part's screen-center
        startDist: number;
        startScaleX: number;
        startScaleY: number;
      }
    | {
        kind: 'rotate';
        slot: string;
        anchorClient: { x: number; y: number };
        startAngleRad: number;
        startRotationDeg: number;
      };
  const dragRef = useRef<DragState | null>(null);
  const [ready, setReady] = useState(false);
  const [dragging, setDragging] = useState(false);
  const project = useStore((s) => s.project);
  const activeSkin = useStore((s) => s.activeSkin);
  const activeAnimation = useStore((s) => s.activeAnimation);
  const hidden = useStore((s) => s.hidden);
  const assetVersion = useStore((s) => s.assetVersion);
  const selectSlot = useStore((s) => s.selectSlot);
  const tool = useStore((s) => s.tool);
  const refreshProjectSkins = useStore((s) => s.refreshProjectSkins);

  // Bring up Pixi once
  useEffect(() => {
    if (!containerRef.current) return;
    let cancelled = false;
    let appInstance: Application | null = null;

    (async () => {
      const app = new Application();
      try {
        await app.init({
          background: '#ffffff',
          resizeTo: containerRef.current!,
          antialias: true,
        });
      } catch (e) {
        console.error('Pixi init failed', e);
        return;
      }
      if (cancelled) {
        try { app.destroy(true); } catch { /* noop */ }
        return;
      }
      appInstance = app;
      appRef.current = app;
      registerCanvas(app);
      if (containerRef.current) containerRef.current.appendChild(app.canvas);
      setReady(true);
    })();

    return () => {
      cancelled = true;
      setReady(false);
      if (appInstance) {
        try { appInstance.destroy(true); } catch { /* noop */ }
      }
      appRef.current = null;
      registerCanvas(null);
    };
  }, []);

  useEffect(() => {
    if (!ready) return;
    const app = appRef.current;
    if (!app || !project) return;
    let cancelled = false;
    const stage = app.stage;
    stage.removeChildren();
    spineRef.current = null;
    registerSpine(null);
    defaultAttachmentsRef.current = new Map();
    slotRectsRef.current = [];
    originalAttachmentXYRef.current = new Map();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (window as any).__origMap = originalAttachmentXYRef.current;

    (async () => {
      // Generated skins: prefer the per-skin Spine JSON + atlas (rigged
      // character with new textures). If those don't exist yet, fall back to
      // showing the flat reskinned image.
      if (activeSkin && activeSkin !== 'default' && project.spine_json && project.atlas) {
        const skinJsonName = `${project.spine_json.replace(/\.json$/i, '')}-${activeSkin}.json`;
        const skinAtlasName = `${project.spine_json.replace(/\.json$/i, '')}-${activeSkin}.atlas`;
        const probe = await fetch(api.fileUrl(skinAtlasName));
        if (probe.ok) {
          try {
            const stamp = Date.now();
            const skelAlias = `${project.name}-${activeSkin}-skel-${stamp}`;
            const atlasAlias = `${project.name}-${activeSkin}-atlas-${stamp}`;
            Assets.add({ alias: skelAlias, src: `${api.fileUrl(skinJsonName)}?v=${stamp}` });
            // The backend rewrites the atlas's PNG line to include the same
            // ?v= query so URL resolution keeps the cache-buster on the PNG
            // (without it Pixi/the browser serves a stale texture after
            // inpaint/rebake — the atlas changes but the bare-filename PNG
            // reference doesn't).
            Assets.add({ alias: atlasAlias, src: `${api.fileUrl(skinAtlasName)}?v=${stamp}` });
            await Assets.load([skelAlias, atlasAlias]);
            if (cancelled || !appRef.current) return;
            const spine = Spine.from({ skeleton: skelAlias, atlas: atlasAlias });
            spineRef.current = spine;
            registerSpine(spine);
            if (spine.skeleton.data.findSkin(activeSkin)) {
              spine.skeleton.setSkinByName(activeSkin);
              spine.skeleton.setSlotsToSetupPose();
            }
            const initialAnim = activeAnimation && spine.skeleton.data.findAnimation(activeAnimation)
              ? activeAnimation
              : (spine.skeleton.data.animations[0]?.name ?? null);
            if (initialAnim) {
              spine.state.setAnimation(0, initialAnim, true);
            }
            const bounds = spine.getBounds();
            const padding = 40;
            const sw = app.canvas.width - padding * 2;
            const sh = app.canvas.height - padding * 2;
            const scale = Math.min(sw / Math.max(1, bounds.width), sh / Math.max(1, bounds.height));
            spine.scale.set(scale);
            spine.x = app.canvas.width / 2 - (bounds.x + bounds.width / 2) * scale;
            spine.y = app.canvas.height / 2 - (bounds.y + bounds.height / 2) * scale;
            stage.addChild(spine);
            originalAttachmentXYRef.current = snapshotOriginalAtt(spine);
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (window as any).__origMap = originalAttachmentXYRef.current;
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (window as any).__origMapSize = originalAttachmentXYRef.current.size;
            applyTransforms(
              spine,
              originalAttachmentXYRef.current,
              project.transforms?.[activeSkin] ?? {},
            );
            spine.update(0);
            slotRectsRef.current = computeAllSlotRects(spine);
            return;
          } catch (e) {
            console.warn('per-skin spine load failed', e);
          }
        }
      }

      // Default skin: render the project's main spine via spine-pixi
      if (project.spine_json && project.atlas) {
        try {
          const stamp = Date.now();
          // Cache-bust the URLs themselves so Pixi's Assets cache + browser
          // HTTP cache don't return data from a previously-open project.
          const skeletonUrl = `${api.fileUrl(project.spine_json)}?v=${stamp}`;
          const atlasUrl = `${api.fileUrl(project.atlas)}?v=${stamp}`;
          const skelAlias = `${project.name}-skel-${stamp}`;
          const atlasAlias = `${project.name}-atlas-${stamp}`;

          Assets.add({ alias: skelAlias, src: skeletonUrl });
          Assets.add({ alias: atlasAlias, src: atlasUrl });
          await Assets.load([skelAlias, atlasAlias]);
          if (cancelled || !appRef.current) return;

          const spine = Spine.from({ skeleton: skelAlias, atlas: atlasAlias });
          spineRef.current = spine;
          registerSpine(spine);
          // Force the default skin so the rig always renders the original
          // attachments (avoids a previously-active skin lingering).
          if (spine.skeleton.data.findSkin('default')) {
            spine.skeleton.setSkinByName('default');
            spine.skeleton.setSlotsToSetupPose();
          }
          // Snapshot each slot's default attachment name so we can restore it
          // when the user un-hides the slot.
          defaultAttachmentsRef.current = new Map();
          for (const slot of spine.skeleton.slots) {
            defaultAttachmentsRef.current.set(
              slot.data.index,
              slot.attachment?.name ?? null,
            );
          }
          // Set rest-pose / first animation if any
          if (spine.skeleton.data.animations.length > 0) {
            spine.state.setAnimation(0, spine.skeleton.data.animations[0].name, true);
          }
          // Switch to active skin if it isn't 'default'
          if (activeSkin && activeSkin !== 'default' &&
              spine.skeleton.data.findSkin(activeSkin)) {
            spine.skeleton.setSkinByName(activeSkin);
            spine.skeleton.setSlotsToSetupPose();
          }

          // Center & fit
          const bounds = spine.getBounds();
          const padding = 40;
          const sw = app.canvas.width - padding * 2;
          const sh = app.canvas.height - padding * 2;
          const scale = Math.min(sw / Math.max(1, bounds.width), sh / Math.max(1, bounds.height));
          spine.scale.set(scale);
          spine.x = app.canvas.width / 2 - (bounds.x + bounds.width / 2) * scale;
          spine.y = app.canvas.height / 2 - (bounds.y + bounds.height / 2) * scale;
          stage.addChild(spine);
          originalAttachmentXYRef.current = snapshotOriginalAtt(spine);
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (window as any).__origMap = originalAttachmentXYRef.current;
          applyTransforms(
            spine,
            originalAttachmentXYRef.current,
            project.transforms?.[activeSkin] ?? {},
          );
          spine.update(0);
          slotRectsRef.current = computeAllSlotRects(spine);
          return;
        } catch (e) {
          console.warn('spine-pixi load failed', e);
        }
      }
    })();

    return () => { cancelled = true; };
  }, [ready, project, activeSkin, assetVersion]);

  // React to transforms changes (e.g. after reset) without reloading assets
  useEffect(() => {
    const spine = spineRef.current;
    if (!spine) return;
    if (originalAttachmentXYRef.current.size === 0) return;
    applyTransforms(
      spine,
      originalAttachmentXYRef.current,
      project?.transforms?.[activeSkin] ?? {},
    );
    spine.update(0);
    slotRectsRef.current = computeAllSlotRects(spine);
  }, [project?.transforms, activeSkin]);

  // React to animation dropdown changes (without reloading the spine)
  useEffect(() => {
    const spine = spineRef.current;
    if (!spine) return;
    if (activeAnimation && spine.skeleton.data.findAnimation(activeAnimation)) {
      spine.state.setAnimation(0, activeAnimation, true);
    } else {
      // Rest / setup pose
      spine.state.setEmptyAnimation(0, 0);
      spine.skeleton.setToSetupPose();
    }
  }, [activeAnimation, ready, project, activeSkin, assetVersion]);

  // React to slot visibility toggles
  useEffect(() => {
    const spine = spineRef.current;
    if (!spine) return;
    for (const slot of spine.skeleton.slots) {
      const slotName = slot.data.name;
      if (hidden.has(slotName)) {
        slot.setAttachment(null);
      } else {
        const original = defaultAttachmentsRef.current.get(slot.data.index);
        if (original != null) {
          const att = spine.skeleton.getAttachmentByName(slotName, original);
          slot.setAttachment(att);
        }
      }
    }
  }, [hidden, ready, project, activeSkin]);

  // Pointer handling — `tool === 'hand'` drives drag-to-reposition,
  // otherwise pointerdown picks the topmost slot bbox and selects it.
  useEffect(() => {
    if (!ready) return;
    const app = appRef.current;
    if (!app) return;
    const canvas = app.canvas;

    // Live, per-click hit-test using each attachment's world vertices.
    // Avoids the stale-cache problem during animation, and skips the slow
    // visibility-toggle bbox sweep entirely.
    const pickSlotLive = (clientX: number, clientY: number): string | null => {
      const spine = spineRef.current;
      if (!spine) return null;
      const r = canvas.getBoundingClientRect();
      const cssToCanvas = canvas.width / r.width;
      const px = (clientX - r.left) * cssToCanvas;
      const py = (clientY - r.top) * cssToCanvas;
      const sx = spine.x;
      const sy = spine.y;
      const sX = spine.scale.x || 1;
      const sY = spine.scale.y || 1;
      const drawOrder = spine.skeleton.drawOrder;
      for (let i = drawOrder.length - 1; i >= 0; i--) {
        const slot = drawOrder[i];
        if (hidden.has(slot.data.name)) continue;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const att = slot.attachment as any;
        if (!att) continue;
        let verts: Float32Array | null = null;
        try {
          // Detect attachment kind by available API rather than class name
          // (constructor names get mangled in production bundles).
          // Mesh attachments expose `worldVerticesLength`; region attachments
          // don't and use the 4-arg `computeWorldVertices` signature.
          const meshLen: number | undefined = att.worldVerticesLength;
          if (typeof meshLen === 'number' && meshLen > 0) {
            verts = new Float32Array(meshLen);
            att.computeWorldVertices(slot, 0, meshLen, verts, 0, 2);
          } else if (typeof att.computeWorldVertices === 'function') {
            verts = new Float32Array(8);
            att.computeWorldVertices(slot, verts, 0, 2);
          }
        } catch { /* skip */ }
        if (!verts || verts.length === 0) continue;
        let minSx = Infinity, minSy = Infinity, maxSx = -Infinity, maxSy = -Infinity;
        for (let j = 0; j < verts.length; j += 2) {
          // skeleton-world → canvas pixel coords. spine-pixi-v8 renders
          // skeleton Y as-is into pixi Y-down (no flip), so adding here.
          const cx = sx + verts[j] * sX;
          const cy = sy + verts[j + 1] * sY;
          if (cx < minSx) minSx = cx;
          if (cx > maxSx) maxSx = cx;
          if (cy < minSy) minSy = cy;
          if (cy > maxSy) maxSy = cy;
        }
        if (px >= minSx && px <= maxSx && py >= minSy && py <= maxSy) {
          return slot.data.name;
        }
      }
      return null;
    };

    // Convert canvas-pixel coords → spine skeleton-world coords. spine-pixi-v8
    // renders skeleton Y as-is into pixi Y-down (no flip), so no negation.
    const screenToWorld = (clientX: number, clientY: number) => {
      const spine = spineRef.current!;
      const r = canvas.getBoundingClientRect();
      const cssToCanvas = canvas.width / r.width;
      const px = (clientX - r.left) * cssToCanvas;
      const py = (clientY - r.top) * cssToCanvas;
      return {
        x: (px - spine.x) / (spine.scale.x || 1),
        y: (py - spine.y) / (spine.scale.y || 1),
      };
    };

    // Compute the current screen-pixel position of the part's attachment
    // center (post-transforms, current animation frame). Returned as client
    // coords so it lines up with PointerEvent.clientX/Y.
    const partCenterClient = (slotName: string): { x: number; y: number } | null => {
      const spine = spineRef.current;
      if (!spine) return null;
      const slot = spine.skeleton.findSlot(slotName);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const att = slot?.attachment as any;
      if (!slot || !att || typeof att.x !== 'number') return null;
      const world = { x: att.x, y: att.y };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (slot.bone as any).localToWorld(world);
      const rect = canvas.getBoundingClientRect();
      const cssToCanvas = canvas.width / rect.width;
      const cx = spine.x + world.x * (spine.scale.x || 1);
      const cy = spine.y + world.y * (spine.scale.y || 1);
      return { x: rect.left + cx / cssToCanvas, y: rect.top + cy / cssToCanvas };
    };

    const onDown = (e: PointerEvent) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__downFiredAt = Date.now();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__downTool = tool;
      const slotName = pickSlotLive(e.clientX, e.clientY);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__downPick = slotName;
      if (!slotName) return;
      const spine = spineRef.current;
      if (!spine) {
        if (tool !== 'select') return;
        selectSlot(slotName);
        return;
      }
      if (tool === 'select') {
        selectSlot(slotName);
        return;
      }
      const slot = spine.skeleton.findSlot(slotName);
      const orig = originalAttachmentXYRef.current.get(slotName);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const att = slot?.attachment as any;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__downGuards = { hasSlot: !!slot, hasOrig: !!orig, hasAtt: !!att, attXType: typeof att?.x };
      if (!slot || !orig || !att || typeof att.x !== 'number') return;

      if (tool === 'hand') {
        const startWorld = { x: att.x, y: att.y };
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (slot.bone as any).localToWorld(startWorld);
        dragRef.current = {
          kind: 'hand',
          slot: slotName,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          bone: slot.bone as any,
          startWorld,
          startClient: { x: e.clientX, y: e.clientY },
        };
      } else if (tool === 'scale' || tool === 'rotate') {
        const anchor = partCenterClient(slotName);
        if (!anchor) return;
        if (tool === 'scale') {
          // Use distance from part center → factor.
          const startDist = Math.max(
            1,
            Math.hypot(e.clientX - anchor.x, e.clientY - anchor.y),
          );
          dragRef.current = {
            kind: 'scale',
            slot: slotName,
            anchorClient: anchor,
            startDist,
            startScaleX: typeof att.scaleX === 'number' ? att.scaleX : orig.scaleX,
            startScaleY: typeof att.scaleY === 'number' ? att.scaleY : orig.scaleY,
          };
        } else {
          // rotate
          const startAngleRad = Math.atan2(e.clientY - anchor.y, e.clientX - anchor.x);
          dragRef.current = {
            kind: 'rotate',
            slot: slotName,
            anchorClient: anchor,
            startAngleRad,
            startRotationDeg: typeof att.rotation === 'number' ? att.rotation : orig.rotation,
          };
        }
      } else {
        return;
      }

      selectSlot(slotName);
      setDragging(true);
      canvas.setPointerCapture(e.pointerId);
      e.preventDefault();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__dragRef = dragRef.current;
    };

    const onMove = (e: PointerEvent) => {
      const drag = dragRef.current;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__lastMoveAt = Date.now();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__lastMoveDragKind = drag?.kind ?? null;
      if (!drag) return;
      const spine = spineRef.current;
      if (!spine) return;
      const slot = spine.skeleton.findSlot(drag.slot);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const att = slot?.attachment as any;
      if (!slot || !att || typeof att.x !== 'number') return;

      if (drag.kind === 'hand') {
        const startWorldFromClient = screenToWorld(drag.startClient.x, drag.startClient.y);
        const curWorldFromClient = screenToWorld(e.clientX, e.clientY);
        const worldEnd = {
          x: drag.startWorld.x + (curWorldFromClient.x - startWorldFromClient.x),
          y: drag.startWorld.y + (curWorldFromClient.y - startWorldFromClient.y),
        };
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (drag.bone as any).worldToLocal(worldEnd);
        att.x = worldEnd.x;
        att.y = worldEnd.y;
      } else if (drag.kind === 'scale') {
        const curDist = Math.max(
          1,
          Math.hypot(e.clientX - drag.anchorClient.x, e.clientY - drag.anchorClient.y),
        );
        const f = curDist / drag.startDist;
        att.scaleX = drag.startScaleX * f;
        att.scaleY = drag.startScaleY * f;
      } else if (drag.kind === 'rotate') {
        const curAngleRad = Math.atan2(
          e.clientY - drag.anchorClient.y,
          e.clientX - drag.anchorClient.x,
        );
        let deltaDeg = ((curAngleRad - drag.startAngleRad) * 180) / Math.PI;
        // Normalize delta to [-180, 180] so spinning past the seam doesn't jump.
        while (deltaDeg > 180) deltaDeg -= 360;
        while (deltaDeg < -180) deltaDeg += 360;
        att.rotation = drag.startRotationDeg + deltaDeg;
      }
      if (typeof att.updateRegion === 'function') att.updateRegion();
    };

    const onUp = async (e: PointerEvent) => {
      const d = dragRef.current;
      dragRef.current = null;
      try { canvas.releasePointerCapture(e.pointerId); } catch { /* noop */ }
      setDragging(false);
      if (!d) return;
      const spine = spineRef.current;
      if (!spine) return;
      const slot = spine.skeleton.findSlot(d.slot);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const att = slot?.attachment as any;
      if (!slot || !att || typeof att.x !== 'number') return;
      const orig = originalAttachmentXYRef.current.get(d.slot);
      if (!orig) return;

      const cur = { ...(project?.transforms?.[activeSkin] ?? {}) };
      const existing = cur[d.slot] ?? {};
      let next: Transform = { ...existing };
      if (d.kind === 'hand') {
        next = { ...next, x: att.x - orig.x, y: att.y - orig.y };
      } else if (d.kind === 'scale') {
        const denom = orig.scaleX || 1;
        next = { ...next, scale: att.scaleX / denom };
      } else if (d.kind === 'rotate') {
        next = { ...next, rotation: att.rotation - orig.rotation };
      }
      // Normalize: drop fields equal to default.
      const norm: Transform = {};
      if (Math.abs(next.x ?? 0) > 0.01) norm.x = next.x;
      if (Math.abs(next.y ?? 0) > 0.01) norm.y = next.y;
      if (Math.abs(next.rotation ?? 0) > 0.01) norm.rotation = next.rotation;
      if (next.scale !== undefined && Math.abs(next.scale - 1) > 0.001) norm.scale = next.scale;

      const isEmpty = Object.keys(norm).length === 0;
      if (isEmpty) {
        if (d.slot in cur) delete cur[d.slot];
        else return;
      } else {
        cur[d.slot] = norm;
      }
      try {
        await api.putTransforms(activeSkin, cur);
        const fresh = await api.getStatus();
        if (fresh.open) refreshProjectSkins(fresh as Project);
      } catch (err) {
        console.warn('save transforms failed', err);
      }
    };

    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointercancel', onUp);
    return () => {
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointercancel', onUp);
    };
  }, [ready, hidden, selectSlot, tool, project, activeSkin, refreshProjectSkins]);

  const wrapClass = [
    'canvas-wrap',
    tool !== 'select' ? `tool-${tool}` : '',
    tool !== 'select' && dragging ? 'tool-active' : '',
  ].filter(Boolean).join(' ');

  return (
    <div className={wrapClass}>
      <div ref={containerRef} className="pixi-container" />
      {project && <CanvasTools />}
      {!project && <div className="canvas-empty">Open a project to begin.</div>}
    </div>
  );
}
