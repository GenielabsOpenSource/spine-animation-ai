import { useEffect, useRef, useState, useCallback } from 'react';
import { useStore } from '../../state/store';

/* Per-slot mask editor.
 *
 * Shows the FULL reskinned bbox crop (not the SAM-masked version) so the
 * user can see what's outside the current mask and add it back if needed.
 *
 * Tools:
 *   - Brush  : paint into the mask
 *   - Eraser : paint out of the mask
 *   - Lasso  : click to add anchors, double-click to close polygon → fill
 *   - Magic  : SAM-based selection (requires /magic_cut.onnx + embedding)
 *
 * The mask is a binary 8-bit PNG at the same dimensions as the raw image.
 * Save → PUT to /api/skin/{skin}/mask-image/{slot}, backend re-bakes the atlas.
 */

type Tool = 'brush' | 'eraser' | 'lasso' | 'magic';

const VIEW = 540;            // canvas display size (square, fits content)
const MASK_COLOR = [30, 144, 255]; // overlay color (rgb)

export function MaskModal({
  slot,
  onClose,
  onSaved,
}: {
  slot: string;
  onClose: () => void;
  onSaved: () => void;
}) {
  const activeSkin = useStore((s) => s.activeSkin);
  const isGenerated = activeSkin && activeSkin !== 'default';

  const imgRef = useRef<HTMLImageElement | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null); // off-screen, full-res mask
  const overlayRef = useRef<HTMLCanvasElement | null>(null);    // visible overlay on top of img
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [tool, setTool] = useState<Tool>('brush');
  const [brushSize, setBrushSize] = useState(40);
  const [overlayOpacity, setOverlayOpacity] = useState(0.55);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [magicReady, setMagicReady] = useState(false);

  // Lasso state
  const lassoPoints = useRef<Array<[number, number]>>([]);
  const drawingStrokeRef = useRef(false);

  const slotPath = isGenerated
    ? `.genie/skins/${activeSkin}/extracted_raw/${slot}.png`
    : `${slot}.png`;
  // Cache-bust so re-opening the modal after a save reloads
  const stampRef = useRef(Date.now());
  const rawUrl = isGenerated
    ? `/api/skin/${encodeURIComponent(activeSkin)}/raw/${encodeURIComponent(slot)}?v=${stampRef.current}`
    : `/api/project/file/${encodeURIComponent(slot)}.png`;
  const maskUrl = isGenerated
    ? `/api/skin/${encodeURIComponent(activeSkin)}/mask-image/${encodeURIComponent(slot)}?v=${stampRef.current}`
    : null;

  // ───────── Geometry: image fit inside VIEW × VIEW box ─────────
  const fit = imgSize
    ? (() => {
        const s = Math.min(VIEW / imgSize.w, VIEW / imgSize.h);
        return {
          x: (VIEW - imgSize.w * s) / 2,
          y: (VIEW - imgSize.h * s) / 2,
          w: imgSize.w * s,
          h: imgSize.h * s,
          scale: s,
        };
      })()
    : null;

  const viewToImage = (px: number, py: number) => {
    if (!fit || !imgSize) return { x: 0, y: 0 };
    return {
      x: Math.round((px - fit.x) / fit.scale),
      y: Math.round((py - fit.y) / fit.scale),
    };
  };

  // ───────── Canvas helpers ─────────
  const getOrCreateMaskCanvas = useCallback(() => {
    if (!maskCanvasRef.current && imgSize) {
      const c = document.createElement('canvas');
      c.width = imgSize.w;
      c.height = imgSize.h;
      maskCanvasRef.current = c;
    }
    return maskCanvasRef.current;
  }, [imgSize]);

  // Redraw the visible overlay (mask tinted, plus lasso preview if drawing)
  const redrawOverlay = useCallback(() => {
    const overlay = overlayRef.current;
    if (!overlay || !fit || !imgSize) return;
    const ctx = overlay.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, VIEW, VIEW);

    const mask = getOrCreateMaskCanvas();
    if (mask) {
      // Tint the mask: render mask alpha through a colored fill
      const tmp = document.createElement('canvas');
      tmp.width = imgSize.w;
      tmp.height = imgSize.h;
      const tctx = tmp.getContext('2d')!;
      tctx.drawImage(mask, 0, 0);
      tctx.globalCompositeOperation = 'source-in';
      tctx.fillStyle = `rgba(${MASK_COLOR[0]},${MASK_COLOR[1]},${MASK_COLOR[2]},1)`;
      tctx.fillRect(0, 0, imgSize.w, imgSize.h);
      ctx.globalAlpha = overlayOpacity;
      ctx.drawImage(tmp, fit.x, fit.y, fit.w, fit.h);
      ctx.globalAlpha = 1;
    }

    if (tool === 'lasso' && lassoPoints.current.length > 0) {
      ctx.beginPath();
      const [x0, y0] = lassoPoints.current[0];
      ctx.moveTo(x0, y0);
      for (let i = 1; i < lassoPoints.current.length; i++) {
        const [x, y] = lassoPoints.current[i];
        ctx.lineTo(x, y);
      }
      ctx.strokeStyle = '#4ae';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
      // anchor dots
      for (const [x, y] of lassoPoints.current) {
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#4ae';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }, [fit, imgSize, tool, overlayOpacity, getOrCreateMaskCanvas]);

  useEffect(() => { redrawOverlay(); }, [redrawOverlay]);

  // ───────── Initial mask load ─────────
  // Backend saves masks as 8-bit grayscale PNGs (PIL "L" mode). When loaded
  // into a canvas they're fully opaque, so the source-in compositing in
  // redrawOverlay would tint the whole bbox blue. Convert to alpha-as-mask
  // so the mask canvas matches what the brush tools produce: white pixels
  // have alpha = grayscale value, everything else stays transparent.
  useEffect(() => {
    if (!imgSize || !maskUrl) return;
    const c = getOrCreateMaskCanvas();
    if (!c) return;
    const ctx = c.getContext('2d')!;
    const im = new Image();
    im.crossOrigin = '';
    im.onload = () => {
      ctx.clearRect(0, 0, c.width, c.height);
      // Sample the mask at full resolution, then rebuild it with grayscale → alpha.
      const tmp = document.createElement('canvas');
      tmp.width = c.width;
      tmp.height = c.height;
      const tctx = tmp.getContext('2d')!;
      tctx.drawImage(im, 0, 0, c.width, c.height);
      const id = tctx.getImageData(0, 0, c.width, c.height);
      const d = id.data;
      for (let i = 0; i < d.length; i += 4) {
        // Grayscale PNGs come back fully opaque (alpha 255) with luminance in
        // R/G/B — derive the mask alpha from RGB only. For RGBA-saved masks
        // (e.g. brush-painted then re-loaded), max(R,G,B) is also white where
        // painted and the alpha channel handles transparent gaps via the
        // multiply: keep whichever is smaller of (RGB-luma, original-alpha).
        const rgb = Math.max(d[i], d[i + 1], d[i + 2]);
        const v = Math.min(rgb, d[i + 3]);
        d[i] = 255; d[i + 1] = 255; d[i + 2] = 255;
        d[i + 3] = v;
      }
      ctx.putImageData(id, 0, 0);
      redrawOverlay();
    };
    im.onerror = () => {
      // No existing mask — start fully opaque white so the entire bbox is
      // kept until the user trims it. Matches "Reset to SAM"-equivalent.
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, c.width, c.height);
      redrawOverlay();
    };
    im.src = maskUrl;
  }, [imgSize, maskUrl, getOrCreateMaskCanvas, redrawOverlay]);

  // ───────── Brush / eraser ─────────
  const stamp = (px: number, py: number, additive: boolean) => {
    const c = getOrCreateMaskCanvas();
    if (!c || !fit) return;
    const ctx = c.getContext('2d')!;
    const { x, y } = viewToImage(px, py);
    const r = brushSize / fit.scale / 2;
    ctx.globalCompositeOperation = additive ? 'source-over' : 'destination-out';
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
    redrawOverlay();
  };

  // ───────── Lasso ─────────
  const closeLasso = (additive: boolean) => {
    const c = getOrCreateMaskCanvas();
    if (!c || lassoPoints.current.length < 3 || !fit) {
      lassoPoints.current = [];
      redrawOverlay();
      return;
    }
    const ctx = c.getContext('2d')!;
    ctx.save();
    ctx.globalCompositeOperation = additive ? 'source-over' : 'destination-out';
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    const first = lassoPoints.current[0];
    const a = viewToImage(first[0], first[1]);
    ctx.moveTo(a.x, a.y);
    for (let i = 1; i < lassoPoints.current.length; i++) {
      const p = viewToImage(lassoPoints.current[i][0], lassoPoints.current[i][1]);
      ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    lassoPoints.current = [];
    redrawOverlay();
  };

  const onPointerDown = (e: React.PointerEvent) => {
    const box = containerRef.current?.getBoundingClientRect();
    if (!box) return;
    const px = e.clientX - box.left;
    const py = e.clientY - box.top;
    if (tool === 'brush') { drawingStrokeRef.current = true; stamp(px, py, true); }
    else if (tool === 'eraser') { drawingStrokeRef.current = true; stamp(px, py, false); }
    else if (tool === 'lasso') {
      if (e.detail >= 2 && lassoPoints.current.length >= 3) {
        // double-click closes the polygon (additive). Hold Alt to subtract.
        closeLasso(!e.altKey);
      } else {
        lassoPoints.current = [...lassoPoints.current, [px, py]];
        redrawOverlay();
      }
    } else if (tool === 'magic') {
      runMagicSelect(px, py, !e.altKey);
    }
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!drawingStrokeRef.current) return;
    const box = containerRef.current?.getBoundingClientRect();
    if (!box) return;
    const px = e.clientX - box.left;
    const py = e.clientY - box.top;
    stamp(px, py, tool === 'brush');
  };
  const onPointerUp = () => { drawingStrokeRef.current = false; };

  // ───────── Magic Select (SAM via ONNX) ─────────
  const onnxSessionRef = useRef<any>(null);
  const embeddingRef = useRef<any>(null);
  const magicPointsRef = useRef<Array<{ x: number; y: number; label: 0|1 }>>([]);

  const ensureMagic = async (): Promise<boolean> => {
    if (magicReady) return true;
    try {
      const ort = await import('onnxruntime-web');
      // Probe the onnx file
      const probe = await fetch('/magic_cut.onnx', { method: 'HEAD' });
      if (!probe.ok) {
        setError('magic_cut.onnx not in /public — drop the SAM decoder file there to enable Magic Select.');
        return false;
      }
      onnxSessionRef.current = await ort.InferenceSession.create('/magic_cut.onnx');
      // Fetch embedding (cached on backend)
      const r = await fetch(`/api/skin/${encodeURIComponent(activeSkin)}/embedding/${encodeURIComponent(slot)}`, { method: 'POST' });
      if (!r.ok) {
        setError(`embedding endpoint failed: ${await r.text()}`);
        return false;
      }
      embeddingRef.current = await r.json();
      setMagicReady(true);
      return true;
    } catch (e) {
      setError(`magic select unavailable: ${(e as Error).message}`);
      return false;
    }
  };

  const runMagicSelect = async (px: number, py: number, additive: boolean) => {
    if (!await ensureMagic()) return;
    const { x, y } = viewToImage(px, py);
    magicPointsRef.current = [...magicPointsRef.current, { x, y, label: additive ? 1 : 0 }];
    try {
      const ort = await import('onnxruntime-web');
      const emb = embeddingRef.current.result;
      const oldH = emb.image_shape[0];
      const oldW = emb.image_shape[1];
      const encodedSize = emb.encoded_image_size || 1024;
      const scaleX = encodedSize / Math.max(oldH, oldW) * (oldW / oldW);  // simplifies to encodedSize / max
      const scale = encodedSize / Math.max(oldH, oldW);
      // decode embedding
      const bin = atob(emb.image_embedding);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      const f32 = new Float32Array(bytes.buffer);
      const pts = magicPointsRef.current;
      const n = pts.length;
      const coords = new Float32Array(2 * (n + 1));
      const labels = new Float32Array(n + 1);
      for (let i = 0; i < n; i++) {
        coords[2 * i] = pts[i].x * scale;
        coords[2 * i + 1] = pts[i].y * scale;
        labels[i] = pts[i].label;
      }
      coords[2 * n] = 0; coords[2 * n + 1] = 0; labels[n] = -1;
      const inputs: any = {
        image_embeddings: new ort.Tensor('float32', f32, [1, 256, 64, 64]),
        point_coords: new ort.Tensor('float32', coords, [1, n + 1, 2]),
        point_labels: new ort.Tensor('float32', labels, [1, n + 1]),
        mask_input: new ort.Tensor('float32', new Float32Array(256 * 256), [1, 1, 256, 256]),
        has_mask_input: new ort.Tensor('float32', new Float32Array([0])),
        orig_im_size: new ort.Tensor('float32', new Float32Array([oldH, oldW]), [2]),
      };
      const out = await onnxSessionRef.current.run(inputs);
      const masksTensor = out.masks || out[Object.keys(out)[0]];
      const dims = masksTensor.dims; // [1,1,H,W]
      const data = masksTensor.data as Float32Array;
      const mh = dims[2], mw = dims[3];
      // Render to mask canvas
      const c = getOrCreateMaskCanvas();
      if (!c) return;
      const tmp = document.createElement('canvas');
      tmp.width = mw; tmp.height = mh;
      const tctx = tmp.getContext('2d')!;
      const id = tctx.createImageData(mw, mh);
      for (let i = 0; i < mw * mh; i++) {
        const v = data[i] > (emb.mask_threshold ?? 0) ? 255 : 0;
        id.data[i*4] = v; id.data[i*4+1] = v; id.data[i*4+2] = v; id.data[i*4+3] = v;
      }
      tctx.putImageData(id, 0, 0);
      const ctx = c.getContext('2d')!;
      ctx.save();
      ctx.globalCompositeOperation = additive ? 'source-over' : 'destination-out';
      ctx.drawImage(tmp, 0, 0, c.width, c.height);
      ctx.restore();
      redrawOverlay();
    } catch (e) {
      setError(`magic select failed: ${(e as Error).message}`);
    }
  };

  // ───────── Save ─────────
  const save = async () => {
    if (!isGenerated) return;
    const c = maskCanvasRef.current;
    if (!c) return;
    setBusy(true);
    setError(null);
    try {
      // Convert mask to a single-channel-friendly PNG (white = keep, black = drop)
      // Our canvas is RGBA; we just save it — backend converts to L channel.
      const blob: Blob = await new Promise((r) => c.toBlob((b) => r(b!), 'image/png'));
      const r = await fetch(`/api/skin/${encodeURIComponent(activeSkin)}/mask-image/${encodeURIComponent(slot)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'image/png' },
        body: blob,
      });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      onSaved();
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally { setBusy(false); }
  };

  const reset = () => {
    const c = getOrCreateMaskCanvas();
    if (!c || !imgSize) return;
    const ctx = c.getContext('2d')!;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, c.width, c.height);
    lassoPoints.current = [];
    magicPointsRef.current = [];
    redrawOverlay();
  };

  const clearAll = () => {
    const c = getOrCreateMaskCanvas();
    if (!c) return;
    const ctx = c.getContext('2d')!;
    ctx.clearRect(0, 0, c.width, c.height);
    lassoPoints.current = [];
    magicPointsRef.current = [];
    redrawOverlay();
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal modal-wide" onClick={(e) => e.stopPropagation()}>
        <header>
          <h2>Mask — <code>{slot}</code></h2>
          <button onClick={onClose}>×</button>
        </header>
        <div className="modal-body">
          <div className="mask-editor-wide">
            <div className="mask-toolbar">
              {(['brush','eraser','lasso','magic'] as Tool[]).map(t => (
                <button
                  key={t}
                  onClick={() => {
                    if (t === 'magic' && !useStore.getState().ensureSecrets(['SAM_SERVER_URL'], 'use Magic-select')) return;
                    setTool(t); lassoPoints.current = []; magicPointsRef.current = []; redrawOverlay();
                  }}
                  className={tool === t ? 'tool active' : 'tool'}
                >{t}</button>
              ))}
              <div className="tool-divider" />
              <button onClick={reset}>Reset to SAM</button>
              <button onClick={clearAll}>Clear</button>
            </div>

            <div
              ref={containerRef}
              className="mask-viewport-wide"
              style={{ width: VIEW, height: VIEW }}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerLeave={onPointerUp}
            >
              <img
                ref={imgRef}
                src={rawUrl}
                alt={slot}
                style={{
                  position: 'absolute',
                  left: fit?.x ?? 0, top: fit?.y ?? 0,
                  width: fit?.w ?? VIEW, height: fit?.h ?? VIEW,
                  pointerEvents: 'none',
                }}
                onLoad={(e) => {
                  const im = e.currentTarget;
                  setImgSize({ w: im.naturalWidth, h: im.naturalHeight });
                }}
              />
              <canvas
                ref={overlayRef}
                width={VIEW}
                height={VIEW}
                style={{ position: 'absolute', inset: 0, cursor: tool === 'lasso' || tool === 'magic' ? 'crosshair' : 'cell' }}
              />
            </div>

            <div className="mask-controls-wide">
              <div className="slider">
                <label>Brush size</label>
                <input type="range" min={2} max={120} value={brushSize}
                  onChange={(e) => setBrushSize(parseInt(e.target.value, 10))}
                  disabled={tool !== 'brush' && tool !== 'eraser'} />
                <span className="val">{brushSize}px</span>
              </div>
              <div className="slider">
                <label>Overlay</label>
                <input type="range" min={0} max={1} step={0.01} value={overlayOpacity}
                  onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))} />
                <span className="val">{Math.round(overlayOpacity*100)}%</span>
              </div>
              <p className="hint" style={{fontSize:12, color:'#8a90a1'}}>
                Brush/Eraser: paint into or out of the mask. Lasso: click points,
                double-click to close (Alt+double-click subtracts). Magic: click
                points to add (Alt-click to subtract) — runs SAM via /magic_cut.onnx.
              </p>
              {error && <p style={{color:'#e88', fontSize:12}}>{error}</p>}
              {!isGenerated && (
                <p style={{color:'#e88', fontSize:12}}>Pick a generated skin first.</p>
              )}
              <div className="actions">
                <button onClick={onClose}>Cancel</button>
                <button className="primary" onClick={save} disabled={busy || !isGenerated}>
                  {busy ? 'Saving…' : 'Save mask'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
