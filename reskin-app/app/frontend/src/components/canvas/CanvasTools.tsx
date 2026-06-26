import { useState } from 'react';
import { useStore } from '../../state/store';
import { api, type Project } from '../../api/client';

export function CanvasTools() {
  const project = useStore((s) => s.project);
  const tool = useStore((s) => s.tool);
  const setTool = useStore((s) => s.setTool);
  const activeSkin = useStore((s) => s.activeSkin);
  const refreshProjectSkins = useStore((s) => s.refreshProjectSkins);
  const bumpAssetVersion = useStore((s) => s.bumpAssetVersion);
  const [resetting, setResetting] = useState(false);

  if (!project) return null;
  const transforms = project.transforms?.[activeSkin] ?? {};
  const hasTransforms = Object.keys(transforms).length > 0;

  const onReset = async () => {
    if (resetting) return;
    setResetting(true);
    try {
      await api.resetTransforms(activeSkin);
      const fresh = await api.getStatus();
      if (fresh.open) refreshProjectSkins(fresh as Project);
      bumpAssetVersion();
    } catch (e) {
      alert(`reset failed: ${(e as Error).message}`);
    } finally {
      setResetting(false);
    }
  };

  const toggle = (t: typeof tool) => setTool(tool === t ? 'select' : t);

  return (
    <div className="canvas-tools">
      <button
        className={`canvas-tool-btn ${tool === 'hand' ? 'active' : ''}`}
        title="Drag parts to reposition them (saves per-look offsets)"
        onClick={() => toggle('hand')}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M18 11V6a2 2 0 0 0-4 0v5" />
          <path d="M14 10V4a2 2 0 0 0-4 0v6" />
          <path d="M10 10.5V6a2 2 0 0 0-4 0v8" />
          <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15" />
        </svg>
        <span>Hand</span>
      </button>
      <button
        className={`canvas-tool-btn ${tool === 'scale' ? 'active' : ''}`}
        title="Drag away from / toward a part's center to grow / shrink it (uniform scale)"
        onClick={() => toggle('scale')}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M3 7V3h4" />
          <path d="M21 7V3h-4" />
          <path d="M3 17v4h4" />
          <path d="M21 17v4h-4" />
        </svg>
        <span>Scale</span>
      </button>
      <button
        className={`canvas-tool-btn ${tool === 'rotate' ? 'active' : ''}`}
        title="Drag in a circle around a part's center to rotate it"
        onClick={() => toggle('rotate')}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M21 12a9 9 0 1 1-3-6.7" />
          <polyline points="21 4 21 10 15 10" />
        </svg>
        <span>Rotate</span>
      </button>
      <button
        className="canvas-tool-btn"
        title={hasTransforms ? `Reset ${Object.keys(transforms).length} edit(s) for "${activeSkin}"` : 'No edits to reset'}
        disabled={!hasTransforms || resetting}
        onClick={onReset}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <polyline points="1 4 1 10 7 10" />
          <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
        </svg>
        <span>{resetting ? 'Resetting…' : 'Reset'}</span>
      </button>
    </div>
  );
}
