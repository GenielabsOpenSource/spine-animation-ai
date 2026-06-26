import { useState } from 'react';
import { useStore } from '../../state/store';
import { api, type GenerateResponse, type Project, type RebakeResponse } from '../../api/client';
import { snapshotCanvas } from '../../canvasSnapshot';

const slug = (s: string) =>
  s.toLowerCase().trim().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 32);

export function GlobalGenerateModal({ onClose }: { onClose: () => void }) {
  const project = useStore((s) => s.project);
  const setActiveSkin = useStore((s) => s.setActiveSkin);
  const reskinMethod = useStore((s) => s.reskinMethod);
  const [prompt, setPrompt] = useState('');
  const [skinName, setSkinName] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [rebakeResult, setRebakeResult] = useState<RebakeResponse | null>(null);
  const [wipe, setWipe] = useState(50);
  const [beforeUrl, setBeforeUrl] = useState<string | null>(null);

  const finalSkinName = skinName.trim() || slug(prompt) || 'unnamed';

  const generate = async () => {
    if (!project || !prompt.trim()) return;
    setBusy(true);
    try {
      // Snapshot upload is only required by the atlas method (which uses the
      // canvas pose as Gemini's left-half reference). Exploded mode doesn't
      // use it, but we still snapshot for the wipe-slider before-image.
      try {
        const snap = await snapshotCanvas();
        if (snap) {
          await api.uploadSnapshot(snap, finalSkinName);
          if (beforeUrl) URL.revokeObjectURL(beforeUrl);
          setBeforeUrl(URL.createObjectURL(snap));
        }
      } catch (e) {
        console.warn('snapshot upload failed', e);
      }

      const r = await api.generate(finalSkinName, prompt, reskinMethod);

      let rb: RebakeResponse | null = null;
      try {
        rb = await api.rebake(finalSkinName);
      } catch (e) {
        console.warn('rebake failed', e);
      }

      setResult(r);
      setRebakeResult(rb);
    } catch (e) {
      alert(`generate failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const refreshProjectSkins = useStore((s) => s.refreshProjectSkins);
  const accept = async () => {
    if (!result) return;
    try {
      const fresh = await api.getStatus();
      if (fresh.open) refreshProjectSkins(fresh as Project);
    } catch (e) { /* non-fatal */ }
    setActiveSkin(finalSkinName);
    onClose();
  };

  const methodLabel = reskinMethod === 'exploded' ? 'Exploded parts' : 'Atlas + snapshot';

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <header>
          <h2>New look</h2>
          <button onClick={onClose}>×</button>
        </header>

        {!result ? (
          <div className="modal-body">
            <div className="muted" style={{ fontSize: 'var(--text-xs)' }}>
              Method: <b>{methodLabel}</b> · change in Settings
            </div>
            <label className="field">
              <span>Prompt</span>
              <textarea
                rows={3}
                placeholder='Describe the look you want, e.g. "dark elf, black wings, gold armor"'
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
            </label>
            <label className="field">
              <span>Look name</span>
              <input
                type="text"
                placeholder={slug(prompt) || 'look name'}
                value={skinName}
                onChange={(e) => setSkinName(e.target.value)}
              />
            </label>
            <div className="actions">
              <button onClick={onClose}>Cancel</button>
              <button className="primary" disabled={busy || !prompt.trim()} onClick={generate}>
                {busy ? 'Generating…' : 'Generate'}
              </button>
            </div>
          </div>
        ) : (
          <div className="modal-body">
            <div className="report">
              {rebakeResult ? (
                <>Atlas regions: <b>{rebakeResult.saved.length}</b>{rebakeResult.sam_used ? ' · SAM masks applied' : ' · plain bbox crops (no SAM)'} · method: <b>{result.method}</b></>
              ) : (
                <span className="muted">Rebake failed — preview shows the raw reskin output.</span>
              )}
            </div>

            <div className="reskin-preview-grid">
              <div className="reskin-preview-card">
                <div className="reskin-preview-label">
                  {result.method === 'atlas' ? 'Pose' : 'Composite'}
                </div>
                <div className="wipe-wrap">
                  <img
                    src={api.fileUrl(result.method === 'atlas' && result.reskinned_snapshot
                      ? result.reskinned_snapshot
                      : result.reskinned_composite)}
                    className="wipe-base"
                    alt="reskinned"
                  />
                  {beforeUrl && result.method === 'atlas' && (
                    <>
                      <div className="wipe-clip" style={{ width: `${wipe}%` }}>
                        <div className="wipe-clip-inner">
                          <img src={beforeUrl} alt="original snapshot" />
                        </div>
                      </div>
                      <input
                        type="range" min={0} max={100} value={wipe}
                        onChange={(e) => setWipe(parseInt(e.target.value, 10))}
                        className="wipe-slider"
                      />
                    </>
                  )}
                </div>
              </div>

              <div className="reskin-preview-card">
                <div className="reskin-preview-label">
                  {result.method === 'atlas' ? 'Atlas' : 'Parts grid'}
                </div>
                <img
                  src={api.fileUrl(result.method === 'atlas' && result.reskinned_atlas
                    ? result.reskinned_atlas
                    : result.reskinned_composite)}
                  className="reskin-atlas-thumb"
                  alt="reskinned source"
                />
              </div>
            </div>

            <div className="actions">
              <button onClick={() => { setResult(null); setRebakeResult(null); }}>Try again</button>
              <button className="primary" onClick={accept}>Accept</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
