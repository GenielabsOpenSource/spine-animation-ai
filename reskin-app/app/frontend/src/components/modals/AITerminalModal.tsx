import { useState } from 'react';
import { useStore } from '../../state/store';
import { api, type Project } from '../../api/client';

/* Retouch — per-slot AI redraw.
 *
 * Redraws a single slot's texture without re-running a full Generate. On the
 * default look, the first retouch creates a fresh look (random name) from the
 * original part textures and switches into it, so the user keeps editing in
 * their own look. Backend pulls the slot's current PNG, sends it to Gemini with
 * a "redraw matching silhouette" prompt, repacks the atlas, and writes a fresh
 * per-skin Spine JSON. The canvas reload picks up the new texture.
 */

function randomLookName(existing: string[]): string {
  const taken = new Set(['default', ...existing]);
  for (let i = 0; i < 50; i++) {
    const name = `look-${Math.random().toString(36).slice(2, 6)}`;
    if (!taken.has(name)) return name;
  }
  return `look-${Date.now().toString(36)}`;
}

export function AITerminalModal({
  slot,
  onClose,
  onSaved,
}: {
  slot: string;
  onClose: () => void;
  onSaved: () => void;
}) {
  const activeSkin = useStore((s) => s.activeSkin);
  const project = useStore((s) => s.project);
  const setActiveSkin = useStore((s) => s.setActiveSkin);
  const refreshProjectSkins = useStore((s) => s.refreshProjectSkins);
  const [prompt, setPrompt] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isDefault = !activeSkin || activeSkin === 'default';
  // Reserve a name up-front so we can show it; only materialised on Retouch.
  const [newLookName] = useState(() =>
    isDefault ? randomLookName(project?.skins ?? []) : null,
  );

  const beforeUrl = !isDefault
    ? api.fileUrl(`.genie/skins/${activeSkin}/extracted/${slot}.png`)
    : api.fileUrl(`${slot}.png`);

  const run = async () => {
    if (!project || !prompt.trim()) return;
    const targetSkin = isDefault
      ? (newLookName ?? randomLookName(project.skins))
      : activeSkin;
    setError(null);
    setBusy(true);
    try {
      await api.inpaintSlot(targetSkin, slot, prompt);
      if (isDefault) {
        // Surface the freshly-created look in the dropdown, then switch into it.
        try {
          const fresh = await api.getStatus();
          if (fresh.open) refreshProjectSkins(fresh as Project);
        } catch { /* non-fatal */ }
        setActiveSkin(targetSkin);
      }
      onSaved();
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <header>
          <h2>Retouch — <code>{slot}</code></h2>
          <button onClick={onClose}>×</button>
        </header>
        <div className="modal-body">
          <div className="ai-terminal">
            <div className="ai-terminal-current">
              <div className="muted" style={{fontSize:11, marginBottom:6}}>CURRENT</div>
              <img src={beforeUrl} alt={slot} />
            </div>
            <div className="ai-terminal-form">
              {isDefault && (
                <p className="hint" style={{fontSize:12, marginTop:0, marginBottom:'var(--space-3)', color:'var(--charcoal-5)'}}>
                  This creates a new look <code>{newLookName}</code> from the
                  original and switches you into it. Keep retouching, masking,
                  and editing there.
                </p>
              )}
              <label className="field">
                <span>Describe the change</span>
                <textarea
                  rows={4}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder='e.g. "glowing red demonic horns" — keep silhouette identical'
                />
              </label>
              <p className="hint" style={{fontSize:12, color:'#8a90a1'}}>
                Redraws just this part's current texture, preserving its
                silhouette. Useful for fixing one bad part without re-running
                Generate.
              </p>
              {error && <p style={{color:'#e88', fontSize:12}}>{error}</p>}
              <div className="actions">
                <button onClick={onClose}>Cancel</button>
                <button
                  className="primary"
                  onClick={run}
                  disabled={busy || !prompt.trim()}
                >
                  {busy ? 'Retouching…' : 'Retouch'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
