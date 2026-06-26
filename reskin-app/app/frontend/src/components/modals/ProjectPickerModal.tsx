import { useState } from 'react';
import { api, isMultiProjectChoice, type MultiProjectChoice, type Project } from '../../api/client';

export function ProjectPickerModal({
  choice,
  onResolved,
  onClose,
}: {
  choice: MultiProjectChoice;
  onResolved: (project: Project) => void;
  onClose: () => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pick = async (path: string) => {
    setBusy(path);
    setError(null);
    try {
      const r = await api.openProject(path);
      if (isMultiProjectChoice(r)) {
        setError('Picked folder still has multiple projects.');
        return;
      }
      onResolved(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <header>
          <h2>Pick a character</h2>
          <button onClick={onClose}>×</button>
        </header>
        <div className="modal-body">
          <div className="muted" style={{ fontSize: 'var(--text-xs)' }}>
            {choice.candidates.length} Spine projects in this folder. Pick one to open.
          </div>
          <div className="project-picker-list">
            {choice.candidates.map((c) => (
              <button
                key={c.path}
                className="project-picker-card"
                disabled={busy !== null}
                onClick={() => pick(c.path)}
              >
                <div className="project-picker-title">
                  {busy === c.path ? 'Opening…' : c.display_name}
                </div>
                <div className="project-picker-path">{c.path}</div>
              </button>
            ))}
          </div>
          {error && <div className="muted" style={{ color: 'var(--destructive)' }}>{error}</div>}
        </div>
      </div>
    </div>
  );
}
