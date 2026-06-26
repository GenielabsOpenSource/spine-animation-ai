import { useStore } from '../state/store';

export function TopBar({
  onOpen,
  onGenerate,
  onExport,
  onSettings,
  onLogs,
  exporting,
}: {
  onOpen: () => void;
  onGenerate: () => void;
  onExport: () => void;
  onSettings: () => void;
  onLogs: () => void;
  exporting: boolean;
}) {
  const project = useStore((s) => s.project);
  const activeSkin = useStore((s) => s.activeSkin);
  const setActiveSkin = useStore((s) => s.setActiveSkin);
  const activeAnimation = useStore((s) => s.activeAnimation);
  const setActiveAnimation = useStore((s) => s.setActiveAnimation);
  const animations = project?.animations ?? [];
  const secrets = useStore((s) => s.secrets);
  const missingRequired = (secrets ?? []).filter((s) => s.required && !s.is_set);

  return (
    <header className="topbar">
      <div className="topbar-left">
        <div className="brand">
          <img src="/genie-icon.svg" alt="" />
          <span>Reskin Studio</span>
        </div>
        {project ? (
          <div className="topbar-meta">
            <span className="proj">project <code>{project.name}</code></span>
            <select
              value={activeSkin}
              onChange={(e) => setActiveSkin(e.target.value)}
              className="skin-select"
              aria-label="Active look"
            >
              {Array.from(new Set(['default', ...project.skins])).map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
            {animations.length > 0 && (
              <select
                value={activeAnimation ?? '__none__'}
                onChange={(e) => setActiveAnimation(
                  e.target.value === '__none__' ? null : e.target.value,
                )}
                className="skin-select"
                aria-label="Active animation"
                title="Animation"
              >
                <option value="__none__">— rest pose —</option>
                {animations.map((a) => (
                  <option key={a} value={a}>{a}</option>
                ))}
              </select>
            )}
          </div>
        ) : (
          <span className="proj muted">No project open</span>
        )}
      </div>
      <div className="topbar-actions">
        <button onClick={onLogs} aria-label="Logs" disabled={!project}>Logs</button>
        <button
          className={`settings-btn ${missingRequired.length ? 'has-alert' : ''}`}
          onClick={onSettings}
          aria-label={missingRequired.length
            ? `Settings — ${missingRequired.length} required API key${missingRequired.length > 1 ? 's' : ''} missing`
            : 'Settings'}
          title={missingRequired.length
            ? `Missing: ${missingRequired.map((s) => s.label).join(', ')}`
            : undefined}
        >
          Settings
          {missingRequired.length > 0 && <span className="key-badge" aria-hidden="true" />}
        </button>
        <button onClick={onOpen}>Open project</button>
        <button className="success" onClick={onGenerate} disabled={!project}>Generate</button>
        <button className="primary" onClick={onExport} disabled={!project || exporting}>
          {exporting ? 'Exporting…' : 'Export'}
        </button>
      </div>
    </header>
  );
}
