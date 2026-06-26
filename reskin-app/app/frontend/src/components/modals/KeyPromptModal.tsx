import type { KeyPrompt } from '../../state/store';

/* Shown when the user triggers an action (Generate, Retouch, Magic-select…)
 * that needs an API key they haven't set yet. Points them at Settings. */
export function KeyPromptModal({
  prompt,
  onOpenSettings,
  onClose,
}: {
  prompt: KeyPrompt;
  onOpenSettings: () => void;
  onClose: () => void;
}) {
  const plural = prompt.missing.length > 1;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        onClick={(e) => e.stopPropagation()}
        style={{ width: 'min(440px, 92vw)' }}
      >
        <header>
          <h2>Add your API key{plural ? 's' : ''}</h2>
          <button onClick={onClose}>×</button>
        </header>
        <div className="modal-body">
          <p className="hint" style={{ marginTop: 0 }}>
            To {prompt.actionLabel} you need to set the following under Settings → API keys:
          </p>
          <ul className="keyprompt-list">
            {prompt.missing.map((m) => (
              <li key={m.name}>
                <span>{m.label}</span>
                {m.help_url && (
                  <a href={m.help_url} target="_blank" rel="noreferrer">Get a key</a>
                )}
              </li>
            ))}
          </ul>
          <div className="actions">
            <button onClick={onClose}>Not now</button>
            <button className="primary" onClick={onOpenSettings}>Open settings</button>
          </div>
        </div>
      </div>
    </div>
  );
}
