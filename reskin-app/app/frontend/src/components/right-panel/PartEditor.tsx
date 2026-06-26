import { useEffect, useState, useRef } from 'react';
import { useStore } from '../../state/store';
import { api, type Project, type SlotEdit } from '../../api/client';
import { AITerminalModal } from '../modals/AITerminalModal';
import { MaskModal } from '../modals/MaskModal';

const DEFAULT_EDIT: Required<SlotEdit> = {
  hue_shift: 0,
  sat_mult: 1,
  light_shift: 0,
  brightness: 0,
  contrast: 1,
  rgb_balance: [0, 0, 0],
  dx: 0,
  dy: 0,
  rotation: 0,
  scale: 1,
};

export function PartEditor() {
  const slot = useStore((s) => s.selectedSlot);
  const activeSkin = useStore((s) => s.activeSkin);
  const edits = useStore((s) => s.edits);
  const setEdit = useStore((s) => s.setEdit);
  const clearEdit = useStore((s) => s.clearEdit);
  const bumpAssetVersion = useStore((s) => s.bumpAssetVersion);
  const refreshProjectSkins = useStore((s) => s.refreshProjectSkins);

  const edit: Required<SlotEdit> = { ...DEFAULT_EDIT, ...(slot ? edits[slot] : {}) };
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const debouncer = useRef<number | null>(null);
  const [aiOpen, setAiOpen] = useState(false);
  const [maskOpen, setMaskOpen] = useState(false);

  // Debounced live preview
  useEffect(() => {
    if (!slot) return;
    if (debouncer.current) window.clearTimeout(debouncer.current);
    debouncer.current = window.setTimeout(async () => {
      try {
        const blob = await api.fetchEditPreview(slot, activeSkin, edit);
        setPreviewUrl(URL.createObjectURL(blob));
      } catch (e) {
        console.warn('preview failed', e);
      }
    }, 180);
    return () => {
      if (debouncer.current) window.clearTimeout(debouncer.current);
    };
  }, [slot, activeSkin, JSON.stringify(edit)]);

  if (!slot) return null;

  const update = (patch: Partial<SlotEdit>) => setEdit(slot, { ...edit, ...patch });

  const refreshCanvas = async () => {
    try {
      const fresh = await api.getStatus();
      if (fresh.open) refreshProjectSkins(fresh as Project);
    } catch (e) {
      console.warn('refresh project status failed', e);
    }
    bumpAssetVersion();
  };

  return (
    <aside className="rightpanel">
      <div className="rp-header">
        <h3>{slot}</h3>
        <button onClick={() => clearEdit(slot)}>Reset edits</button>
      </div>

      <div className="rp-tabs">
        <button className="active">Edit</button>
        <button
          onClick={() => setMaskOpen(true)}
          disabled={activeSkin === 'default'}
          title={activeSkin === 'default' ? 'Pick a generated look first' : 'Trim this part\'s texture'}
        >Mask</button>
        <button
          className="ai-terminal-btn"
          onClick={() => {
            if (useStore.getState().ensureSecrets(['GEMINI_API_KEY', 'FAL_KEY'], 'retouch a part')) {
              setAiOpen(true);
            }
          }}
          title={activeSkin === 'default' ? 'Retouch this part — creates a new look from the original' : 'Touch up this part with AI'}
        >Retouch</button>
      </div>
      {aiOpen && (
        <AITerminalModal
          slot={slot}
          onClose={() => setAiOpen(false)}
          onSaved={refreshCanvas}
        />
      )}
      {maskOpen && (
        <MaskModal
          slot={slot}
          onClose={() => setMaskOpen(false)}
          onSaved={refreshCanvas}
        />
      )}

      <div className="rp-body">
      {previewUrl && (
        <div className="preview">
          <img src={previewUrl} alt={`${slot} preview`} />
        </div>
      )}

      <Section title="Color">
        <Slider label="Hue" value={edit.hue_shift} min={-180} max={180} step={1}
                onChange={(v) => update({ hue_shift: v })} />
        <Slider label="Saturation" value={edit.sat_mult} min={0} max={3} step={0.01}
                onChange={(v) => update({ sat_mult: v })} />
        <Slider label="Lightness" value={edit.light_shift} min={-1} max={1} step={0.01}
                onChange={(v) => update({ light_shift: v })} />
        <Slider label="Brightness" value={edit.brightness} min={-1} max={1} step={0.01}
                onChange={(v) => update({ brightness: v })} />
        <Slider label="Contrast" value={edit.contrast} min={0} max={3} step={0.01}
                onChange={(v) => update({ contrast: v })} />
      </Section>

      <Section title="RGB Balance">
        <Slider label="R" value={edit.rgb_balance[0]} min={-0.5} max={0.5} step={0.01}
                onChange={(v) => update({ rgb_balance: [v, edit.rgb_balance[1], edit.rgb_balance[2]] })} />
        <Slider label="G" value={edit.rgb_balance[1]} min={-0.5} max={0.5} step={0.01}
                onChange={(v) => update({ rgb_balance: [edit.rgb_balance[0], v, edit.rgb_balance[2]] })} />
        <Slider label="B" value={edit.rgb_balance[2]} min={-0.5} max={0.5} step={0.01}
                onChange={(v) => update({ rgb_balance: [edit.rgb_balance[0], edit.rgb_balance[1], v] })} />
      </Section>

      <Section title="Transform">
        <Slider label="X" value={edit.dx} min={-200} max={200} step={1}
                onChange={(v) => update({ dx: v })} />
        <Slider label="Y" value={edit.dy} min={-200} max={200} step={1}
                onChange={(v) => update({ dy: v })} />
        <Slider label="Rotation" value={edit.rotation} min={-180} max={180} step={1}
                onChange={(v) => update({ rotation: v })} />
        <Slider label="Scale" value={edit.scale} min={0.2} max={3} step={0.01}
                onChange={(v) => update({ scale: v })} />
      </Section>
      </div>
    </aside>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details open className="section">
      <summary>{title}</summary>
      <div className="section-body">{children}</div>
    </details>
  );
}

function Slider({
  label, value, min, max, step, onChange,
}: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <div className="slider">
      <label>{label}</label>
      <input type="range" min={min} max={max} step={step} value={value}
             onChange={(e) => onChange(parseFloat(e.target.value))} />
      <span className="val">{Number(value).toFixed(2)}</span>
    </div>
  );
}
