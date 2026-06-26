import { useEffect, useRef, useState } from 'react';
import { useStore } from '../../state/store';
import { api, type AppSettings, type ErosionSettings, type ReferenceSettings, type ReskinMethod, type SecretField, type SegmentationMethod, type SegmentationSettings } from '../../api/client';

const METHODS: { value: ReskinMethod; title: string; body: string }[] = [
  {
    value: 'atlas',
    title: 'Atlas + snapshot',
    body: 'Send Gemini the rendered character pose alongside the original packed atlas. Pose gives style context; atlas regions are reskinned in place, then segmented per region.',
  },
  {
    value: 'exploded',
    title: 'Exploded parts',
    body: 'Bin-pack every part with 10 px gaps and send that to Gemini. No pose context, but each part is unambiguously isolated in the layout.',
  },
];

const DEFAULT_EROSION: ErosionSettings = {
  enabled: true,
  px_small: 0,
  px_medium: 1,
  px_large: 2,
  px_xlarge: 3,
  small_threshold: 60,
  medium_threshold: 200,
  large_threshold: 500,
};

const DEFAULT_REFERENCE: ReferenceSettings = {
  enabled: false,
  prompt: '',
  has_image: false,
};

const DEFAULT_SEGMENTATION: SegmentationSettings = { method: 'sam' };

const SEGMENTATION_METHODS: { value: SegmentationMethod; title: string; body: string }[] = [
  {
    value: 'sam',
    title: 'SAM (bbox-prompted)',
    body: "Send each region's bbox to your SAM Flask server. Best when SAM is available and the parts have well-defined silhouettes that bbox prompts can isolate.",
  },
  {
    value: 'bg_components',
    title: 'Background removal + connected components',
    body: "Strip the atlas background via Bria, label connected components, and match each region to the highest-IoU component using the original part's silhouette. Doesn't need SAM; works well when parts are visually disjoint in the atlas.",
  },
];

const EROSION_FIELDS: { key: keyof Pick<ErosionSettings, 'px_small' | 'px_medium' | 'px_large' | 'px_xlarge'>; label: string; hint: (s: ErosionSettings) => string }[] = [
  { key: 'px_small', label: 'Tiny parts', hint: (s) => `side < ${s.small_threshold} px` },
  { key: 'px_medium', label: 'Small parts', hint: (s) => `side < ${s.medium_threshold} px` },
  { key: 'px_large', label: 'Medium parts', hint: (s) => `side < ${s.large_threshold} px` },
  { key: 'px_xlarge', label: 'Large parts', hint: (s) => `side ≥ ${s.large_threshold} px` },
];

export function SettingsModal({ onClose }: { onClose: () => void }) {
  const method = useStore((s) => s.reskinMethod);
  const setMethod = useStore((s) => s.setReskinMethod);
  const [erosion, setErosion] = useState<ErosionSettings>(DEFAULT_EROSION);
  const [reference, setReference] = useState<ReferenceSettings>(DEFAULT_REFERENCE);
  const [segmentation, setSegmentation] = useState<SegmentationSettings>(DEFAULT_SEGMENTATION);
  const [refImageStamp, setRefImageStamp] = useState<number>(() => Date.now());
  const [refUploading, setRefUploading] = useState(false);
  const refFileInput = useRef<HTMLInputElement | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [secrets, setSecrets] = useState<SecretField[] | null>(null);
  const [secretVals, setSecretVals] = useState<Record<string, string>>({});
  const samUrlRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.getSettings()
      .then((s) => {
        if (cancelled) return;
        setErosion(s.erosion ?? DEFAULT_EROSION);
        setReference(s.reference ?? DEFAULT_REFERENCE);
        setSegmentation(s.segmentation ?? DEFAULT_SEGMENTATION);
        setLoaded(true);
      })
      .catch((e) => { if (!cancelled) { setErr((e as Error).message); setLoaded(true); } });
    api.getSecrets()
      .then((list) => {
        if (cancelled) return;
        setSecrets(list);
        setSecretVals(Object.fromEntries(list.map((s) => [s.name, s.value])));
      })
      .catch(() => { /* keys are optional to load; keep the rest usable */ });
    return () => { cancelled = true; };
  }, []);

  const updateErosion = (patch: Partial<ErosionSettings>) =>
    setErosion((cur) => ({ ...cur, ...patch }));
  const updateReference = (patch: Partial<ReferenceSettings>) =>
    setReference((cur) => ({ ...cur, ...patch }));

  const onPickRefImage = () => refFileInput.current?.click();

  const onRefFileChosen = async (file: File | null) => {
    if (!file) return;
    setRefUploading(true);
    setErr(null);
    try {
      await api.uploadReferenceImage(file);
      setReference((cur) => ({ ...cur, has_image: true }));
      setRefImageStamp(Date.now());
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setRefUploading(false);
    }
  };

  const onRemoveRefImage = async () => {
    setRefUploading(true);
    setErr(null);
    try {
      await api.deleteReferenceImage();
      setReference((cur) => ({ ...cur, has_image: false }));
      setRefImageStamp(Date.now());
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setRefUploading(false);
    }
  };

  const onDone = async () => {
    setSaving(true);
    setErr(null);
    try {
      const next: AppSettings = { erosion, reference, segmentation };
      const saved = await api.putSettings(next);
      setErosion(saved.erosion);
      setReference(saved.reference);
      setSegmentation(saved.segmentation);
      const savedSecrets = await api.putSecrets(secretVals);
      setSecrets(savedSecrets);
      setSecretVals(Object.fromEntries(savedSecrets.map((s) => [s.name, s.value])));
      onClose();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <header>
          <h2>Settings</h2>
          <button onClick={onClose}>×</button>
        </header>
        <div className="modal-body">
          <div className="settings-section">
            <div className="settings-section-title">API keys</div>
            <p className="settings-radio-body" style={{ margin: 0 }}>
              Saved on this machine only (~/.genie-reskin). An existing app/.env pre-fills these.
            </p>
            {secrets == null ? (
              <div className="settings-radio-body">Loading…</div>
            ) : (
              secrets.map((s) => {
                const val = secretVals[s.name] ?? '';
                const missing = s.required && !val.trim();
                return (
                  <div key={s.name} className={`key-field ${missing ? 'missing' : ''}`}>
                    <div className="key-field-head">
                      <span className="key-field-label">{s.label}</span>
                      <span className={`key-tag ${s.required ? 'req' : 'opt'}`}>
                        {s.required ? 'Required' : 'Optional'}
                      </span>
                      {s.help_url && (
                        <a className="key-field-help" href={s.help_url} target="_blank" rel="noreferrer">
                          Get a key
                        </a>
                      )}
                    </div>
                    <input
                      ref={s.name === 'SAM_SERVER_URL' ? samUrlRef : undefined}
                      type="text"
                      value={val}
                      placeholder={s.kind === 'url' ? 'https://your-sam-server…' : 'Paste your key'}
                      autoComplete="off"
                      spellCheck={false}
                      disabled={!loaded}
                      onChange={(e) => setSecretVals((cur) => ({ ...cur, [s.name]: e.target.value }))}
                      style={{ width: '100%', fontFamily: 'ui-monospace, monospace' }}
                    />
                    <div className="key-field-hint">{s.description}</div>
                  </div>
                );
              })
            )}
          </div>

          <div className="settings-section" style={{ marginTop: 'var(--space-5)' }}>
            <div className="settings-section-title">Reskin method</div>
            <div className="settings-radio-group">
              {METHODS.map((m) => (
                <label
                  key={m.value}
                  className={`settings-radio ${method === m.value ? 'selected' : ''}`}
                >
                  <input
                    type="radio"
                    name="reskin-method"
                    value={m.value}
                    checked={method === m.value}
                    onChange={() => setMethod(m.value)}
                  />
                  <div className="settings-radio-text">
                    <div className="settings-radio-title">{m.title}</div>
                    <div className="settings-radio-body">{m.body}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <div className="settings-section" style={{ marginTop: 'var(--space-5)' }}>
            <div className="settings-section-title">Segmentation method</div>
            <div className="settings-radio-group">
              {SEGMENTATION_METHODS.map((m) => (
                <label
                  key={m.value}
                  className={`settings-radio ${segmentation.method === m.value ? 'selected' : ''}`}
                >
                  <input
                    type="radio"
                    name="segmentation-method"
                    value={m.value}
                    disabled={!loaded}
                    checked={segmentation.method === m.value}
                    onChange={() => {
                      setSegmentation({ method: m.value });
                      if (m.value === 'sam' && !(secretVals['SAM_SERVER_URL'] ?? '').trim()) {
                        setTimeout(() => samUrlRef.current?.focus(), 0);
                      }
                    }}
                  />
                  <div className="settings-radio-text">
                    <div className="settings-radio-title">{m.title}</div>
                    <div className="settings-radio-body">{m.body}</div>
                  </div>
                </label>
              ))}
            </div>
            {segmentation.method === 'sam' && !(secretVals['SAM_SERVER_URL'] ?? '').trim() && (
              <div className="settings-error">
                SAM segmentation needs your SAM server URL — add it under <b>API keys</b> above.
              </div>
            )}
          </div>

          <div className="settings-section" style={{ marginTop: 'var(--space-5)' }}>
            <div className="settings-section-title">Mask erosion</div>
            <p className="settings-radio-body" style={{ margin: 0 }}>
              Trims the soft halo Gemini paints at part edges by shrinking the SAM mask after segmentation. Radius scales with part size so thin features survive. Set values to 0 px to skip a tier.
            </p>
            <label className="settings-radio" style={{ alignItems: 'center' }}>
              <input
                type="checkbox"
                checked={erosion.enabled}
                disabled={!loaded}
                onChange={(e) => updateErosion({ enabled: e.target.checked })}
                style={{ accentColor: 'var(--genie-4)' }}
              />
              <div className="settings-radio-text">
                <div className="settings-radio-title">Apply erosion to segmented parts</div>
                <div className="settings-radio-body">
                  When off, masks are used as SAM produced them — keeps soft halos but preserves every pixel.
                </div>
              </div>
            </label>
            <div className="erosion-grid">
              {EROSION_FIELDS.map((f) => (
                <label key={f.key} className="erosion-field">
                  <span className="erosion-field-label">{f.label}</span>
                  <span className="erosion-field-hint">{f.hint(erosion)}</span>
                  <input
                    type="number"
                    min={0}
                    max={20}
                    step={1}
                    value={erosion[f.key]}
                    disabled={!loaded || !erosion.enabled}
                    onChange={(e) => updateErosion({ [f.key]: Math.max(0, parseInt(e.target.value || '0', 10)) } as Partial<ErosionSettings>)}
                  />
                  <span className="erosion-field-unit">px</span>
                </label>
              ))}
            </div>
          </div>

          <div className="settings-section" style={{ marginTop: 'var(--space-5)' }}>
            <div className="settings-section-title">Reference image</div>
            <p className="settings-radio-body" style={{ margin: 0 }}>
              Optional: send a style reference (e.g. a sheet of characters in the target style) as the first image to the model on every full reskin, plus an extra prompt that tells the model how to use it.
            </p>
            <label className="settings-radio" style={{ alignItems: 'center' }}>
              <input
                type="checkbox"
                checked={reference.enabled}
                disabled={!loaded}
                onChange={(e) => updateReference({ enabled: e.target.checked })}
                style={{ accentColor: 'var(--genie-4)' }}
              />
              <div className="settings-radio-text">
                <div className="settings-radio-title">Use reference on full reskin</div>
                <div className="settings-radio-body">
                  When off, neither the image nor the extra prompt are sent.
                </div>
              </div>
            </label>
            <div className="reference-row">
              <div className={`reference-thumb ${reference.has_image ? '' : 'empty'}`}>
                {reference.has_image ? (
                  <img
                    src={api.referenceImageUrl(refImageStamp)}
                    alt="reference"
                    onError={() => setReference((cur) => ({ ...cur, has_image: false }))}
                  />
                ) : (
                  <span>No image</span>
                )}
              </div>
              <div className="reference-actions">
                <input
                  ref={refFileInput}
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const f = e.target.files?.[0] ?? null;
                    e.target.value = '';
                    onRefFileChosen(f);
                  }}
                />
                <button onClick={onPickRefImage} disabled={refUploading || !loaded}>
                  {refUploading ? 'Uploading…' : reference.has_image ? 'Replace image' : 'Upload image'}
                </button>
                <button
                  onClick={onRemoveRefImage}
                  disabled={refUploading || !loaded || !reference.has_image}
                >
                  Remove
                </button>
              </div>
            </div>
            <label className="field" style={{ marginTop: 'var(--space-2)' }}>
              <span>Reference prompt</span>
              <textarea
                rows={3}
                value={reference.prompt}
                disabled={!loaded || !reference.enabled}
                onChange={(e) => updateReference({ prompt: e.target.value })}
                placeholder='e.g. "Match the painterly brushwork and warm pastel palette of the reference."'
              />
            </label>
          </div>

          {err && <div className="settings-error">{err}</div>}

          <div className="actions">
            <button onClick={onClose} disabled={saving}>Cancel</button>
            <button className="primary" onClick={onDone} disabled={saving || !loaded}>
              {saving ? 'Saving…' : 'Done'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
