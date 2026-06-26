import { useEffect, useMemo, useState } from 'react';
import { api, type LogEvent } from '../../api/client';

/* Visual pipeline logs.
 *
 * Each event card shows the operation name, skin/slot context, params as a
 * key/value list, and input/output thumbnails arranged left → right with an
 * arrow between them. Click a thumbnail to open the full image in a new tab.
 */

type Filter = {
  operation: string;
  skin: string;
  slot: string;
};

const OPERATION_LABELS: Record<string, string> = {
  composite_build: 'Generate · Composite build',
  gemini_inpaint: 'Retouch · Gemini',
  bria_remove_background: 'Retouch · Bria',
  atlas_repack: 'Retouch · Repack',
  mask_save: 'Mask save',
  gemini_full_reskin: 'Generate · Gemini',
  segmentation_sam: 'Rebake · SAM',
  segmentation_bg_components: 'Rebake · BG components',
  region_mask_apply: 'Rebake · region mask',
  atlas_pack: 'Rebake · atlas pack',
};

export function LogsModal({ onClose }: { onClose: () => void }) {
  const [events, setEvents] = useState<LogEvent[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [filter, setFilter] = useState<Filter>({ operation: '', skin: '', slot: '' });

  const load = async () => {
    setErr(null);
    try {
      const r = await api.getLogs(500);
      setEvents(r.events);
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  useEffect(() => { load(); }, []);

  const onClear = async () => {
    if (!window.confirm('Clear all pipeline logs (events + image snapshots)?')) return;
    setBusy(true);
    try {
      await api.clearLogs();
      await load();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const ops = useMemo(() => {
    const set = new Set<string>();
    (events ?? []).forEach((e) => set.add(e.operation));
    return Array.from(set).sort();
  }, [events]);
  const skins = useMemo(() => {
    const set = new Set<string>();
    (events ?? []).forEach((e) => e.skin_name && set.add(e.skin_name));
    return Array.from(set).sort();
  }, [events]);
  const slots = useMemo(() => {
    const set = new Set<string>();
    (events ?? []).forEach((e) => e.slot && set.add(e.slot));
    return Array.from(set).sort();
  }, [events]);

  const filtered = useMemo(() => {
    return (events ?? []).filter((e) => {
      if (filter.operation && e.operation !== filter.operation) return false;
      if (filter.skin && e.skin_name !== filter.skin) return false;
      if (filter.slot && e.slot !== filter.slot) return false;
      return true;
    });
  }, [events, filter]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal modal-wide logs-modal" onClick={(e) => e.stopPropagation()}>
        <header>
          <h2>Pipeline logs</h2>
          <button onClick={onClose}>×</button>
        </header>
        <div className="modal-body">
          <div className="logs-toolbar">
            <select
              value={filter.operation}
              onChange={(e) => setFilter((f) => ({ ...f, operation: e.target.value }))}
            >
              <option value="">All operations</option>
              {ops.map((o) => (
                <option key={o} value={o}>
                  {OPERATION_LABELS[o] ?? o}
                </option>
              ))}
            </select>
            <select
              value={filter.skin}
              onChange={(e) => setFilter((f) => ({ ...f, skin: e.target.value }))}
            >
              <option value="">All looks</option>
              {skins.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
            <select
              value={filter.slot}
              onChange={(e) => setFilter((f) => ({ ...f, slot: e.target.value }))}
            >
              <option value="">All parts</option>
              {slots.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
            <div style={{ flex: 1 }} />
            <span className="muted" style={{ fontSize: 12 }}>
              {events == null ? 'Loading…' : `${filtered.length} of ${events.length}`}
            </span>
            <button onClick={load} disabled={busy}>Refresh</button>
            <button onClick={onClear} disabled={busy || !events?.length}>Clear all</button>
          </div>

          {err && <div className="settings-error">{err}</div>}

          <div className="logs-list">
            {events != null && filtered.length === 0 && (
              <div className="logs-empty">
                {events.length === 0
                  ? 'No events yet. Run Generate, Retouch, or save a mask to populate the log.'
                  : 'No events match the current filters.'}
              </div>
            )}
            {filtered.map((ev) => (
              <LogCard key={ev.id} ev={ev} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function LogCard({ ev }: { ev: LogEvent }) {
  const date = new Date(ev.timestamp * 1000);
  const opLabel = OPERATION_LABELS[ev.operation] ?? ev.operation;
  const isError = ev.status === 'error';
  return (
    <div className={`log-card ${isError ? 'error' : ev.status === 'skipped' ? 'skipped' : ''}`}>
      <div className="log-card-head">
        <div className="log-op">{opLabel}</div>
        <div className="log-meta">
          {ev.skin_name && <span className="log-pill">look: {ev.skin_name}</span>}
          {ev.slot && <span className="log-pill">part: {ev.slot}</span>}
          {ev.duration_ms != null && (
            <span className="log-pill">{Math.round(ev.duration_ms)} ms</span>
          )}
          {ev.status !== 'ok' && <span className={`log-pill log-status-${ev.status}`}>{ev.status}</span>}
          <span className="log-time">{date.toLocaleTimeString()}</span>
        </div>
      </div>
      {ev.error && <div className="log-error">{ev.error}</div>}
      <div className="log-card-body">
        <ImageStrip label="Input" paths={ev.input_images} />
        {(ev.input_images.length > 0 || ev.output_images.length > 0) && <div className="log-arrow">→</div>}
        <ImageStrip label="Output" paths={ev.output_images} />
        <ParamsList params={ev.params} />
      </div>
    </div>
  );
}

function ImageStrip({ label, paths }: { label: string; paths: string[] }) {
  if (paths.length === 0) return <div className="log-strip-empty">No {label.toLowerCase()}</div>;
  return (
    <div className="log-strip">
      <div className="log-strip-label">{label}</div>
      <div className="log-thumbs">
        {paths.map((p) => (
          <a
            key={p}
            href={api.fileUrl(p)}
            target="_blank"
            rel="noreferrer"
            className="log-thumb"
            title={p}
          >
            <img src={api.fileUrl(p)} alt={p} />
          </a>
        ))}
      </div>
    </div>
  );
}

function ParamsList({ params }: { params: Record<string, unknown> }) {
  const entries = Object.entries(params);
  if (entries.length === 0) return null;
  return (
    <div className="log-params">
      <div className="log-strip-label">Params</div>
      <dl>
        {entries.map(([k, v]) => (
          <div key={k} className="log-param">
            <dt>{k}</dt>
            <dd>{formatValue(v)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function formatValue(v: unknown): string {
  if (v == null) return '—';
  if (typeof v === 'string') return v.length > 200 ? v.slice(0, 200) + '…' : v;
  if (typeof v === 'number') return Number.isFinite(v) ? v.toString() : String(v);
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  try {
    const j = JSON.stringify(v);
    return j.length > 200 ? j.slice(0, 200) + '…' : j;
  } catch {
    return String(v);
  }
}
