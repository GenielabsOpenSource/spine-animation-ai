import { useEffect, useRef, useState } from 'react';
import { useStore } from '../state/store';
import { api, type Project } from '../api/client';

function originalThumbUrl(project: Project, slotName: string): string {
  // Prefer a PNG named after the slot; else fall back to the slot's
  // default-skin attachment region (post-explode projects use that mapping).
  if (project.slots.find((s) => s.name === slotName && s.has_part_png)) {
    return api.fileUrl(`${slotName}.png`);
  }
  const att = project.default_attachments?.[slotName];
  if (att) {
    const regionName = Object.values(att)[0]?.name as string | undefined;
    if (regionName) return api.fileUrl(`${regionName}.png`);
    const attachmentKey = Object.keys(att)[0];
    if (attachmentKey) return api.fileUrl(`${attachmentKey}.png`);
  }
  return api.fileUrl(`${slotName}.png`);
}

export function Sidebar() {
  const project = useStore((s) => s.project);
  const selected = useStore((s) => s.selectedSlot);
  const select = useStore((s) => s.selectSlot);
  const hidden = useStore((s) => s.hidden);
  const toggleHidden = useStore((s) => s.toggleHidden);
  const activeSkin = useStore((s) => s.activeSkin);
  const refreshProjectSkins = useStore((s) => s.refreshProjectSkins);
  const bumpAssetVersion = useStore((s) => s.bumpAssetVersion);
  const assetVersion = useStore((s) => s.assetVersion);
  const [busy, setBusy] = useState<string | null>(null);
  const [bulkBusy, setBulkBusy] = useState(false);
  const listRef = useRef<HTMLUListElement>(null);

  useEffect(() => {
    if (!selected || !listRef.current) return;
    const el = listRef.current.querySelector<HTMLLIElement>(
      `li[data-slot="${CSS.escape(selected)}"]`,
    );
    if (!el) return;
    const scroller = listRef.current.parentElement as HTMLElement | null;  // .sidebar
    if (!scroller) return;
    const target = el.offsetTop - scroller.clientHeight / 2 + el.offsetHeight / 2;
    const max = scroller.scrollHeight - scroller.clientHeight;
    scroller.scrollTop = Math.max(0, Math.min(max, target));
  }, [selected]);

  if (!project) {
    return (
      <aside className="sidebar">
        <div className="empty">Open a project to see its parts.</div>
      </aside>
    );
  }

  const revertedForSkin = new Set(project.reverted_slots?.[activeSkin] ?? []);

  const thumbUrl = (slotName: string) => {
    if (activeSkin === 'default' || revertedForSkin.has(slotName)) {
      return originalThumbUrl(project, slotName);
    }
    return `${api.fileUrl(`.genie/skins/${activeSkin}/extracted/${slotName}.png`)}?v=${assetVersion}`;
  };

  const onSwap = async (slotName: string) => {
    if (activeSkin === 'default') return;
    const isReverted = revertedForSkin.has(slotName);
    setBusy(slotName);
    try {
      await api.revertSlots(activeSkin, [slotName], !isReverted);
      const fresh = await api.getStatus();
      if (fresh.open) refreshProjectSkins(fresh as Project);
      bumpAssetVersion();
    } catch (e) {
      alert(`swap failed: ${(e as Error).message}`);
    } finally {
      setBusy(null);
    }
  };

  const allReverted =
    project.slots.length > 0 &&
    project.slots.every((s) => revertedForSkin.has(s.name));

  const onResetAll = async () => {
    if (activeSkin === 'default' || bulkBusy) return;
    setBulkBusy(true);
    try {
      const names = project.slots.map((s) => s.name);
      await api.revertSlots(activeSkin, names, !allReverted);
      const fresh = await api.getStatus();
      if (fresh.open) refreshProjectSkins(fresh as Project);
      bumpAssetVersion();
    } catch (e) {
      alert(`reset all failed: ${(e as Error).message}`);
    } finally {
      setBulkBusy(false);
    }
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h3>Parts</h3>
        <div className="sidebar-header-actions">
          {activeSkin !== 'default' && (
            <button
              className="reset-all-btn"
              onClick={onResetAll}
              disabled={bulkBusy}
              title={allReverted
                ? 'Restore every part to the reskinned version'
                : 'Reset every part to the original — then restore the parts you want'}
            >
              {bulkBusy ? '…' : allReverted ? 'Restore all' : 'Reset all'}
            </button>
          )}
          <span className="muted">{project.slots.length}</span>
        </div>
      </div>
      <ul className="slot-list" ref={listRef}>
        {project.slots.map((slot) => {
          const isSel = slot.name === selected;
          const isHidden = hidden.has(slot.name);
          const isReverted = revertedForSkin.has(slot.name);
          const swapDisabled = activeSkin === 'default' || busy === slot.name || bulkBusy;
          return (
            <li
              key={slot.name}
              data-slot={slot.name}
              className={`slot ${isSel ? 'selected' : ''} ${isHidden ? 'hidden' : ''} ${isReverted ? 'reverted' : ''}`}
              onClick={() => select(slot.name)}
            >
              <img
                className="thumb"
                src={thumbUrl(slot.name)}
                alt={slot.name}
                onError={(e) => ((e.target as HTMLImageElement).style.opacity = '0.2')}
              />
              <span className="slot-name">{slot.name}</span>
              <button
                className={`icon-btn ${isReverted ? 'active' : ''}`}
                title={
                  activeSkin === 'default'
                    ? 'Pick a generated look to swap parts'
                    : isReverted
                      ? 'Restore reskinned part'
                      : 'Swap with original part'
                }
                disabled={swapDisabled}
                onClick={(e) => { e.stopPropagation(); onSwap(slot.name); }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <polyline points="1 4 1 10 7 10" />
                  <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
                </svg>
              </button>
              <button
                className="icon-btn"
                title={isHidden ? 'Show part' : 'Hide part'}
                onClick={(e) => { e.stopPropagation(); toggleHidden(slot.name); }}
              >
                <img
                  src={isHidden ? '/icons/frame-background-transparent.svg' : '/icons/frame.svg'}
                  alt=""
                />
              </button>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
