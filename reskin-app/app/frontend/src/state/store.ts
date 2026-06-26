import { create } from 'zustand';
import { api, type Project, type ReskinMethod, type SecretField, type SlotEdit } from '../api/client';

export type KeyPrompt = {
  actionLabel: string;
  missing: { name: string; label: string; help_url: string }[];
};

export type AppMode =
  | { kind: 'idle' }
  | { kind: 'generating'; skinName: string }
  | { kind: 'review'; skinName: string; reskinnedUrl: string };

const RESKIN_METHOD_STORAGE_KEY = 'genie.reskin.method';
const VALID_METHODS: ReskinMethod[] = ['atlas', 'exploded'];

function loadReskinMethod(): ReskinMethod {
  try {
    const v = localStorage.getItem(RESKIN_METHOD_STORAGE_KEY);
    if (v && (VALID_METHODS as string[]).includes(v)) return v as ReskinMethod;
  } catch { /* localStorage unavailable */ }
  return 'atlas';
}

export type CanvasTool = 'select' | 'hand' | 'scale' | 'rotate';

type Store = {
  project: Project | null;
  activeSkin: string;          // 'default' or a generated skin's name
  activeAnimation: string | null;  // null = rest pose; otherwise an animation name
  mode: AppMode;
  reskinMethod: ReskinMethod;
  assetVersion: number;        // bumped to force canvas + thumbnails to reload
  tool: CanvasTool;

  selectedSlot: string | null;
  hidden: Set<string>;

  edits: Record<string, SlotEdit>;

  setProject: (p: Project) => void;
  refreshProjectSkins: (p: Project) => void;
  setActiveSkin: (name: string) => void;
  setActiveAnimation: (name: string | null) => void;
  setMode: (m: AppMode) => void;
  setReskinMethod: (m: ReskinMethod) => void;
  bumpAssetVersion: () => void;
  setTool: (t: CanvasTool) => void;

  selectSlot: (s: string | null) => void;
  toggleHidden: (s: string) => void;

  setEdit: (slot: string, e: SlotEdit) => void;
  clearEdit: (slot: string) => void;
  clearAllEdits: () => void;

  secrets: SecretField[] | null;
  keyPrompt: KeyPrompt | null;
  refreshSecrets: () => Promise<void>;
  // Returns true if every named secret is set; otherwise opens the key prompt
  // and returns false. Unknown (not yet loaded) → permissive.
  ensureSecrets: (names: string[], actionLabel: string) => boolean;
  setKeyPrompt: (p: KeyPrompt | null) => void;
};

export const useStore = create<Store>((set, get) => ({
  project: null,
  activeSkin: 'default',
  activeAnimation: null,
  mode: { kind: 'idle' },
  reskinMethod: loadReskinMethod(),
  assetVersion: 0,
  tool: 'select',

  selectedSlot: null,
  hidden: new Set(),

  edits: {},

  setProject: (project) =>
    set({
      project,
      activeSkin: 'default',
      activeAnimation: project.animations && project.animations.length > 0
        ? project.animations[0]
        : null,
      selectedSlot: null,
      edits: {},
      hidden: new Set(),
      mode: { kind: 'idle' },
    }),
  refreshProjectSkins: (project) => set({ project }),
  setActiveSkin: (activeSkin) => set({ activeSkin }),
  setActiveAnimation: (activeAnimation) => set({ activeAnimation }),
  setMode: (mode) => set({ mode }),
  setReskinMethod: (reskinMethod) => {
    try { localStorage.setItem(RESKIN_METHOD_STORAGE_KEY, reskinMethod); } catch { /* ignore */ }
    set({ reskinMethod });
  },
  bumpAssetVersion: () => set((s) => ({ assetVersion: s.assetVersion + 1 })),
  setTool: (tool) => set({ tool }),

  selectSlot: (selectedSlot) => set({ selectedSlot }),
  toggleHidden: (s) =>
    set((state) => {
      const next = new Set(state.hidden);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return { hidden: next };
    }),

  setEdit: (slot, e) =>
    set((state) => ({ edits: { ...state.edits, [slot]: e } })),
  clearEdit: (slot) =>
    set((state) => {
      const { [slot]: _, ...rest } = state.edits;
      return { edits: rest };
    }),
  clearAllEdits: () => set({ edits: {} }),

  secrets: null,
  keyPrompt: null,
  setKeyPrompt: (keyPrompt) => set({ keyPrompt }),
  refreshSecrets: async () => {
    try {
      const secrets = await api.getSecrets();
      set({ secrets });
    } catch {
      /* couldn't read — leave as-is; gates stay permissive */
    }
  },
  ensureSecrets: (names, actionLabel) => {
    const secrets = get().secrets;
    if (!secrets) return true;  // not loaded yet → don't block
    const missing = secrets.filter((s) => names.includes(s.name) && !s.is_set);
    if (missing.length === 0) return true;
    set({ keyPrompt: {
      actionLabel,
      missing: missing.map((s) => ({ name: s.name, label: s.label, help_url: s.help_url })),
    } });
    return false;
  },
}));
