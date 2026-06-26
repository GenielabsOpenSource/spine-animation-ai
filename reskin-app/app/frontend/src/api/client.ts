/* Typed fetch client for the Genie Reskin backend. */

export type SlotInfo = {
  name: string;
  bone: string;
  attachment: string | null;
  has_part_png: boolean;
};

export type AttachmentMeta = {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  rotation?: number;
  scaleX?: number;
  scaleY?: number;
  [k: string]: unknown;
};

export type Project = {
  path: string;
  name: string;
  spine_json: string;
  atlas: string | null;
  sheet: string | null;
  skins: string[];
  slots: SlotInfo[];
  default_attachments: Record<string, Record<string, AttachmentMeta>>;
  reverted_slots?: Record<string, string[]>;  // skin name → reverted slot names
  transforms?: Record<string, Record<string, SlotTransform>>;
  animations?: string[];
};

export type SlotTransform = {
  // All fields optional. Position/rotation are deltas added to the
  // default-skin attachment values; scale is a multiplicative factor.
  x?: number;
  y?: number;
  rotation?: number;  // degrees, additive
  scale?: number;     // multiplier on scaleX and scaleY (default 1)
};

export type ProjectCandidate = {
  base: string;
  display_name: string;
  path: string;
};

export type MultiProjectChoice = {
  multi_project: true;
  folder: string;
  candidates: ProjectCandidate[];
};

export function isMultiProjectChoice(
  v: Project | MultiProjectChoice,
): v is MultiProjectChoice {
  return (v as MultiProjectChoice).multi_project === true;
}

export type ReskinMethod = 'atlas' | 'exploded';

export type GenerateLayout = {
  composite_w: number;
  composite_h: number;
  mode?: ReskinMethod;
  // atlas mode
  snapshot_rect?: { x: number; y: number; w: number; h: number };
  atlas_rect?: { x: number; y: number; w: number; h: number };
  atlas_scale?: number;
  atlas_sheet_name?: string;
  atlas_meta_name?: string;
  // exploded mode
  padding?: number;
  placements?: Record<string, { x: number; y: number; w: number; h: number }>;
};

export type GenerateResponse = {
  skin_name: string;
  method: ReskinMethod;
  composite: string;
  reskinned_composite: string;
  reskinned_snapshot?: string;  // atlas mode only
  reskinned_atlas?: string;     // atlas mode only
  layout: GenerateLayout;
};

export type RebakeResponse = {
  saved: string[];
  atlas: string;
  atlas_image: string;
  skin_spine_json: string;
  sam_used: boolean;
  masks_count: number;
};

export type SlotEdit = {
  hue_shift?: number;
  sat_mult?: number;
  light_shift?: number;
  brightness?: number;
  contrast?: number;
  rgb_balance?: [number, number, number];
  dx?: number;
  dy?: number;
  rotation?: number;
  scale?: number;
};

export type ExportResponse = {
  spine_json: string;
  atlas_image: string;
  atlas_meta: string;
  skin_dir: string;
  files_copied: string[];
};

export type ErosionSettings = {
  enabled: boolean;
  px_small: number;
  px_medium: number;
  px_large: number;
  px_xlarge: number;
  small_threshold: number;
  medium_threshold: number;
  large_threshold: number;
};

export type ReferenceSettings = {
  enabled: boolean;
  prompt: string;
  has_image: boolean;
};

export type SegmentationMethod = 'sam' | 'bg_components';

export type SegmentationSettings = {
  method: SegmentationMethod;
};

export type AppSettings = {
  erosion: ErosionSettings;
  reference: ReferenceSettings;
  segmentation: SegmentationSettings;
};

async function jpost<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  openProject: async (path: string): Promise<Project | MultiProjectChoice> => {
    const r = await fetch('/api/project/open', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    if (r.status === 409) {
      const body = await r.json();
      const choice = body && body.detail ? body.detail : body;
      if (choice && choice.multi_project) return choice as MultiProjectChoice;
    }
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return (await r.json()) as Project;
  },
  getStatus: async (): Promise<{ open: boolean } & Partial<Project>> => {
    const r = await fetch('/api/project/status');
    return r.json();
  },
  fileUrl: (rel: string) =>
    `/api/project/file/${rel.split('/').map(encodeURIComponent).join('/')}`,
  generate: (skin_name: string, prompt: string, method: ReskinMethod = 'atlas') =>
    jpost<GenerateResponse>('/api/reskin/generate', { skin_name, prompt, method }),
  uploadSnapshot: async (pngBlob: Blob, skinName: string) => {
    const r = await fetch(`/api/project/snapshot?skin_name=${encodeURIComponent(skinName)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'image/png' },
      body: pngBlob,
    });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  },
  rebake: (skin_name: string) =>
    jpost<RebakeResponse>('/api/reskin/rebake', { skin_name }),
  revertSlots: (skin_name: string, slots: string[], revert: boolean) =>
    jpost<{ ok: boolean; atlas: string; skin_spine_json: string; reverted_slots: string[] }>(
      `/api/skin/${encodeURIComponent(skin_name)}/revert-slots`,
      { slots, revert },
    ),
  putTransforms: async (skin_name: string, transforms: Record<string, SlotTransform>) => {
    const r = await fetch(`/api/skin/${encodeURIComponent(skin_name)}/transforms`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(transforms),
    });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  },
  resetTransforms: async (skin_name: string) => {
    const r = await fetch(`/api/skin/${encodeURIComponent(skin_name)}/transforms`, {
      method: 'DELETE',
    });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  },
  inpaintSlot: (skin_name: string, slot: string, prompt: string) =>
    jpost<{ok: boolean; slot: string; atlas: string; skin_spine_json: string}>(
      '/api/skin/inpaint-slot',
      { skin_name, slot, prompt },
    ),
  editPreviewUrl: (slot: string, skin_name: string, edit: SlotEdit) => {
    /* Edit preview is a POST that returns an image. Components do their own
       fetch + URL.createObjectURL to get a blob URL. */
    return { slot, skin_name, edit };
  },
  fetchEditPreview: async (slot: string, skin_name: string, edit: SlotEdit): Promise<Blob> => {
    const r = await fetch('/api/skin/edit-preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slot, skin_name, edit }),
    });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.blob();
  },
  exportSkin: (skin_name: string, edits: Record<string, SlotEdit>, write_into_main_json = false) =>
    jpost<ExportResponse>('/api/skin/export', { skin_name, edits, write_into_main_json }),
  getSettings: async (): Promise<AppSettings> => {
    const r = await fetch('/api/settings');
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  },
  putSettings: async (s: AppSettings): Promise<AppSettings> => {
    const r = await fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(s),
    });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  },
  getSecrets: async (): Promise<SecretField[]> => {
    const r = await fetch('/api/secrets');
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return (await r.json()).secrets;
  },
  putSecrets: async (updates: Record<string, string>): Promise<SecretField[]> => {
    const r = await fetch('/api/secrets', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ updates }),
    });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return (await r.json()).secrets;
  },
  referenceImageUrl: (cacheBust?: number) =>
    `/api/settings/reference-image${cacheBust ? `?v=${cacheBust}` : ''}`,
  uploadReferenceImage: async (blob: Blob): Promise<{ ok: boolean; size: number }> => {
    const r = await fetch('/api/settings/reference-image', {
      method: 'PUT',
      headers: { 'Content-Type': blob.type || 'image/png' },
      body: blob,
    });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  },
  deleteReferenceImage: async (): Promise<{ ok: boolean }> => {
    const r = await fetch('/api/settings/reference-image', { method: 'DELETE' });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  },
  getLogs: async (limit = 500): Promise<{ events: LogEvent[] }> => {
    const r = await fetch(`/api/logs?limit=${limit}`);
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  },
  clearLogs: async (): Promise<{ ok: boolean; cleared: number }> => {
    const r = await fetch('/api/logs', { method: 'DELETE' });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  },
};

export type SecretField = {
  name: string;
  label: string;
  required: boolean;
  kind: 'key' | 'url';
  help_url: string;
  description: string;
  value: string;
  is_set: boolean;
  from_env: boolean;
};

export type LogEvent = {
  id: string;
  timestamp: number;
  operation: string;
  skin_name: string | null;
  slot: string | null;
  params: Record<string, unknown>;
  input_images: string[];
  output_images: string[];
  duration_ms: number | null;
  status: string;
  error: string | null;
  notes: string | null;
};
