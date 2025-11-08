'use client';
import { useEffect, useRef, useState } from 'react';

type FileRow = {
  file_id: string;     // backend response key
  filename: string;
  file_path: string;
  size_kb: number;
  ext: string;
  upload_at?: string;
  tags?: string[];
  ingested: boolean;
};

const PRESET_TAGS = ['sentimental', 'ghost', 'finance', 'documentary', 'fiction', 'policy', 'hr'];

export default function UploadPage() {
  const [serverFiles, setServerFiles] = useState<FileRow[]>([]);
  const [picked, setPicked] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [checked, setChecked] = useState<Record<string, boolean>>(
    Object.fromEntries(PRESET_TAGS.map(t => [t, false]))
  );
  const [customTags, setCustomTags] = useState(''); // comma separated
  const [busyIngesting, setIngestBusy] = useState(false);
  const [fileForUser, setFileForUser] = useState(false);
  const [ingestedCount, setIngestedCount] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = async () => {
    const r = await fetch('/api/files');
    const j = await r.json().catch(() => ({}));
    // Normalize keys if your /files returns {id,name} instead:
    const rows: FileRow[] = (j.files ?? []).map((x: any) => ({
      file_id: x.file_id ?? x.id,
      filename: x.filename ?? x.name,
      file_path: x.file_path ?? '',
      ext: x.ext,
      size_kb: x.size_kb,
      upload_at: x.upload_at,
      tags: x.tags ?? [],
      ingested: x.ingested ?? false,
    }));
    setServerFiles(rows);

    const ingestedFiles = rows.filter(f => f.ingested).length;
    setIngestedCount(ingestedFiles);
  };

  useEffect(() => { void refresh(); }, []);

  function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.currentTarget.files ? Array.from(e.currentTarget.files) : [];
    setPicked(files);
  }

  function selectedTags(): string[] {
    const fromChecks = PRESET_TAGS.filter(t => checked[t]);
    const fromCustom = customTags
      .split(',')
      .map(s => s.trim())
      .filter(Boolean);
    // unique
    return Array.from(new Set([...fromChecks, ...fromCustom]));
  }

  async function uploadAll() {
    if (picked.length === 0) return;
    const tags = selectedTags();
    if (tags.length === 0 && !confirm('No tags selected. Proceed?')) return;
    setBusy(true);
    try {
      // sequential to be gentle on the server; switch to Promise.all for parallel
      for (const f of picked) {
        if (f.size > (Number(process.env.NEXT_PUBLIC_UPLOAD_FILE_LIMIT_MB) || 25) * 1024 * 1024) {
          throw new Error(`File must be < 25MB: ${f.name}`);
        }

        const form = new FormData();
        form.append('file', f);
        form.append('file_for_user', fileForUser ? '1' : '0');
        form.append('tags', JSON.stringify(tags));
        const r = await fetch('/api/upload', {
          method: 'POST',
          body: form,
        });
        if (!r.ok) {
          if(r.status === 413) {
            throw new Error(`File too large: ${f.name}`);
          } else {
            const txt = await r.text().catch(() => '');
            throw new Error(txt || `Upload failed (${r.status})`);
          }
        }
      }
      await refresh();
      setPicked([]);
      if (fileInputRef.current) fileInputRef.current.value = '';
      setCustomTags('');
      setChecked(Object.fromEntries(PRESET_TAGS.map(t => [t, false])));

    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function toggleTag(tag: string) {
    setChecked(prev => ({ ...prev, [tag]: !prev[tag] }));
  }

  async function ingest(file_id: string, file_path: string) {
    if (!confirm(`Ingest file_id=${file_id} ?`)) return;

    setIngestBusy(true);
    const r = await fetch('/api/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ "file_id": file_id, "file_path": file_path }),
    });

    if (!r.ok) {
      const txt = await r.text().catch(() => "");
      alert(txt || `Ingest failed (${r.status})`);
    }
    else {
      const j = await r.json().catch(() => ({}));
      alert(j.message ?? 'Ingest completed.');

      refresh();
    }
    setIngestBusy(false);
  }

  return (
    <main className="mx-auto max-w-3xl p-6">
      <div className="border rounded-2xl p-5 bg-white">
        <h2 className="text-xl font-semibold mb-4">1) Upload files</h2>

        <div className="mb-3 text-sm text-gray-600">Select files (.txt, .md, .pdf, .docx, .csv, .json)</div>
        <input ref={fileInputRef} type="file" multiple onChange={onPick} disabled={busy} />
        <label className="text-sm font-medium mb-2">File specific for the user:</label>
        <input
          type="checkbox"
          checked={fileForUser}
          onChange={() => { setFileForUser(!fileForUser) }}
        />
        <br /><br />
        <div className="mt-4">
          <div className="text-sm font-medium mb-2">Tags for this upload:</div>
          <div className="flex flex-wrap gap-3 mb-3">
            {PRESET_TAGS.map(tag => (
              <label key={tag} className="inline-flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={!!checked[tag]}
                  onChange={() => toggleTag(tag)}
                  disabled={busy}
                />
                <span className="text-sm">{tag}</span>
              </label>
            ))}
          </div>
          <input
            className="w-full border rounded-xl p-2 text-sm"
            placeholder="Optional extra tags (comma separated)"
            value={customTags}
            onChange={e => setCustomTags(e.target.value)}
            disabled={busy}
          />
          <div className="text-xs text-gray-500 mt-1">These tags apply to all selected files in this batch.</div>
        </div>
        <br /><br />
        <div className="mt-4 text-xs text-gray-500">Files are saved to <code>./uploads</code>.</div>

        <button
          className="mt-4 px-4 py-2 rounded-xl bg-blue-600 text-white disabled:opacity-50"
          onClick={uploadAll}
          disabled={busy || picked.length === 0}
        >
          {busy ? 'Uploading…' : `Upload (${picked.length})`}
        </button>
      </div>

      {/* Existing files list */}
      <div className="mt-6 border rounded-2xl p-5 bg-gray-50">
        <h2 className="text-lg font-semibold mb-3">2) Uploaded files</h2>
        {serverFiles.length === 0 ? (
          <div className="text-sm text-gray-500">No files yet.</div>
        ) : (
          serverFiles.map(f => (
            <div key={f.file_id} className="flex items-center justify-between py-2 border-b last:border-b-0">
              <div>
                <div className="text-sm">{f.file_id}</div>
                <div className="text-sm">{f.filename}</div>
                <div className="text-sm">{f.file_path}</div>
                <div className="text-xs text-gray-500">
                  {f.size_kb.toFixed(1)} KB · {f.ext}
                </div>
                <div className="text-sm">
                  {f.tags && f.tags.length > 0 && (
                    <span className="ml-2">Tags: {f.tags}</span>
                  )}
                </div>
                <div className="text-sm">{f.upload_at}</div>
              </div>
              {/* You can keep your ingest button here if needed */}
              <button
                className="px-3 py-1 text-sm border rounded-lg bg-white disabled:opacity-50"
                onClick={() => ingest(f.file_id, f.file_path)}
                disabled={busyIngesting || serverFiles.length === 0 || busy || f.ingested}
              >
                Ingest
              </button>
            </div>
          ))
        )}
        <div>
          <br />
          <button
            className="px-3 py-1 text-sm border rounded-lg bg-white disabled:opacity-50"
            onClick={() => ingest('ALL', 'ALL')}
            disabled={busyIngesting || serverFiles.length === 0 || busy || ingestedCount === serverFiles.length}
          >
            Ingest All
          </button>
        </div>
      </div>
    </main>
  );
}
