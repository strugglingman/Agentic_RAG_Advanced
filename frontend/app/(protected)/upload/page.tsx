'use client';
import { useEffect, useRef, useState, useCallback } from 'react';
import { IngestProgressBar } from '@/components/IngestProgressBar';
import { FileTable, FileRow } from '@/components/FileTable';

const PRESET_TAGS = ['sentimental', 'ghost', 'finance', 'documentary', 'fiction', 'policy', 'hr'];

export default function UploadPage() {
  const [serverFiles, setServerFiles] = useState<FileRow[]>([]);
  const [picked, setPicked] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [checked, setChecked] = useState<Record<string, boolean>>(
    Object.fromEntries(PRESET_TAGS.map(t => [t, false]))
  );
  const [customTags, setCustomTags] = useState('');
  const [fileForUser, setFileForUser] = useState(false);
  const [ingestedCount, setIngestedCount] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Progress bar state
  const [ingestingFileId, setIngestingFileId] = useState<string | null>(null);
  const [activeFileIds, setActiveFileIds] = useState<string[]>([]);

  const refresh = useCallback(async () => {
    const r = await fetch('/api/files');
    const j = await r.json().catch(() => ({}));
    const rows: FileRow[] = (j.files ?? []).map((x: any) => ({
      file_id: x.file_id ?? x.id,
      filename: x.filename ?? x.name,
      file_path: x.file_path ?? '',
      ext: x.ext,
      size_kb: x.size_kb,
      upload_at: x.upload_at,
      tags: x.tags ?? [],
      ingested: x.ingested ?? false,
      file_for_user: x.file_for_user ?? false,
    }));
    setServerFiles(rows);
    setIngestedCount(rows.filter(f => f.ingested).length);
  }, []);

  const refreshActiveFiles = useCallback(async () => {
    try {
      const r = await fetch('/api/ingest/active');
      const j = await r.json().catch(() => ({ file_ids: [] }));
      setActiveFileIds(j.file_ids ?? []);
    } catch {
      setActiveFileIds([]);
    }
  }, []);

  useEffect(() => {
    void refresh();
    void refreshActiveFiles();
  }, [refresh, refreshActiveFiles]);

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
    return Array.from(new Set([...fromChecks, ...fromCustom]));
  }

  async function uploadAll() {
    if (picked.length === 0) return;
    const tags = selectedTags();
    if (tags.length === 0 && !confirm('No tags selected. Proceed?')) return;
    setBusy(true);
    try {
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
          if (r.status === 413) {
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

  // Single file ingest (used by progress bar)
  function startIngest(file_id: string) {
    if (!confirm(`Ingest ${file_id === 'ALL' ? 'all files' : 'this file'}?`)) return;
    setIngestingFileId(file_id);
  }

  // Batch ingest handler for FileTable
  function handleBatchIngest(fileIds: string[]) {
    if (fileIds.length === 0) return;
    const msg = fileIds.length === 1
      ? 'Ingest this file?'
      : `Ingest ${fileIds.length} files?`;
    if (!confirm(msg)) return;
    // For batch, we join IDs with comma as a special marker
    // The progress bar will handle splitting them
    setIngestingFileId(fileIds.length === 1 ? fileIds[0] : fileIds.join(','));
  }

  // Batch delete handler for FileTable
  async function handleBatchDelete(fileIds: string[], removeVectors: boolean) {
    if (fileIds.length === 0) return;
    setBusy(true);
    try {
      const res = await fetch('/api/files/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_ids: fileIds, remove_vectors: removeVectors }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Delete failed');
      }
      alert(`Deleted ${data.total_deleted} file(s)` +
        (data.total_chunks_deleted > 0 ? ` and ${data.total_chunks_deleted} chunks` : ''));
      await refresh();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Delete failed');
    } finally {
      setBusy(false);
    }
  }

  const handleIngestComplete = useCallback(() => {
    setIngestingFileId(null);
    refresh();
    refreshActiveFiles();
  }, [refresh, refreshActiveFiles]);

  const handleIngestError = useCallback((error: string) => {
    setIngestingFileId(null);
    alert(`Ingestion error: ${error}`);
    refresh();
    refreshActiveFiles();
  }, [refresh, refreshActiveFiles]);

  // Real-time update when a single file finishes ingesting
  const handleFileCompleted = useCallback((fileId: string) => {
    setServerFiles(prev => prev.map(f =>
      f.file_id === fileId ? { ...f, ingested: true } : f
    ));
    setIngestedCount(prev => prev + 1);
  }, []);

  return (
    <main className="mx-auto max-w-3xl p-6">
      <div className="border rounded-2xl p-5 bg-white">
        <h2 className="text-xl font-semibold mb-4">1) Upload files</h2>

        <div className="mb-3 text-sm text-gray-600">Select files (.txt, .md, .pdf, .docx, .csv, .json)</div>
        <input ref={fileInputRef} type="file" multiple onChange={onPick} disabled={busy} />
        <label className="inline-flex items-center gap-2 mt-3 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={fileForUser}
            onChange={() => { setFileForUser(!fileForUser) }}
            className="rounded border-gray-300"
          />
          <span className="text-sm font-medium">File specific for the user</span>
        </label>
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
          {busy ? 'Uploading...' : `Upload (${picked.length})`}
        </button>
      </div>

      {/* Existing files list */}
      <div className="mt-6 border rounded-2xl p-5 bg-white">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">2) Uploaded files</h2>
          <button
            className="px-3 py-1.5 text-sm font-medium border rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
            onClick={() => startIngest('ALL')}
            disabled={!!ingestingFileId || serverFiles.length === 0 || busy || ingestedCount === serverFiles.length}
          >
            Ingest All Pending
          </button>
        </div>

        {/* Progress bar */}
        {ingestingFileId && (
          <IngestProgressBar
            fileId={ingestingFileId}
            onComplete={handleIngestComplete}
            onError={handleIngestError}
            onFileCompleted={handleFileCompleted}
          />
        )}

        {/* File table with batch operations */}
        <FileTable
          files={serverFiles}
          activeFileIds={activeFileIds}
          isIngesting={!!ingestingFileId}
          onIngest={handleBatchIngest}
          onDelete={handleBatchDelete}
        />
      </div>
    </main>
  );
}
