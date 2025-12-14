"use client";
import React, { useState, useEffect, useMemo } from "react";
import { useFilters } from "./filters-context"

type Props = {
  selectedExts: Set<string>;
  setSelectedExts: React.Dispatch<React.SetStateAction<Set<string>>>;
  selectedTags: string[];
  setSelectedTags: React.Dispatch<React.SetStateAction<string[]>>;
  customTags: string;
  setCustomTags: React.Dispatch<React.SetStateAction<string>>;
}
const PRESET_TAGS = ['sentimental', 'ghost', 'finance', 'documentary', 'fiction', 'policy', 'hr'];

export default function SidebarFilters() {
  const { selectedExts, setSelectedExts, selectedTags, setSelectedTags, customTags, setCustomTags } = useFilters();
  const [model, setModel] = useState("gpt-5.2");
  const [exts, setExts] = useState<string[]>([]);
  const [onlyErrors, setOnlyErrors] = useState(false);
  const filterLabel = useMemo(
    () => (selectedExts.size ? Array.from(selectedExts).map(e => '.' + e).join(', ') : 'All'),
    [selectedExts]
  );

  // load available extensions and tags
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch('/api/files');
        const j = await res.json().catch(() => ({}));
        const uniq = Array.from(
          new Set(
            (j.files ?? [])
              .map((f: any) => String(f.ext || '').replace(/^\./, '').toLowerCase())
              .filter(Boolean)
          )
        ).sort();
        setExts(uniq as string[]);
      } catch {}
    })();
  }, []);

  function toggleExt(ext: string) {
    setSelectedExts(prev => {
      const newExts = new Set(prev);
      newExts.has(ext) ? newExts.delete(ext) : newExts.add(ext);
      return newExts;
    });
  }

  const toggleTag = (tag: string) =>
    setSelectedTags(prev =>
      prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
  );

  return (
    <section className="space-y-3">
      <h3 className="text-sm font-semibold px-1">Filters</h3>
      <div className="flex items-center justify-between">
            <div className="text-xs font-medium text-neutral-600">Filter by extension</div>
            <div className="text-[14px] text-neutral-500">Active: {filterLabel}</div>
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {exts.length === 0 ? (
              <span className="text-xs text-neutral-500">No uploaded files yet.</span>
            ) : (
              exts.map(ext => (
                <label key={ext} className="inline-flex items-center gap-2 text-xs cursor-pointer">
                  <input
                    type="checkbox"
                    className="accent-neutral-900"
                    checked={selectedExts.has(ext)}
                    onChange={() => toggleExt(ext)}
                  />
                  <span className="text-sm">.{ext}</span>
                </label>
              ))
            )}
          </div>
          <div className="space-y-2">
            <label className="block text-xs text-neutral-600 px-1">Tag</label>
            <div className="flex flex-wrap gap-2">
              {PRESET_TAGS.map((t) => (
                <label key={t} className="inline-flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    className="accent-neutral-900"
                    checked={selectedTags.includes(t)}
                    onChange={() => toggleTag(t)}
                  />
                  <span className="text-sm">{t}</span>
                </label>
              ))}
            </div>
            <input
              className="w-full border rounded-xl p-2 text-sm"
              placeholder="Optional extra tags (comma separated)"
              value={customTags}
              onChange={e => setCustomTags(e.target.value)}
            />
          </div>
          <br /><br />
      <div className="space-y-2">
        <label className="block text-xs text-neutral-600 px-1">Model</label>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="w-full rounded-md border px-3 py-2 text-sm"
        >
          <option value="gpt-5.2">GPT-5.2</option>
          <option value="gpt-4o-mini">GPT-4o mini</option>
          <option value="gpt-4o">GPT-4o</option>
          <option value="llama-3.1">Llama 3.1</option>
        </select>
      </div>
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={onlyErrors}
          onChange={(e) => setOnlyErrors(e.target.checked)}
          className="h-4 w-4"
        />
        Only show error turns
      </label>

      <button
        onClick={() => {
          // TODO: wire this to your query/ingest/chat
          // e.g., lift state to context or pass setters via props
        }}
        className="w-full rounded-md bg-neutral-900 text-white text-sm py-2"
      >
        Apply
      </button>
    </section>
  );
}
