'use client';

import { useState, useMemo, useEffect, useRef } from 'react';

export type FileRow = {
  file_id: string;
  filename: string;
  file_path: string;
  size_kb: number;
  ext: string;
  upload_at?: string;
  tags?: string[];
  ingested: boolean;
  file_for_user?: boolean;  // True if user-specific, False if shared
};

interface FileTableProps {
  files: FileRow[];
  activeFileIds: string[];
  isIngesting: boolean;
  onIngest: (fileIds: string[]) => void;
  onDelete: (fileIds: string[], removeVectors: boolean) => void;
}

type SortField = 'filename' | 'size_kb' | 'upload_at' | 'ingested';
type SortDirection = 'asc' | 'desc';

export function FileTable({
  files,
  activeFileIds,
  isIngesting,
  onIngest,
  onDelete,
}: FileTableProps) {
  // Selection state
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Sorting state
  const [sortField, setSortField] = useState<SortField>('upload_at');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  // Filter state
  const [filterStatus, setFilterStatus] = useState<'all' | 'pending' | 'ingested'>('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Delete modal state
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [removeVectors, setRemoveVectors] = useState(true);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 15;

  // Track last clicked index for shift-click range selection
  const lastClickedIndex = useRef<number | null>(null);

  // Filtered and sorted files
  const displayedFiles = useMemo(() => {
    let result = [...files];

    // Apply status filter
    if (filterStatus === 'pending') {
      result = result.filter(f => !f.ingested);
    } else if (filterStatus === 'ingested') {
      result = result.filter(f => f.ingested);
    }

    // Apply search filter (filename and tags only - file_id not shown in UI)
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter(f =>
        f.filename.toLowerCase().includes(query) ||
        (f.tags || []).some(t => t.toLowerCase().includes(query))
      );
    }

    // Apply sorting
    result.sort((a, b) => {
      let cmp = 0;
      switch (sortField) {
        case 'filename':
          cmp = a.filename.localeCompare(b.filename);
          break;
        case 'size_kb':
          cmp = a.size_kb - b.size_kb;
          break;
        case 'upload_at':
          cmp = (a.upload_at || '').localeCompare(b.upload_at || '');
          break;
        case 'ingested':
          cmp = (a.ingested ? 1 : 0) - (b.ingested ? 1 : 0);
          break;
      }
      return sortDirection === 'asc' ? cmp : -cmp;
    });

    return result;
  }, [files, filterStatus, searchQuery, sortField, sortDirection]);

  // Paginated files
  const totalPages = Math.ceil(displayedFiles.length / pageSize);
  const paginatedFiles = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return displayedFiles.slice(start, start + pageSize);
  }, [displayedFiles, currentPage, pageSize]);

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [filterStatus, searchQuery]);

  // Selection helpers
  const allSelected = displayedFiles.length > 0 && displayedFiles.every(f => selectedIds.has(f.file_id));
  const someSelected = selectedIds.size > 0;

  const toggleAll = () => {
    if (allSelected) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(displayedFiles.map(f => f.file_id)));
    }
  };

  const toggleOne = (fileId: string, event: React.MouseEvent) => {
    const currentIndex = displayedFiles.findIndex(f => f.file_id === fileId);

    // Shift-click: select range from last clicked to current
    if (event.shiftKey && lastClickedIndex.current !== null) {
      const start = Math.min(lastClickedIndex.current, currentIndex);
      const end = Math.max(lastClickedIndex.current, currentIndex);
      const next = new Set(selectedIds);

      for (let i = start; i <= end; i++) {
        next.add(displayedFiles[i].file_id);
      }
      setSelectedIds(next);
    } else {
      // Normal click: toggle single item
      const next = new Set(selectedIds);
      if (next.has(fileId)) {
        next.delete(fileId);
      } else {
        next.add(fileId);
      }
      setSelectedIds(next);
    }

    lastClickedIndex.current = currentIndex;
  };

  // Sort handler
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  // Get selected files that can be ingested (not already ingested, not in progress)
  const selectedForIngest = useMemo(() => {
    return Array.from(selectedIds).filter(id => {
      const file = files.find(f => f.file_id === id);
      return file && !file.ingested && !activeFileIds.includes(id);
    });
  }, [selectedIds, files, activeFileIds]);

  // Get selected files for deletion
  const selectedForDelete = useMemo(() => {
    return Array.from(selectedIds).filter(id => {
      // Can't delete files being ingested
      return !activeFileIds.includes(id);
    });
  }, [selectedIds, activeFileIds]);

  // Action handlers
  const handleIngestSelected = () => {
    if (selectedForIngest.length === 0) return;
    onIngest(selectedForIngest);
    setSelectedIds(new Set());
  };

  const handleDeleteSelected = () => {
    if (selectedForDelete.length === 0) return;
    setShowDeleteModal(true);
  };

  const confirmDelete = () => {
    onDelete(selectedForDelete, removeVectors);
    setSelectedIds(new Set());
    setShowDeleteModal(false);
  };

  // Status badge component
  const StatusBadge = ({ file }: { file: FileRow }) => {
    if (activeFileIds.includes(file.file_id)) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-700">
          <span className="w-1.5 h-1.5 rounded-full bg-yellow-500 animate-pulse" />
          In Progress
        </span>
      );
    }
    if (file.ingested) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-700">
          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
          Ingested
        </span>
      );
    }
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
        Pending
      </span>
    );
  };

  // Sort indicator
  const SortIndicator = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return (
      <span className="ml-1">
        {sortDirection === 'asc' ? '↑' : '↓'}
      </span>
    );
  };

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Search */}
        <div className="relative flex-1 min-w-[200px]">
          <input
            type="text"
            placeholder="Search files..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-2 text-sm border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <svg className="absolute left-3 top-2.5 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>

        {/* Status filter */}
        <div className="flex rounded-lg border overflow-hidden">
          {(['all', 'pending', 'ingested'] as const).map(status => (
            <button
              key={status}
              onClick={() => setFilterStatus(status)}
              className={`px-3 py-1.5 text-sm font-medium transition-colors ${
                filterStatus === status
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-600 hover:bg-gray-50'
              }`}
            >
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </button>
          ))}
        </div>

        {/* File count */}
        <div className="text-sm text-gray-500">
          {displayedFiles.length} file{displayedFiles.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Batch action bar - fixed height container to prevent layout shift */}
      <div className="h-[52px] flex items-center">
        <div className={`flex-1 flex items-center gap-3 p-3 rounded-lg transition-all duration-150 ${
          someSelected
            ? 'bg-blue-50 border border-blue-200 opacity-100'
            : 'opacity-0 pointer-events-none'
        }`}>
          <span className="text-sm font-medium text-blue-700">
            {selectedIds.size} selected
          </span>
          <div className="flex-1" />
          <button
            onClick={handleIngestSelected}
            disabled={selectedForIngest.length === 0 || isIngesting}
            className="px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Ingest ({selectedForIngest.length})
          </button>
          <button
            onClick={handleDeleteSelected}
            disabled={selectedForDelete.length === 0}
            className="px-3 py-1.5 text-sm font-medium text-white bg-red-600 rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Delete ({selectedForDelete.length})
          </button>
          <button
            onClick={() => setSelectedIds(new Set())}
            className="px-3 py-1.5 text-sm font-medium text-gray-600 bg-white border rounded-lg hover:bg-gray-50"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="border rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b">
            <tr>
              <th className="w-10 px-3 py-3">
                <input
                  type="checkbox"
                  checked={allSelected}
                  onChange={toggleAll}
                  className="rounded border-gray-300"
                />
              </th>
              <th
                className="px-3 py-3 text-left font-medium text-gray-600 cursor-pointer hover:text-gray-900"
                onClick={() => handleSort('filename')}
              >
                File <SortIndicator field="filename" />
              </th>
              <th
                className="px-3 py-3 text-left font-medium text-gray-600 cursor-pointer hover:text-gray-900 w-24"
                onClick={() => handleSort('size_kb')}
              >
                Size <SortIndicator field="size_kb" />
              </th>
              <th
                className="px-3 py-3 text-left font-medium text-gray-600 cursor-pointer hover:text-gray-900 w-32"
                onClick={() => handleSort('upload_at')}
              >
                Uploaded <SortIndicator field="upload_at" />
              </th>
              <th
                className="px-3 py-3 text-left font-medium text-gray-600 cursor-pointer hover:text-gray-900 w-28"
                onClick={() => handleSort('ingested')}
              >
                Status <SortIndicator field="ingested" />
              </th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {paginatedFiles.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-3 py-8 text-center text-gray-500">
                  {files.length === 0 ? 'No files uploaded yet' : 'No files match your filters'}
                </td>
              </tr>
            ) : (
              paginatedFiles.map(file => (
                <tr
                  key={file.file_id}
                  className={`hover:bg-gray-50 ${selectedIds.has(file.file_id) ? 'bg-blue-50' : ''}`}
                >
                  <td className="px-3 py-3">
                    <input
                      type="checkbox"
                      checked={selectedIds.has(file.file_id)}
                      onClick={(e) => toggleOne(file.file_id, e)}
                      onChange={() => {}} // Controlled by onClick for shift-click support
                      className="rounded border-gray-300"
                    />
                  </td>
                  <td className="px-3 py-3">
                    <div className="flex items-center gap-2">
                      {/* File type icon */}
                      <div className="w-8 h-8 flex items-center justify-center rounded bg-gray-100 text-gray-500 text-xs font-medium uppercase">
                        {file.ext}
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-gray-900 truncate" title={file.filename}>
                            {file.filename}
                          </span>
                          {file.file_for_user && (
                            <span className="inline-flex items-center px-1.5 py-0.5 text-xs font-medium bg-purple-100 text-purple-700 rounded" title="User-specific file">
                              Private
                            </span>
                          )}
                        </div>
                        {file.tags && file.tags.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-1">
                            {file.tags.slice(0, 4).map(tag => (
                              <span key={tag} className="px-1.5 py-0.5 text-xs bg-blue-50 text-blue-700 rounded">
                                {tag}
                              </span>
                            ))}
                            {file.tags.length > 4 && (
                              <span className="px-1.5 py-0.5 text-xs bg-gray-100 text-gray-600 rounded">
                                +{file.tags.length - 4}
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </td>
                  <td className="px-3 py-3 text-gray-600">
                    {file.size_kb < 1024
                      ? `${file.size_kb.toFixed(1)} KB`
                      : `${(file.size_kb / 1024).toFixed(1)} MB`
                    }
                  </td>
                  <td className="px-3 py-3 text-gray-600">
                    {file.upload_at
                      ? new Date(file.upload_at).toLocaleDateString()
                      : '-'
                    }
                  </td>
                  <td className="px-3 py-3">
                    <StatusBadge file={file} />
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-2">
          <div className="text-sm text-gray-500">
            Showing {((currentPage - 1) * pageSize) + 1}-{Math.min(currentPage * pageSize, displayedFiles.length)} of {displayedFiles.length}
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setCurrentPage(1)}
              disabled={currentPage === 1}
              className="px-2 py-1 text-sm border rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              First
            </button>
            <button
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="px-2 py-1 text-sm border rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Prev
            </button>
            <span className="px-3 py-1 text-sm">
              Page {currentPage} of {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="px-2 py-1 text-sm border rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
            <button
              onClick={() => setCurrentPage(totalPages)}
              disabled={currentPage === totalPages}
              className="px-2 py-1 text-sm border rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Last
            </button>
          </div>
        </div>
      )}

      {/* Delete confirmation modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white rounded-xl shadow-xl max-w-md w-full mx-4 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Delete {selectedForDelete.length} file{selectedForDelete.length !== 1 ? 's' : ''}?
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              This action cannot be undone. The files will be permanently deleted from the server.
            </p>

            <label className="flex items-center gap-2 p-3 bg-gray-50 rounded-lg mb-4 cursor-pointer">
              <input
                type="checkbox"
                checked={removeVectors}
                onChange={e => setRemoveVectors(e.target.checked)}
                className="rounded border-gray-300"
              />
              <div>
                <div className="text-sm font-medium text-gray-900">
                  Also remove from vector database
                </div>
                <div className="text-xs text-gray-500">
                  Remove indexed chunks from ChromaDB
                </div>
              </div>
            </label>

            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowDeleteModal(false)}
                className="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                className="px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-lg hover:bg-red-700"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
