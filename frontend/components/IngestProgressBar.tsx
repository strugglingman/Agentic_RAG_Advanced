'use client';

import { useEffect, useState, useCallback } from 'react';

interface ProgressData {
  job_id: string;
  current_file?: string;
  current_index?: number;
  total_files: number;
  total_chunks?: number;
  ingested_files?: number;
  processed_files?: number;
  status: string;
  error?: string;
  message?: string;
  completed_file_id?: string;
}

interface IngestProgressBarProps {
  fileId: string;
  onComplete: () => void;
  onError: (error: string) => void;
  onFileCompleted?: (fileId: string) => void;
}

export function IngestProgressBar({ fileId, onComplete, onError, onFileCompleted }: IngestProgressBarProps) {
  const [progress, setProgress] = useState<ProgressData | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isConnected, setIsConnected] = useState(false);

  // Start SSE connection
  useEffect(() => {
    const controller = new AbortController();

    async function startIngestion() {
      try {
        // Convert fileId to file_ids array format
        // "ALL" becomes ["ALL"], single ID becomes ["id"]
        // Comma-separated IDs become ["id1", "id2", ...]
        const file_ids = fileId.includes(',') ? fileId.split(',') : [fileId];

        const response = await fetch('/api/ingest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ file_ids }),
          signal: controller.signal,
        });

        if (!response.ok) {
          const text = await response.text();
          onError(text || `Failed to start ingestion (${response.status})`);
          return;
        }

        const jobIdHeader = response.headers.get('X-Job-ID');
        if (jobIdHeader) {
          setJobId(jobIdHeader);
        }

        setIsConnected(true);

        const reader = response.body?.getReader();
        if (!reader) {
          onError('Failed to read response stream');
          return;
        }

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Process complete SSE messages (delimited by \n\n)
          let delimiterIndex;
          while ((delimiterIndex = buffer.indexOf('\n\n')) !== -1) {
            const message = buffer.slice(0, delimiterIndex);
            buffer = buffer.slice(delimiterIndex + 2);

            // Parse the complete message
            let eventType = '';
            let eventData = '';
            for (const line of message.split('\n')) {
              if (line.startsWith('event: ')) {
                eventType = line.slice(7).trim();
              } else if (line.startsWith('data: ')) {
                eventData = line.slice(6).trim();
              }
            }

            if (eventType && eventData) {
              try {
                const data = JSON.parse(eventData) as ProgressData;
                setProgress(data);

                if (data.job_id) {
                  setJobId(data.job_id);
                }

                // Notify parent when a file is completed (for real-time UI update)
                if (data.completed_file_id && onFileCompleted) {
                  onFileCompleted(data.completed_file_id);
                }

                if (eventType === 'complete') {
                  onComplete();
                } else if (eventType === 'cancelled') {
                  onComplete();
                } else if (eventType === 'error') {
                  onError(data.error || 'Unknown error');
                }
              } catch (e) {
                console.error('Failed to parse SSE data:', e);
              }
            }
          }
        }
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          onError(err.message || 'Connection failed');
        }
      } finally {
        setIsConnected(false);
      }
    }

    startIngestion();

    return () => {
      controller.abort();
    };
  }, [fileId, onComplete, onError, onFileCompleted]);

  const handleCancel = useCallback(async () => {
    if (!jobId || isCancelling) return;

    setIsCancelling(true);
    try {
      const res = await fetch('/api/ingest/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        console.error('Cancel failed:', data.error || res.status);
      }
    } catch (err) {
      console.error('Cancel error:', err);
    }
  }, [jobId, isCancelling]);

  // Loading state
  if (!isConnected && !progress) {
    return (
      <div className="relative overflow-hidden rounded-xl bg-gradient-to-r from-slate-50 to-slate-100 border border-slate-200 p-5 mb-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-full border-2 border-slate-200 border-t-blue-500 animate-spin" />
          </div>
          <div>
            <div className="text-sm font-medium text-slate-700">Initializing...</div>
            <div className="text-xs text-slate-500">Preparing ingestion process</div>
          </div>
        </div>
      </div>
    );
  }

  if (progress?.total_files === 0) {
    return null;
  }

  const currentIndex = progress?.current_index || progress?.processed_files || 0;
  const totalFiles = progress?.total_files || 0;
  const percentComplete = totalFiles > 0 ? Math.round((currentIndex / totalFiles) * 100) : 0;
  const isProcessing = progress?.status === 'processing';
  const isCompleted = progress?.status === 'completed';
  const isCancelled = progress?.status === 'cancelled';
  const isError = progress?.status === 'error';

  return (
    <div className={`relative overflow-hidden rounded-xl border p-5 mb-4 transition-all duration-300 ${
      isCompleted ? 'bg-gradient-to-r from-emerald-50 to-green-50 border-emerald-200' :
      isCancelled ? 'bg-gradient-to-r from-amber-50 to-yellow-50 border-amber-200' :
      isError ? 'bg-gradient-to-r from-red-50 to-rose-50 border-red-200' :
      'bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200'
    }`}>
      {/* Animated background pattern for processing state */}
      {isProcessing && (
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent animate-shimmer"
               style={{ backgroundSize: '200% 100%', animation: 'shimmer 2s infinite linear' }} />
        </div>
      )}

      <div className="relative">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            {/* Status icon */}
            <div className={`flex items-center justify-center w-10 h-10 rounded-full ${
              isCompleted ? 'bg-emerald-100' :
              isCancelled ? 'bg-amber-100' :
              isError ? 'bg-red-100' :
              'bg-blue-100'
            }`}>
              {isProcessing && (
                <svg className="w-5 h-5 text-blue-600 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
              )}
              {isCompleted && (
                <svg className="w-5 h-5 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              )}
              {isCancelled && (
                <svg className="w-5 h-5 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              )}
              {isError && (
                <svg className="w-5 h-5 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              )}
            </div>

            <div>
              <div className={`text-sm font-semibold ${
                isCompleted ? 'text-emerald-700' :
                isCancelled ? 'text-amber-700' :
                isError ? 'text-red-700' :
                'text-blue-700'
              }`}>
                {isProcessing && 'Ingesting Documents'}
                {isCompleted && 'Ingestion Complete'}
                {isCancelled && 'Ingestion Cancelled'}
                {isError && 'Ingestion Failed'}
              </div>
              <div className="text-xs text-slate-500">
                {currentIndex} of {totalFiles} files processed
              </div>
            </div>
          </div>

          {/* Percentage badge */}
          <div className={`px-3 py-1 rounded-full text-sm font-bold ${
            isCompleted ? 'bg-emerald-100 text-emerald-700' :
            isCancelled ? 'bg-amber-100 text-amber-700' :
            isError ? 'bg-red-100 text-red-700' :
            'bg-blue-100 text-blue-700'
          }`}>
            {percentComplete}%
          </div>
        </div>

        {/* Progress bar */}
        <div className="relative h-3 bg-white/50 rounded-full overflow-hidden mb-3 shadow-inner">
          <div
            className={`absolute inset-y-0 left-0 rounded-full transition-all duration-500 ease-out ${
              isCompleted ? 'bg-gradient-to-r from-emerald-400 to-green-500' :
              isCancelled ? 'bg-gradient-to-r from-amber-400 to-yellow-500' :
              isError ? 'bg-gradient-to-r from-red-400 to-rose-500' :
              'bg-gradient-to-r from-blue-400 to-indigo-500'
            }`}
            style={{ width: `${percentComplete}%` }}
          >
            {/* Animated shine effect */}
            {isProcessing && (
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-pulse" />
            )}
          </div>
        </div>

        {/* Current file */}
        {progress?.current_file && isProcessing && (
          <div className="flex items-center gap-2 mb-3 p-2 bg-white/50 rounded-lg">
            <svg className="w-4 h-4 text-slate-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span className="text-xs text-slate-600 truncate">{progress.current_file}</span>
          </div>
        )}

        {/* Stats row */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            {(progress?.total_chunks ?? 0) > 0 && (
              <div className="flex items-center gap-1.5">
                <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                </svg>
                <span className="text-xs text-slate-600 font-medium">{progress?.total_chunks} chunks</span>
              </div>
            )}
          </div>

          {/* Cancel button */}
          {isProcessing && (
            <button
              onClick={handleCancel}
              disabled={isCancelling}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-red-600 bg-white border border-red-200 rounded-lg hover:bg-red-50 hover:border-red-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isCancelling ? (
                <>
                  <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Cancelling...
                </>
              ) : (
                <>
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                  Cancel
                </>
              )}
            </button>
          )}
        </div>

        {/* Status messages */}
        {isCompleted && (
          <div className="mt-3 p-2 bg-emerald-100/50 rounded-lg">
            <p className="text-xs text-emerald-700">
              {progress?.message || `Successfully ingested ${progress?.ingested_files} files with ${progress?.total_chunks} chunks`}
            </p>
          </div>
        )}
        {isCancelled && (
          <div className="mt-3 p-2 bg-amber-100/50 rounded-lg">
            <p className="text-xs text-amber-700">
              Ingestion was cancelled. {progress?.processed_files} files were processed before cancellation.
            </p>
          </div>
        )}
        {isError && (
          <div className="mt-3 p-2 bg-red-100/50 rounded-lg">
            <p className="text-xs text-red-700">Error: {progress?.error}</p>
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .animate-shimmer {
          animation: shimmer 2s infinite linear;
        }
      `}</style>
    </div>
  );
}
