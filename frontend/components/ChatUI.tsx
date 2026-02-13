'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useFilters } from './filters-context';
import { useChat } from './chat-context';
import ShimmerBubble from './ShimmerBubble';
import { looksLikeInjection } from '../lib/safety';
import { consumeSSEStream, parseHitlPayload, parseContextPayload, type SSEAccumulator } from '../lib/sse-parse';
import { MarkdownRenderer } from './MarkdownRenderer';

type Role = 'user' | 'assistant';
type Msg = { id?: string; role: Role; content: string; created_at?: string; ts?: number; attachments?: AttachmentPreview[] };
type AttachmentPreview = { name: string; type: string; url: string; data?: string };
type AttachmentPayload = { type: 'image' | 'file'; filename: string; mime_type: string; data: string };
type Context = { bm25: number; chunk: string; chunk_id: string; dept_id: string;
  ext: string; file_for_user: boolean; file_id: string; hybrid: number; page: number;
  rerank: number; sem_sim: number; size_kb: number; source: string; tags: string;
  upload_at: string; uploaded_at_ts: number; user_id: string};
type HITLInterrupt = {
  status: string;
  action: string;
  thread_id: string;
  details: { task?: string; recipient?: string; available_attachments?: string[] };
  previous_steps: Array<{ step?: number; question?: string; answer?: string }>;
  partial_answer: string;
  conversation_id: string;
};
  
const cls = (...s: Array<string | false | null | undefined>) => s.filter(Boolean).join(' ');
const tstr = (msg: Msg) => {
  if (msg.created_at) return new Date(msg.created_at).toLocaleTimeString();
  if (msg.ts) return new Date(msg.ts).toLocaleTimeString();
  return '';
};
const languages = [
  { code: 'en-US', text: 'English' },
  { code: 'sv-SE', text: 'Svenska' },
  { code: 'fi-FI', text: 'Suomi' },
  { code: 'fr-FR', text: 'Français' },
  { code: 'de-DE', text: 'Deutsch' },
  { code: 'zh-CN', text: '中文' },
  { code: 'zh-TW', text: '繁體中文' },
  { code: 'ja-JP', text: '日本語' }
];

export default function ChatPage() {
  const { selectedExts, selectedTags, customTags } = useFilters();
  const { messages, setMessages, contexts, setContexts, setConversations, selectedConversation, setSelectedConversation, isLoadingConversation } = useChat();
  const [showContexts, setShowContexts] = useState(false);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [recognition, setRecognition] = useState<SpeechRecognition | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [language, setLanguage] = useState('en-US');
  const [attachments, setAttachments] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [hitlInterrupt, setHitlInterrupt] = useState<HITLInterrupt | null>(null);
  const [isResuming, setIsResuming] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamingRef = useRef(false);
  const speakingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dragCounterRef = useRef(0); // Track drag enter/leave events

  const clearSpeakingTimer = useCallback(() => {
    if (speakingTimeoutRef.current) {
      clearTimeout(speakingTimeoutRef.current);
      speakingTimeoutRef.current = null;
    }
  }, []);

  const pulseSpeaking = useCallback((duration = 1000) => {
    setIsSpeaking(true);
    clearSpeakingTimer();
    speakingTimeoutRef.current = setTimeout(() => {
      setIsSpeaking(false);
      speakingTimeoutRef.current = null;
    }, duration);
  }, [clearSpeakingTimer, setIsSpeaking]);

  const stopSpeaking = useCallback(() => {
    clearSpeakingTimer();
    setIsSpeaking(false);
  }, [clearSpeakingTimer, setIsSpeaking]);

  useEffect(() => {
    if ('webkitSpeechRecognition' in window) {
      const recog = new webkitSpeechRecognition() as SpeechRecognition;
      recog.continuous = true; // continue listening until user stops
      recog.interimResults = true;
      recog.lang = language;

      recog.onresult = (event) => {
        const transcript = Array.from(event.results)
        .map((result) => result[0].transcript)
        .join(' ');

        setInput(transcript);
        pulseSpeaking();
      };

      recog.onerror = (event) => {
        console.error('Speech recognition error', event.error);
        stopSpeaking();
        setIsRecording(false);
      };

      recog.onspeechstart = () => {
        pulseSpeaking();
      };

      recog.onspeechend = () => {
        stopSpeaking();
      };

      recog.onstart = () => {
        stopSpeaking();
      };

      recog.onend = () => {
        setIsRecording(false);
        stopSpeaking();
      };

      setRecognition(recog);
    } else {
      console.warn('Speech Recognition API not supported in this browser.');
    }
  }, [language, pulseSpeaking, stopSpeaking]);

  useEffect(() => {
    return () => {
      clearSpeakingTimer();
    };
  }, [clearSpeakingTimer]);

  // autoscroll on new message
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });

    console.log(messages);
  }, [messages.length]);

  async function* streamChat(body: any) {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(async () => ({ error: await res.text() }));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    if (!res.body) throw new Error('No response body');

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      yield dec.decode(value, { stream: true });
    }
  }

  async function* streamResume(body: { thread_id: string; confirmed: boolean; conversation_id?: string }) {
    const res = await fetch('/api/chat/resume', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(async () => ({ error: await res.text() }));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    if (!res.body) throw new Error('No response body');

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      yield dec.decode(value, { stream: true });
    }
  }

  async function onSend() {
    if ((!input.trim() && attachments.length === 0) || streamingRef.current) return;
    const { flagged, error } = looksLikeInjection(input);
    if (flagged) {
      const msgs: Msg[] = [...messages, { role: 'user', content: error, ts: Date.now() }];
      setMessages(msgs);
      setContexts([]);
      setInput('');
      setAttachments([]);
      return;
    }

    streamingRef.current = true;
    setBusy(true);

    // Convert attachments to base64 for API payload
    const attachmentPayloads: AttachmentPayload[] = await Promise.all(
      attachments.map(async (file) => {
        const base64 = await fileToBase64(file);
        return {
          type: file.type.startsWith('image/') ? 'image' as const : 'file' as const,
          filename: file.name,
          mime_type: file.type,
          data: base64,
        };
      })
    );

    // Create preview URLs for displaying in chat
    const attachmentPreviews: AttachmentPreview[] = attachments.map(file => ({
      name: file.name,
      type: file.type,
      url: URL.createObjectURL(file),
    }));

    const ts = Date.now();
    const userMessage: Msg = {
      role: 'user',
      content: input.trim(),
      ts,
      attachments: attachmentPreviews.length > 0 ? attachmentPreviews : undefined
    };
    const next: Msg[] = [...messages, userMessage, { role: 'assistant', content: '', ts }];
    setMessages(next);
    setInput('');
    setAttachments([]);

    // Only send the latest user message (backend fetches full history from DB)
    const messages_payload = [userMessage];
    const filters_payload = [];
    if (selectedExts.size) {
      filters_payload.push({ exts: Array.from(selectedExts)});
    }
    const allTags = Array.from(
      new Set([
        ...selectedTags,
        ...customTags
        .split(',')
        .map(t => t.trim())
        .filter(Boolean)
      ])
    );
    if (allTags.length) {
      filters_payload.push({tags: allTags.map(t => t.trim().toLowerCase()).filter(Boolean)});
    }

    // check conversaion selection
    const conversationId = selectedConversation ? selectedConversation.id : null;
    const payload = {
      messages: messages_payload,
      conversation_id: conversationId,
      filters: filters_payload.length ? filters_payload : undefined,
      attachments: attachmentPayloads.length > 0 ? attachmentPayloads : undefined
    };

    const appendChunk = (text: string) => {
      setMessages(curr => {
        const copy = [...curr];
        if (!copy.length) return copy;
        copy[copy.length - 1] = { ...copy[copy.length - 1], content: copy[copy.length - 1].content + text };
        return copy;
      });
    };
    const acc: SSEAccumulator = { hitlRaw: '', contextRaw: '' };
    try {
      setContexts([]);
      setHitlInterrupt(null);
      await consumeSSEStream(streamChat(payload), appendChunk, acc);
    } catch (e: any) {
      setMessages(curr => {
        const copy = [...curr];
        if (!copy.length) return copy;
        copy[copy.length - 1] = { role: 'assistant', content: `Stream error: ${e?.message || 'Stream error'}`, ts: Date.now() };
        return copy;
      });
    } finally {
      const hitlData = parseHitlPayload(acc.hitlRaw);
      if (hitlData) {
        setHitlInterrupt(hitlData as HITLInterrupt);
        console.log('[HITL] Interrupt received:', hitlData);
      }

      const ctxArr = parseContextPayload(acc.contextRaw);
      const isContext = (o: any): o is Context => o && typeof o === 'object' && 'chunk' in o && 'source' in o && 'page' in o;
      setContexts(ctxArr.filter(isContext));

      if (selectedConversation == null) {
        try {
          const res = await fetch('/api/conversations');
          if (res.ok) {
            console.log('This is a new conversation from ChatUI');
            const result = await res.json();
            if (result && Array.isArray(result.conversations) && result.conversations.length > 0) {
              setConversations(result.conversations);
              setSelectedConversation(result.conversations[0]);
            }
          }
        } catch (e) {
          console.error('Failed to refresh conversations:', e);
        }
      }

      streamingRef.current = false;
      setBusy(false);
    }
  }

  const startRecording = () => {
    if (recognition) {
      setIsRecording(true);
      stopSpeaking();
      recognition.start();
    }
  };

  const stopRecording = () => {
    if (recognition) {
      setIsRecording(false);
      stopSpeaking();
      recognition.stop();
    }
  };

  // HITL handlers
  const handleHitlResponse = async (confirmed: boolean) => {
    if (!hitlInterrupt || isResuming) return;

    setIsResuming(true);
    streamingRef.current = true;
    setBusy(true);

    // Add a placeholder message for the resume response
    const ts = Date.now();
    setMessages(curr => [...curr, { role: 'assistant', content: '', ts }]);

    const appendChunk = (text: string) => {
      setMessages(curr => {
        const copy = [...curr];
        if (!copy.length) return copy;
        copy[copy.length - 1] = { ...copy[copy.length - 1], content: copy[copy.length - 1].content + text };
        return copy;
      });
    };
    const acc: SSEAccumulator = { hitlRaw: '', contextRaw: '' };
    try {
      setContexts([]);
      const resumePayload = {
        thread_id: hitlInterrupt.thread_id,
        confirmed,
        conversation_id: hitlInterrupt.conversation_id || selectedConversation?.id,
      };
      await consumeSSEStream(streamResume(resumePayload), appendChunk, acc);
    } catch (e: any) {
      setMessages(curr => {
        const copy = [...curr];
        if (!copy.length) return copy;
        copy[copy.length - 1] = { role: 'assistant', content: `Resume error: ${e?.message || 'Stream error'}`, ts: Date.now() };
        return copy;
      });
    } finally {
      const hitlData = parseHitlPayload(acc.hitlRaw);
      if (hitlData) {
        setHitlInterrupt(hitlData as HITLInterrupt);
        console.log('[HITL] Chained interrupt received:', hitlData);
      } else {
        setHitlInterrupt(null);
      }

      const ctxArr = parseContextPayload(acc.contextRaw);
      const isContext = (o: any): o is Context => o && typeof o === 'object' && 'chunk' in o && 'source' in o && 'page' in o;
      setContexts(ctxArr.filter(isContext));

      streamingRef.current = false;
      setBusy(false);
      setIsResuming(false);
    }
  };

  const handleHitlConfirm = () => handleHitlResponse(true);
  const handleHitlCancel = () => handleHitlResponse(false);

  // File upload handlers
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const validFiles: File[] = [];
    const maxSize = 10 * 1024 * 1024; // 10MB

    Array.from(files).forEach(file => {
      if (file.size > maxSize) {
        alert(`${file.name} is too large. Max 10MB.`);
        return;
      }
      validFiles.push(file);
    });

    setAttachments(prev => [...prev, ...validFiles]);
    e.target.value = ''; // Reset input to allow re-selecting same file
  };

  const removeAttachment = (index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };

  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const result = reader.result as string;
        // Remove data URL prefix (e.g., "data:image/png;base64,")
        const base64 = result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
    });
  };

  // Drag and drop handlers
  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current += 1;
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current -= 1;
    if (dragCounterRef.current === 0) {
      setIsDragging(false);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    dragCounterRef.current = 0;

    const files = e.dataTransfer.files;
    if (!files || files.length === 0) return;

    const validFiles: File[] = [];
    const maxSize = 10 * 1024 * 1024; // 10MB
    const acceptedTypes = ['image/', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.csv', '.txt'];

    Array.from(files).forEach(file => {
      // Check file size
      if (file.size > maxSize) {
        alert(`${file.name} is too large. Max 10MB.`);
        return;
      }

      // Check file type
      const isAccepted = acceptedTypes.some(type =>
        type.startsWith('.')
          ? file.name.toLowerCase().endsWith(type)
          : file.type.startsWith(type.replace('/', ''))
      );

      if (!isAccepted) {
        alert(`${file.name} is not a supported file type.`);
        return;
      }

      validFiles.push(file);
    });

    if (validFiles.length > 0) {
      setAttachments(prev => [...prev, ...validFiles]);
    }
  };

  return (
    <div className="h-full w-full bg-neutral-100 flex justify-center">
      <div
        className="flex h-full w-full max-w-[1400px] flex-col border-x border-neutral-200 bg-white shadow-sm relative"
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Drag and drop overlay */}
        {isDragging && (
          <div className="absolute inset-0 z-50 bg-blue-500/10 backdrop-blur-sm border-4 border-dashed border-blue-500 flex items-center justify-center">
            <div className="bg-white rounded-xl shadow-2xl px-8 py-6 flex flex-col items-center gap-3">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-blue-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <div className="text-lg font-semibold text-neutral-900">Drop files to attach</div>
              <div className="text-sm text-neutral-500">Images, PDFs, documents, spreadsheets (Max 10MB each)</div>
            </div>
          </div>
        )}
        {/* Header */}
        <header className="flex h-12 items-center justify-between border-b bg-white px-4">
          <div className="flex items-center gap-2">
            <div className="grid h-7 w-18 place-items-center rounded-md bg-neutral-900 text-xs font-semibold text-white">Chatbot</div>
            <div className="text-sm font-semibold">Assistant</div>
          </div>
          <div className="flex items-center gap-3">
            <div className={cls('text-xs', busy ? 'text-blue-600' : 'text-neutral-500')}>
              {busy ? 'Generating...' : 'Ready'}
            </div>
          </div>
        </header>
        {/* Conversation */}
        <main ref={scrollRef} className="flex-1 overflow-y-auto overscroll-contain [scrollbar-gutter:stable_both-edges]">
          <div className="space-y-4 px-4 py-4 md:px-6">
            {isLoadingConversation ? (
              <div className="mt-16 flex flex-col items-center justify-center gap-4">
                <svg className="animate-spin h-8 w-8 text-blue-600 dark:text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <div className="text-sm text-neutral-500 dark:text-neutral-400">Loading conversation...</div>
              </div>
            ) : messages.length === 0 ? (
              <div className="mt-16 text-center text-sm text-neutral-500 dark:text-neutral-400">
                Ask a question to get started. Your answers will stream in here.
              </div>
            ) : (
              messages.map((m, i) => (
                <div key={i} className={cls('flex gap-3', m.role === 'user' ? 'flex-row-reverse' : '')}>
                  {/* avatar */}
                  <div
                    className={cls(
                      'mt-1 text-[15px] px-2 h-7 w-auto rounded-md grid place-items-center text-xs font-semibold whitespace-nowrap',
                      m.role === 'user' ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900' : 'bg-neutral-200 text-neutral-700 dark:bg-neutral-700 dark:text-neutral-100'
                  )}
                  >
                    {m.role === 'user' ? 'You' : 'Assistant'}
                  </div>
                  {/* bubble */}
                  <div
                    className={cls(
                      'max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed',
                      m.role === 'user' ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900' : 'bg-white border shadow-sm dark:bg-neutral-800 dark:border-neutral-700 dark:text-neutral-100'
                    )}
                  >
                    {/* Attachments preview in message */}
                    {m.attachments && m.attachments.length > 0 && (
                      <div className="flex flex-wrap gap-2 mb-2">
                        {m.attachments.map((att, idx) => (
                          <div
                            key={idx}
                            className="relative cursor-pointer"
                            onClick={() => window.open(att.url, '_blank')}
                            title={`Click to open ${att.name}`}
                          >
                            {att.type.startsWith('image/') ? (
                              <img
                                src={att.url}
                                alt={att.name}
                                className="h-24 w-24 object-cover rounded-lg border border-neutral-600 hover:opacity-80 transition"
                              />
                            ) : (
                              <div className="h-16 w-20 rounded-lg border border-neutral-600 bg-neutral-800 flex flex-col items-center justify-center p-2 hover:bg-neutral-700 transition">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-neutral-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                  <polyline points="14 2 14 8 20 8" />
                                </svg>
                                <span className="text-[10px] text-neutral-400 mt-1 truncate max-w-full">{att.name.split('.').pop()?.toUpperCase()}</span>
                              </div>
                            )}
                            <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-[9px] px-1 py-0.5 truncate rounded-b-lg pointer-events-none">
                              {att.name}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                    <div className={cls('mt-1 text-[16px]')}>
                      <MarkdownRenderer content={m.content} />
                    </div>
                    <div className={cls('mt-1 text-[12px]', m.role === 'user' ? 'text-neutral-300 dark:text-neutral-600' : 'text-neutral-500 dark:text-neutral-400')}>
                      {tstr(m)}
                    </div>
                  </div>
                </div>
              ))
            )
            }
            {contexts.length > 0 && (
              <div className="mt-8">
                <button
                  type="button"
                  onClick={() => setShowContexts(s => !s)}
                  className="flex items-center gap-2 rounded-lg border px-3 py-2 text-sm font-medium bg-neutral-50 hover:bg-neutral-100 transition"
                  aria-expanded={showContexts}
                >
                  <span>{showContexts ? '▾' : '▸'}</span>
                  <span>Sources ({contexts.length})</span>
                </button>
                {showContexts && (
                  <div className="mt-3 space-y-3">
                    {contexts.map((c, idx) => (
                      <ContextCard key={c.chunk_id || idx} context={c} index={idx} />
                    ))}
                  </div>
                )}
              </div>
            )}
            {/* HITL Confirmation Dialog */}
            {hitlInterrupt && !busy && (
              <div className="mt-6 rounded-xl border-2 border-amber-400 bg-amber-50 p-4 shadow-md">
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-amber-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                      <line x1="12" y1="9" x2="12" y2="13" />
                      <line x1="12" y1="17" x2="12.01" y2="17" />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-sm font-semibold text-amber-800">Action Confirmation Required</h3>
                    <p className="mt-1 text-sm text-amber-700">
                      {hitlInterrupt.action === 'send_email' ? 'The assistant wants to send an email.' : `Action: ${hitlInterrupt.action}`}
                    </p>
                    {hitlInterrupt.details && (
                      <div className="mt-3 rounded-lg bg-white p-3 border border-amber-200">
                        {hitlInterrupt.details.recipient && (
                          <div className="text-sm">
                            <span className="font-medium text-neutral-700">To: </span>
                            <span className="text-neutral-600">{hitlInterrupt.details.recipient}</span>
                          </div>
                        )}
                        {hitlInterrupt.details.task && (
                          <div className="mt-1 text-sm">
                            <span className="font-medium text-neutral-700">Task: </span>
                            <span className="text-neutral-600">{hitlInterrupt.details.task}</span>
                          </div>
                        )}
                        {hitlInterrupt.details.available_attachments && hitlInterrupt.details.available_attachments.length > 0 && (
                          <div className="mt-2">
                            <span className="text-sm font-medium text-neutral-700">Attachments:</span>
                            <ul className="mt-1 list-disc list-inside text-sm text-neutral-600">
                              {hitlInterrupt.details.available_attachments.map((att, idx) => (
                                <li key={idx}>{att}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                    <div className="mt-4 flex gap-3">
                      <button
                        type="button"
                        onClick={handleHitlConfirm}
                        disabled={isResuming}
                        className="px-4 py-2 rounded-lg bg-green-600 text-white text-sm font-medium hover:bg-green-700 disabled:opacity-50 transition flex items-center gap-2"
                      >
                        {isResuming ? (
                          <>
                            <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <span>Processing...</span>
                          </>
                        ) : (
                          <>
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <polyline points="20 6 9 17 4 12" />
                            </svg>
                            <span>Confirm & Send</span>
                          </>
                        )}
                      </button>
                      <button
                        type="button"
                        onClick={handleHitlCancel}
                        disabled={isResuming}
                        className="px-4 py-2 rounded-lg bg-neutral-200 text-neutral-700 text-sm font-medium hover:bg-neutral-300 disabled:opacity-50 transition flex items-center gap-2"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <line x1="18" y1="6" x2="6" y2="18" />
                          <line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                        <span>Cancel</span>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
            {busy && (
              <div className="self-start">
                  <ShimmerBubble />
              </div>
            )}
            <div aria-live="polite" className="sr-only">
              {busy ? "Assistant is typing..." : ""}
            </div>
          </div>
        </main>

        {/* Composer */}
        <footer className="border-t bg-white">
          <form
            onSubmit={e => {
              e.preventDefault();
              void onSend();
            }}
            className="mx-auto flex max-w-[1200px] flex-col gap-2 px-4 py-3 w-full"
          >
          {/* Attachment Preview */}
          {attachments.length > 0 && (
            <div className="flex flex-wrap gap-2 px-1">
              {attachments.map((file, idx) => (
                <div key={idx} className="relative group">
                  {file.type.startsWith('image/') ? (
                    <img
                      src={URL.createObjectURL(file)}
                      alt={file.name}
                      onClick={() => window.open(URL.createObjectURL(file), '_blank')}
                      className="h-16 w-16 object-cover rounded-lg border cursor-pointer hover:opacity-80 transition"
                    />
                  ) : (
                    <div
                      onClick={() => window.open(URL.createObjectURL(file), '_blank')}
                      className="h-16 w-16 rounded-lg border bg-neutral-100 flex flex-col items-center justify-center cursor-pointer hover:bg-neutral-200 transition"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-neutral-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                      </svg>
                      <span className="text-[10px] text-neutral-500 mt-1 truncate max-w-[56px] px-1">
                        {file.name.split('.').pop()?.toUpperCase()}
                      </span>
                    </div>
                  )}
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); removeAttachment(idx); }}
                    className="absolute -top-2 -right-2 h-5 w-5 rounded-full bg-red-500 text-white text-xs flex items-center justify-center opacity-0 group-hover:opacity-100 transition"
                  >
                    ×
                  </button>
                  <div className="absolute bottom-0 left-0 right-0 bg-black/50 text-white text-[9px] px-1 truncate rounded-b-lg pointer-events-none">
                    {file.name}
                  </div>
                </div>
              ))}
            </div>
          )}
          {/* Hidden file input */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            multiple
            accept="image/*,.pdf,.doc,.docx,.xls,.xlsx,.csv,.txt"
            className="hidden"
          />
          <div className="flex items-end gap-1 w-full">
            <textarea
              className="text-[16px] flex-1 resize-y rounded-xl border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-900/20 min-h-[80px]"
              placeholder="Send a message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              rows={3}
            />
            <button
              type="submit"
              disabled={busy || (!input.trim() && attachments.length === 0)}
              className="text-[18px] h-10 px-6 rounded-xl bg-neutral-900 text-white text-sm disabled:opacity-50"
            >
              Send
            </button>
            {/* Attach button */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={busy}
              className="h-10 px-3 rounded-xl text-sm transition-all flex items-center gap-1 hover:bg-neutral-100"
              title="Attach files or images"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-neutral-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
              </svg>
              <span className="text-neutral-500 hidden sm:inline">Attach</span>
            </button>
            <button
              type="button"
              onClick={isRecording ? stopRecording : startRecording}
              className="h-10 px-4 rounded-xl text-sm transition-all flex items-center gap-2"
              disabled={busy}
            >
              {isRecording ? (
                <>
                  <div className="flex items-center gap-1">
                    <div className={cls('h-5 w-1 bg-red-600', isSpeaking && 'animate-listening')} style={{ animationDelay: '0ms' }}></div>
                    <div className={cls('h-5 w-1 bg-red-600', isSpeaking && 'animate-listening')} style={{ animationDelay: '150ms' }}></div>
                    <div className={cls('h-5 w-1 bg-red-600', isSpeaking && 'animate-listening')} style={{ animationDelay: '300ms' }}></div>
                    <div className={cls('h-5 w-1 bg-red-600', isSpeaking && 'animate-listening')} style={{ animationDelay: '450ms' }}></div>
                    <div className={cls('h-5 w-1 bg-red-600', isSpeaking && 'animate-listening')} style={{ animationDelay: '600ms' }}></div>
                  </div>
                  <span className="text-red-600 font-medium">Listening...</span>
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-neutral-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 14a4 4 0 0 0 4-4V5a4 4 0 0 0-8 0v5a4 4 0 0 0 4 4z" />
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                    <line x1="12" y1="19" x2="12" y2="23" />
                    <line x1="8" y1="23" x2="16" y2="23" />
                  </svg>
                  <span className="text-neutral-500">Voice Input</span>
                </>
              )}
            </button>
            <div className="flex items-center gap-2">
              <label htmlFor="language" className="text-sm font-medium text-neutral-700">Language:</label>
              <select
                id="language"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="text-sm border border-neutral-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-neutral-900/20 bg-white shadow-sm"
              >
                {languages.map(({code, text}) => (
                    <option value={code}>{text}</option>
                ))}
              </select>
            </div>
          </div>
          </form>
        </footer>
      </div>
    </div>
  );
}

// Individual collapsible context card
function ContextCard({ context, index }: { context: Context; index: number }) {
  const [open, setOpen] = useState(false);
  // Build a compact score line
  const scoreLine = useMemo(() => {
    const parts: string[] = [];
    if (typeof context.sem_sim === 'number') parts.push(`sem ${context.sem_sim.toFixed(2)}`);
    if (typeof context.hybrid === 'number') parts.push(`hyb ${context.hybrid.toFixed(2)}`);
    if (typeof context.rerank === 'number' && context.rerank > 0) parts.push(`rerank ${context.rerank.toFixed(2)}`);
    return parts.join(' • ');
  }, [context]);

  const preview = useMemo(() => {
    const text = context.chunk.replace(/\s+/g, ' ').trim();
    const MAX = open ? 1200 : 320;
    return text.length > MAX ? text.slice(0, MAX) + '…' : text;
  }, [context.chunk, open]);

  return (
    <div className="rounded-lg border bg-white shadow-sm">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="w-full text-left px-3 py-2 flex items-start justify-between gap-3 hover:bg-neutral-50"
        aria-expanded={open}
      >
        <div className="flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold px-1.5 py-0.5 rounded bg-neutral-900 text-white">{index + 1}</span>
            <span className="text-sm font-medium truncate max-w-[240px]" title={context.source}>{context.source}</span>
            {context.page > 0 && <span className="text-xs text-neutral-500">p{context.page}</span>}
            {scoreLine && <span className="text-xs text-neutral-400">{scoreLine}</span>}
          </div>
          <div className="mt-1 text-xs text-neutral-600 line-clamp-4">
            {preview}
          </div>
          {context.tags && (
            <div className="mt-1 flex flex-wrap gap-1">
              {context.tags.split(',').filter(Boolean).slice(0, 6).map(t => (
                <span key={t} className="text-[10px] uppercase tracking-wide bg-neutral-200 text-neutral-700 px-1.5 py-0.5 rounded">
                  {t.trim()}
                </span>
              ))}
              {context.tags.split(',').filter(Boolean).length > 6 && (
                <span className="text-[10px] text-neutral-500">+{context.tags.split(',').filter(Boolean).length - 6} more</span>
              )}
            </div>
          )}
        </div>
        <div className="pl-2">
          <span className="text-xs text-neutral-500">{open ? 'Hide' : 'Expand'}</span>
        </div>
      </button>
      {open && (
        <div className="border-t px-3 py-2 bg-neutral-50">
          <div className="text-xs font-mono text-neutral-500">Chunk ID: {context.chunk_id}</div>
          <div className="mt-1 text-xs text-neutral-700 whitespace-pre-wrap">{context.chunk}</div>
        </div>
      )}
    </div>
  );
}
