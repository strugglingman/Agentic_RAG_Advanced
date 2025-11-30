'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useFilters } from './filters-context';
import { useChat } from './chat-context';
import ShimmerBubble from './ShimmerBubble';
import { looksLikeInjection } from '../lib/safety';

type Role = 'user' | 'assistant';
type Msg = { id?: string; role: Role; content: string; created_at?: string; ts?: number };
type Context = { bm25: number; chunk: string; chunk_id: string; dept_id: string;
  ext: string; file_for_user: boolean; file_id: string; hybrid: number; page: number;
  rerank: number; sem_sim: number; size_kb: number; source: string; tags: string;
  upload_at: string; uploaded_at_ts: number; user_id: string};
  
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

  const scrollRef = useRef<HTMLDivElement>(null);
  const streamingRef = useRef(false);
  const speakingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

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

  async function onSend() {
    if (!input.trim() || streamingRef.current) return;
    const { flagged, error } = looksLikeInjection(input); 
    if (flagged) {
      const msgs: Msg[] = [...messages, { role: 'user', content: error, ts: Date.now() }];
      setMessages(msgs);
      setContexts([]);
      setInput('');
      return;
    }

    streamingRef.current = true;
    setBusy(true);

    const ts = Date.now();
    const next: Msg[] = [...messages, { role: 'user', content: input.trim(), ts }, { role: 'assistant', content: '', ts }];
    setMessages(next);
    setInput('');

    const messages_payload = next.filter(m => m.role == 'user' || m.content.trim());
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
      filters: filters_payload.length ? filters_payload : undefined
    };

    let startContext = false;
    let contextStr = '';
    try {
      setContexts([]);
      for await (const chunk of streamChat(payload)) {
        if (chunk.includes('__CONTEXT__:') || startContext) {
          startContext = true;
          contextStr += chunk;
        }

        if (!startContext) {
          setMessages(curr => {
            const copy = [...curr];
            if (!copy.length) return copy;
            copy[copy.length - 1] = {
              ...copy[copy.length - 1],
              content: copy[copy.length - 1].content + chunk,
            };
            return copy;
          });
        }
      }
    } catch (e: any) {
      setMessages(curr => {
        const copy = [...curr];
        if (!copy.length) return copy;
        copy[copy.length - 1] = { role: 'assistant', content: `Stream error: ${e?.message || 'Stream error'}`, ts: Date.now() };
        return copy;
      });
    } finally {
      if (contextStr) {
        contextStr = contextStr.substring(contextStr.indexOf('__CONTEXT__:') + '__CONTEXT__:'.length).trim();
        try {
          const contextRaw = JSON.parse(contextStr);
          const contextArr = Array.isArray(contextRaw) ? contextRaw : [];
          const isContext = (o: any): o is Context => o && typeof o === 'object' && 'chunk' in o && 'source' in o && 'page' in o;
          setContexts(contextArr.filter(isContext));
        } catch (e) {
          console.error('Failed to parse context JSON:', e);
        }
      }

      if (selectedConversation == null) {
        // After first message, refresh conversation list
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

  return (
    <div className="h-full w-full bg-neutral-100 flex justify-center">
      <div className="flex h-full w-full max-w-6xl flex-col border-x border-neutral-200 bg-white shadow-sm">
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
                    <div className={cls('mt-1 text-[16px]', 'whitespace-pre-wrap')}>{m.content}</div>
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
            className="mx-auto flex max-w-3xl items-end gap-2 px-4 py-3"
          >
          <div className="flex items-end gap-1 w-full">
            <textarea
              className="text-[16px] flex-grow resize-y rounded-xl border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-900/20 w-full md:w-[85%]"
              placeholder="Send a message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              rows={2} // Keep the height unchanged
            />
            <button
              type="submit"
              disabled={busy || !input.trim()}
              className="text-[18px] h-10 px-6 rounded-xl bg-neutral-900 text-white text-sm disabled:opacity-50"
            >
              Send
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
            <span className="text-xs text-neutral-500">p{context.page}</span>
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
