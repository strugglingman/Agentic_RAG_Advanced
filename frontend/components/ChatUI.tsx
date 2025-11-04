'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useFilters } from './filters-context';
import ShimmerBubble from './ShimmerBubble';

type Role = 'user' | 'assistant';
type Msg = { role: Role; content: string; ts?: number };

const cls = (...s: Array<string | false | null | undefined>) => s.filter(Boolean).join(' ');
const tstr = (ts?: number) => (ts ? new Date(ts).toLocaleTimeString() : '');

export default function ChatPage() {
  const { selectedExts, selectedTags, customTags } = useFilters();

  // chat state
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);

  // ui
  const scrollRef = useRef<HTMLDivElement>(null);
  const streamingRef = useRef(false);

  // autoscroll on new message
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
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

    const payload = {
      messages: messages_payload,
      filters: filters_payload.length ? filters_payload : undefined
    };

    try {
      for await (const chunk of streamChat(payload)) {
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
    } catch (e: any) {
      setMessages(curr => {
        const copy = [...curr];
        if (!copy.length) return copy;
        copy[copy.length - 1] = { role: 'assistant', content: `Stream error: ${e?.message || 'Stream error'}`, ts: Date.now() };
        return copy;
      });
    } finally {
      streamingRef.current = false;
      setBusy(false);

      console.log('11111111111111');
      console.log(messages);
    }
  }

  return (
    <div className="h-full w-full bg-neutral-100 flex justify-center">
      <div className="flex h-full w-full max-w-6xl flex-col border-x border-neutral-200 bg-white shadow-sm">
        {/* Header */}
        <header className="flex h-12 items-center justify-between border-b bg-white px-4">
          <div className="flex items-center gap-2">
            <div className="grid h-7 w-18 place-items-center rounded-md bg-neutral-900 text-xs font-semibold text-white">Chatbot</div>
            <div className="text-sm font-semibold">Assistant</div>
          </div>
          <div className={cls('text-xs', busy ? 'text-blue-600' : 'text-neutral-500')}>
            {busy ? 'Generating...' : 'Ready'}
          </div>
        </header>
        {/* Conversation */}
        <main ref={scrollRef} className="flex-1 overflow-y-auto overscroll-contain [scrollbar-gutter:stable_both-edges]">
          <div className="space-y-4 px-4 py-4 md:px-6">
            {messages.length === 0 ? (
              <div className="mt-16 text-center text-sm text-neutral-500">
                Ask a question to get started. Your answers will stream in here.
              </div>
            ) : (
              messages.map((m, i) => (
                <div key={i} className={cls('flex gap-3', m.role === 'user' ? 'flex-row-reverse' : '')}>
                  {/* avatar */}
                  <div
                    className={cls(
                      'mt-1 text-[15px] px-2 h-7 w-auto rounded-md grid place-items-center text-xs font-semibold whitespace-nowrap',
                      m.role === 'user' ? 'bg-neutral-900 text-white' : 'bg-neutral-200 text-neutral-700'
                  )}
                  >
                    {m.role === 'user' ? 'You' : 'Assistant'}
                  </div>
                  {/* bubble */}
                  <div
                    className={cls(
                      'max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed',
                      m.role === 'user' ? 'bg-neutral-900 text-white' : 'bg-white border shadow-sm'
                    )}
                  >
                    <div className={cls('mt-1 text-[16px]', 'whitespace-pre-wrap')}>{m.content}</div>
                    <div className={cls('mt-1 text-[12px]', m.role === 'user' ? 'text-neutral-300' : 'text-neutral-500')}>
                      {tstr(m.ts)}
                    </div>
                  </div>
                </div>
              ))
            )
            }
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
          <textarea
            className="text-[16px] flex-1 min-h-[44px] max-h-[160px] resize-y rounded-xl border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-900/20"
            placeholder="Send a message..."
            value={input}
            onChange={e => setInput(e.target.value)}
            rows={2}
          />
          <button
            type="submit"
            disabled={busy || !input.trim()}
            className="text-[18px] h-10 px-6 rounded-xl bg-neutral-900 text-white text-sm disabled:opacity-50"
          >
            Send
          </button>
          </form>
        </footer>
      </div>
    </div>
  );
}
