"use client";
import SidebarFilters from "./SidebarFilters";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useChat } from "./chat-context";
import { useState, useRef, useEffect } from "react";

const navItems = [
  { id: "chat",   label: "Chat",   href: "/chat" },
  { id: "upload", label: "Upload", href: "/upload" },
  { id: "files",  label: "Files",  href: "/files" },
  { id: "settings", label: "Settings", href: "/settings" },
];

export default function Sidebar() {
  const { conversations, setConversations, selectedConversation, setSelectedConversation, setMessages, setContexts, isLoadingConversation, setIsLoadingConversation } = useChat();
  const pathname = usePathname();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus input when editing starts
  useEffect(() => {
    if (editingId && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editingId]);

  function startEditing(e: React.MouseEvent, conv: any) {
    e.stopPropagation();
    setEditingId(conv.id);
    setEditingTitle(conv.title);
  }

  function cancelEditing() {
    setEditingId(null);
    setEditingTitle("");
  }

  async function saveTitle(convId: string) {
    if (!editingTitle.trim()) {
      cancelEditing();
      return;
    }

    try {
      const res = await fetch(`/api/conversations/${convId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: editingTitle }),
      });
      if (res.ok) {
        const updated_conv = await res.json();
        if (updated_conv && updated_conv.id) {
          setConversations(conversations.map(c =>
            c.id === convId ? { ...c, title: updated_conv.title, updated_at: updated_conv.updated_at } : c
          ));
        }
      }
    } catch(err) {
      console.error('Failed to update title:', err);
    }

    cancelEditing();
  }

  function handleKeyDown(e: React.KeyboardEvent, convId: string) {
    if (e.key === 'Enter') {
      e.preventDefault();
      saveTitle(convId);
    } else if (e.key === 'Escape') {
      cancelEditing();
    }
  }

  async function removeConversation(e: React.MouseEvent, convId: string) {
    // Prevent triggering parent click events
    e.stopPropagation();
    if (!confirm('Delete this conversation?')) return;
    try {
      const res = await fetch(`/api/conversations/${convId}`,
        { method: 'DELETE' }
      );
      if (res.ok) {
        setConversations(conversations.filter(c => c.id !== convId));
        // clear selection if needed
        if (selectedConversation && selectedConversation.id === convId) {
          setSelectedConversation(null);
          setMessages([]);
          setContexts([]);
        }
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err);
    }
  }

  async function selectConversion(e: React.MouseEvent, conv: any) {
    e.stopPropagation();

    if (selectedConversation?.id === conv.id) return;
    // if (isLoadingConversation) return;

    setIsLoadingConversation(true);
    setSelectedConversation(conv);

    // Navigate to chat page if not already there
    if (pathname !== '/chat') {
      window.location.href = '/chat';
    }

    // Fetch messages for the selected conversation
    try {
      const res = await fetch(`/api/conversations/${conv.id}`);
      if (res.ok) {
        const result = await res.json();
        setMessages(result.messages || []);
      }
    } catch (err) {
      console.error('Failed to fetch conversation messages:', err);
    } finally {
      setIsLoadingConversation(false);
    }
  }

  return (
    <div className="p-3 space-y-6">
      <div className="px-2">
        <h2 className="text-xl font-semibold tracking-tight">Agentic RAG Chatbot</h2>
      </div>
      {/* Navigation */}
      <nav className="flex flex-col gap-2" aria-label="Sidebar">
        {navItems.map(item => {
          const active = item.href === pathname || pathname.startsWith(item.href + "/");
          return (
            <Link
              key={item.id}
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={
                [
                  "group w-full text-left px-4 py-2 rounded-lg text-sm font-medium transition",
                  "border",
                  active
                    ? "bg-blue-500 text-white border-blue-500 shadow-sm"
                    : "bg-white text-neutral-700 border-neutral-200 hover:border-blue-400 hover:bg-blue-50",
                  "hover:shadow-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
                  "active:scale-[0.98]"
                ].join(" ")
              }
            >
              <span className="flex items-center gap-2">
                <span className={
                  active
                    ? "inline-block w-2 h-2 rounded-full bg-white"
                    : "inline-block w-2 h-2 rounded-full bg-blue-300 opacity-0 group-hover:opacity-100 transition"
                } />
                {item.label}
              </span>
            </Link>
          );
        })}
      </nav>
      <div className="flex-1 overflow-y-auto">
        <div className="px-2 mb-2 flex items-center justify-between">
          <h3 className="text-xs font-semibold text-neutral-500 dark:text-neutral-400 uppercase">Recent</h3>
          <button
            onClick={() => {
              setSelectedConversation(null);
              setMessages([]);
              setContexts([]);
              // Navigate to chat page
              if (pathname !== '/chat') {
                window.location.href = '/chat';
              }
            }}
            className="px-3 py-1.5 text-xs font-medium rounded-lg bg-blue-500 text-white hover:bg-blue-600 transition"
            title="Start new conversation"
          >
            + New Chat
          </button>
        </div>
        {conversations.map((conv) => {
          const isSelected = selectedConversation?.id === conv.id;
          const isLoading = isLoadingConversation && isSelected;
          return (
            <div
              key={conv.id}
              className={`group relative flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition ${
                isSelected
                  ? 'bg-blue-100 dark:bg-blue-900/30 border border-blue-300 dark:border-blue-700'
                  : 'hover:bg-neutral-100 dark:hover:bg-neutral-700'
              }`}
            >
              {isLoading && (
                <div className="absolute inset-0 bg-blue-100/80 dark:bg-blue-900/50 rounded-lg flex items-center justify-center backdrop-blur-sm">
                  <svg className="animate-spin h-5 w-5 text-blue-600 dark:text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                </div>
              )}
              <button className="flex-1 text-left min-w-0"
                onClick={(e) => selectConversion(e, conv)}
                disabled={isLoading || editingId === conv.id}
              >
                {editingId === conv.id ? (
                  <input
                    ref={inputRef}
                    type="text"
                    value={editingTitle}
                    onChange={(e) => setEditingTitle(e.target.value)}
                    onKeyDown={(e) => handleKeyDown(e, conv.id)}
                    onBlur={() => saveTitle(conv.id)}
                    onClick={(e) => e.stopPropagation()}
                    className="w-full px-2 py-1 text-sm font-medium border border-blue-500 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-neutral-700 dark:text-neutral-100 dark:border-blue-400"
                    disabled={isLoading}
                  />
                ) : (
                  <div>
                    <div className="font-medium truncate dark:text-neutral-100">
                      {conv.title}
                    </div>
                    <div className="text-xs text-neutral-500 dark:text-neutral-400">{new Date(conv.updated_at).toLocaleString()}</div>
                  </div>
                )}
              </button>
              <button
                onClick={(e) => startEditing(e, conv)}
                disabled={isLoading || editingId === conv.id}
                className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-yellow-100 dark:hover:bg-yellow-900/30 transition"
                title="Rename conversation"
              >
                <svg className="w-4 h-4 text-yellow-600 dark:text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                </svg>
              </button>
              <button
                onClick={(e) => removeConversation(e, conv.id)}
                disabled={isLoading || editingId === conv.id}
                className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-100 dark:hover:bg-red-900/30 transition"
                title="Delete conversation"
              >
                <svg className="w-4 h-4 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            </div>
          );
        })}
      </div>
      <hr className="border-neutral-200" />
      {/* Filters */}
      <SidebarFilters
      />
    </div>
  );
}