"use client";
import SidebarFilters from "./SidebarFilters";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useChat } from "./chat-context";

const navItems = [
  { id: "chat",   label: "Chat",   href: "/chat" },
  { id: "upload", label: "Upload", href: "/upload" },
  { id: "files",  label: "Files",  href: "/files" },
  { id: "settings", label: "Settings", href: "/settings" },
];

export default function Sidebar() {
  const { conversations, setConversations, selectedConversation, setSelectedConversation, setMessages, setContexts } = useChat();
  const pathname = usePathname();

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
    }
  }

  return (
    <div className="p-3 space-y-6">
      <div className="px-2">
        <h2 className="text-xl font-semibold tracking-tight">RAG Chatbot</h2>
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
          return (
            <div
              key={conv.id}
              className={`group relative flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition ${
                isSelected
                  ? 'bg-blue-100 dark:bg-blue-900/30 border border-blue-300 dark:border-blue-700'
                  : 'hover:bg-neutral-100 dark:hover:bg-neutral-700'
              }`}
            >
              <button className="flex-1 text-left min-w-0"
                onClick={(e) => selectConversion(e, conv)}
              >
                <div className="font-medium truncate dark:text-neutral-100">{conv.title}</div>
                <div className="text-xs text-neutral-500 dark:text-neutral-400">{new Date(conv.updated_at).toLocaleString()}</div>
              </button>
              <button
                onClick={(e) => removeConversation(e, conv.id)}
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