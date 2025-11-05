"use client";
import React, { createContext, useContext, useState, useEffect } from "react";

type Role = 'user' | 'assistant';
type Msg = { role: Role; content: string; ts?: number };
type Context = { bm25: number; chunk: string; chunk_id: string; dept_id: string;
  ext: string; file_for_user: boolean; file_id: string; hybrid: number; page: number; 
  rerank: number; sem_sim: number; size_kb: number; source: string; tags: string; 
  upload_at: string; uploaded_at_ts: number; user_id: string};

type ChatCtx = {
  messages: Array<Msg>;
  setMessages: React.Dispatch<React.SetStateAction<Array<Msg>>>;
  contexts: Array<Context>;
  setContexts: React.Dispatch<React.SetStateAction<Array<Context>>>;
  clearChat: () => void;
};

const Ctx = createContext<ChatCtx | null>(null);

const STORAGE_KEYS = {
  messages: 'chat-messages',
  contexts: 'chat-contexts',
};

export function useChat() {
  const context = useContext(Ctx);
  if (!context) {
    throw new Error("useChat must be used within a ChatProvider");
  }

  return context;
};

export function ChatProvider({ children }: { children: React.ReactNode }) {
  // Initialize from localStorage
  const [messages, setMessages] = useState<Array<Msg>>(() => {
    if (typeof window === 'undefined') return [];
    try {
      const saved = localStorage.getItem(STORAGE_KEYS.messages);
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      console.error('Failed to load messages from localStorage:', e);
      return [];
    }
  });

  const [contexts, setContexts] = useState<Array<Context>>(() => {
    if (typeof window === 'undefined') return [];
    try {
      const saved = localStorage.getItem(STORAGE_KEYS.contexts);
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      console.error('Failed to load contexts from localStorage:', e);
      return [];
    }
  });

  // Save messages to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEYS.messages, JSON.stringify(messages));
    } catch (e) {
      console.error('Failed to save messages to localStorage:', e);
    }
  }, [messages]);

  // Save contexts to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEYS.contexts, JSON.stringify(contexts));
    } catch (e) {
      console.error('Failed to save contexts to localStorage:', e);
    }
  }, [contexts]);

  // Clear function to reset chat
  const clearChat = () => {
    setMessages([]);
    setContexts([]);
    localStorage.removeItem(STORAGE_KEYS.messages);
    localStorage.removeItem(STORAGE_KEYS.contexts);
  };

  return (
    <Ctx.Provider value={{ messages, setMessages, contexts, setContexts, clearChat }}>
      {children}
    </Ctx.Provider>
  );
};