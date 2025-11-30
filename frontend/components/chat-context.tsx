"use client";
import React, { createContext, useContext, useState, useEffect } from "react";

type Role = 'user' | 'assistant';
type Msg = { id?: string; role: Role; content: string; created_at?: string; ts?: number };
type Context = { bm25: number; chunk: string; chunk_id: string; dept_id: string;
  ext: string; file_for_user: boolean; file_id: string; hybrid: number; page: number; 
  rerank: number; sem_sim: number; size_kb: number; source: string; tags: string; 
  upload_at: string; uploaded_at_ts: number; user_id: string};
type Conversation = { id: string; user_email: string, title: string; created_at: string; updated_at: string; };

type ChatCtx = {
  messages: Array<Msg>;
  setMessages: React.Dispatch<React.SetStateAction<Array<Msg>>>;
  contexts: Array<Context>;
  setContexts: React.Dispatch<React.SetStateAction<Array<Context>>>;
  conversations: Array<Conversation>;
  setConversations: React.Dispatch<React.SetStateAction<Array<Conversation>>>;
  selectedConversation: Conversation | null;
  setSelectedConversation: React.Dispatch<React.SetStateAction<Conversation | null>>;
  isLoadingConversation: boolean;
  setIsLoadingConversation: React.Dispatch<React.SetStateAction<boolean>>;
};

const Ctx = createContext<ChatCtx | null>(null);

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
      //const saved = localStorage.getItem(STORAGE_KEYS.messages);
      const saved = null;
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      console.error('Failed to load messages from localStorage:', e);
      return [];
    }
  });

  const [contexts, setContexts] = useState<Array<Context>>(() => {
    if (typeof window === 'undefined') return [];
    try {
      // const saved = localStorage.getItem(STORAGE_KEYS.contexts);
      //return saved ? JSON.parse(saved) : [];
      return [];
    } catch (e) {
      console.error('Failed to load contexts from localStorage:', e);
      return [];
    }
  });

  const [conversations, setConversations] = useState<Array<Conversation>>([]);
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);
  const [isLoadingConversation, setIsLoadingConversation] = useState(false);

  // Fetch conversations on mount
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch('/api/conversations');
        if (res.ok) {
          const data = await res.json();
          setConversations(data.conversations || []);
        } else {
          console.log('Failed to fetch conversations list:', res.statusText);
        }
      } catch (e) {
        console.error('Error fetching conversations list:', e);
      }
    })();
  }, []);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    try {
      // localStorage.setItem(STORAGE_KEYS.messages, JSON.stringify(messages));
    } catch (e) {
      console.error('Failed to save messages to localStorage:', e);
    }
  }, [messages]);

  // Save contexts to localStorage whenever they change
  useEffect(() => {
    try {
      // localStorage.setItem(STORAGE_KEYS.contexts, JSON.stringify(contexts));
    } catch (e) {
      console.error('Failed to save contexts to localStorage:', e);
    }
  }, [contexts]);

  return (
    <Ctx.Provider value={{ messages, setMessages, contexts, setContexts, conversations, setConversations, selectedConversation, setSelectedConversation, isLoadingConversation, setIsLoadingConversation }}>
      {children}
    </Ctx.Provider>
  );
};