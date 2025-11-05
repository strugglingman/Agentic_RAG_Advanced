"use client";
import { FiltersProvider } from "@/components/filters-context";
import { ChatProvider } from "@/components/chat-context";
import { SessionProvider } from "next-auth/react";
export default function ClientProviders({ children, session }: { children: React.ReactNode, session: any }) {
  return (
    <SessionProvider session={session}>
      <FiltersProvider>
        <ChatProvider>  
          {children}
        </ChatProvider>
      </FiltersProvider>
    </SessionProvider>
  );
}
