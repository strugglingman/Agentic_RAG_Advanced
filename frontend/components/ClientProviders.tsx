"use client";
import { FiltersProvider } from "@/components/filters-context";
import { ChatProvider } from "@/components/chat-context";
import { ThemeProvider } from "@/components/theme-context";
import { SessionProvider } from "next-auth/react";
export default function ClientProviders({ children, session }: { children: React.ReactNode, session: any }) {
  return (
    <SessionProvider session={session}>
      <ThemeProvider>
        <FiltersProvider>
          <ChatProvider>
            {children}
          </ChatProvider>
        </FiltersProvider>
      </ThemeProvider>
    </SessionProvider>
  );
}
