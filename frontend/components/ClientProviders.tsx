"use client";
import { FiltersProvider } from "@/components/filters-context";
import { SessionProvider } from "next-auth/react";
export default function ClientProviders({ children, session }: { children: React.ReactNode, session: any }) {
  return (
    <SessionProvider session={session}>
      <FiltersProvider>
        {children}
      </FiltersProvider>
    </SessionProvider>
  );
}
