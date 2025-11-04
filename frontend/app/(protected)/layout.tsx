import type { ReactNode } from "react";
import { authOptions } from "@/lib/auth";
import { getServerSession } from "next-auth";
import { redirect } from "next/navigation";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import ClientProviders from "@/components/ClientProviders";

export default async function AppLayout({ children }: { children: ReactNode }) {
  const session = await getServerSession(authOptions);
  if (!session) {
    redirect("/");
  }

  return (
    <div className="h-screen w-full overflow-hidden">
      {/* Provider wraps the whole app area so Sidebar AND children share state */}
      <ClientProviders session={session}>
        <div className="grid lg:grid-cols-[320px_minmax(0,1fr)] h-full">
          <aside className="hidden lg:block border-r bg-white">
            <div className="h-full overflow-y-auto">
              <Sidebar /> {/* uses useFilters() for state management */}
            </div>
          </aside>

          <section className="min-w-0 overflow-hidden flex flex-col">
            <div className="border-b px-4 py-2">
              <Header />
            </div>
            <div className="flex-1 min-h-0 overflow-auto">
              {children} {/* ChatPage, Upload, etc. also use useFilters() */}
            </div>
          </section>
        </div>
      </ClientProviders>
    </div>
  );
}
