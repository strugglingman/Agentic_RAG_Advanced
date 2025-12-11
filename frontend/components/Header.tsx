"use client";
import { useSession, signOut } from "next-auth/react";
import { useTheme } from "./theme-context";

export default function Header() {
  const { data, status } = useSession();
  const { theme, toggleTheme } = useTheme();

  if (status === "loading") {
    return (<div className="text-sm text-neutral-500 dark:text-neutral-400">Checking session…</div>);
  } else if (status === "unauthenticated") {
    return (<div className="text-sm text-red-500">Unauthenticated</div>);
  }

  return (
    <div className="flex items-center gap-3 text-sm">
      <span className="text-neutral-600 dark:text-neutral-300">{data?.user?.email}</span>
      <span className="text-neutral-600 dark:text-neutral-300">{data?.user?.dept}</span>
      <button
        onClick={toggleTheme}
        className="rounded px-3 py-1 border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-700 transition"
        title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
      >
        {theme === 'light' ? '🌙' : '☀️'}
      </button>
      <button
        onClick={() => signOut({ callbackUrl: "/" })}
        className="rounded px-3 py-1 border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-700 transition"
      >
        Sign out
      </button>
    </div>
  );
}
