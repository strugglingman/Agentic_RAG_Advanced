"use client";
import { useSession, signOut } from "next-auth/react";

export default function Header() {
  const { data, status } = useSession();
  if (status === "loading") {
    return (<div className="text-sm text-neutral-500">Checking session…</div>);
  } else if (status === "unauthenticated") {
    return (<div className="text-sm text-red-500">Unauthenticated</div>);
  }

  return (
    <div className="flex items-center gap-3 text-sm">
      <span className="text-neutral-600">{data?.user?.email}</span>
      <span className="text-neutral-600">{data?.user?.dept}</span>
      <button
        onClick={() => signOut({ callbackUrl: "/" })}
        className="rounded px-3 py-1 border"
      >
        Sign out
      </button>
    </div>
  );
}
