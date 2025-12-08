"use client";
import { signIn } from "next-auth/react";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function LoginPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setLoading(true);
    const f = new FormData(e.currentTarget);
    const email = String(f.get("email") || "");
    const password = String(f.get("password") || "");
    const res = await signIn("credentials", {
      email, password,
      redirect: false,
      callbackUrl: "/chat",
    });

    if (res?.ok) {
      router.replace(res.url ?? "/chat");
    } else {
      setError("Invalid email or password");
    }

    setLoading(false);
  }

  return (
    <main className="min-h-screen grid place-items-center p-6">
      <form onSubmit={onSubmit} className="w-80 space-y-3">
        <h1 className="text-xl font-semibold text-center">Agentic RAG Chatbot — Login</h1>
        <input name="email" type="email" className="w-full border p-2 rounded" placeholder="you@org.com" required />
        <input name="password" type="password" className="w-full border p-2 rounded" placeholder="Password" required />
        <button disabled={loading} className="w-full bg-black text-white py-2 rounded">
          {loading ? "Signing in..." : "Sign in"}
        </button>
        <Link href="/signup" className="block text-center text-sm text-gray-600 hover:underline">
          Don't have an account? Sign up
        </Link>
        {error && (
        <div
          role="alert"
          aria-live="assertive"
          className="rounded-md border border-red-300 bg-red-50 text-red-700 px-3 py-2 text-sm"
        >
          {error}
        </div>
      )}
      </form>
    </main>
  );
}
