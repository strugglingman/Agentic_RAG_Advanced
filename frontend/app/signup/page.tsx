"use client";
import { useState } from "react";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function SignupPage() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [deptPath, setDeptPath] = useState<string>("");
    const [orgTree, setOrgTree] = useState<any>(null);
    const router = useRouter();

    useEffect(() => {
        (async () => {
            try {
                const res = await fetch('/api/org-structure');
                if (res.ok) {
                    const data = await res.json();
                    setOrgTree(data);
                }
            } catch (err) {
                console.error("Failed to fetch organization structure:", err);
            }
        })();
    }, []);

    function renderOrganizationTree(depts: any[], parentPath: string = ""): React.ReactNode {
        return depts.map((dept) => {
            const currentPath = parentPath ? `${parentPath}|${dept.id}` : dept.id;
            return (
                <div key={currentPath} style={{ marginLeft: parentPath ? "20px" : "0" }}>
                    <label className="flex items-center gap-2 cursor-pointer py-1">
                        <input
                            type="radio"
                            name="dept"
                            value={currentPath}
                            required
                            checked={deptPath === currentPath}
                            onChange={() => setDeptPath(currentPath)}
                        />
                        <span className="text-sm">{dept.name}</span>
                    </label>
                    {dept.children && renderOrganizationTree(dept.children, currentPath)}
                </div>
            );
        });
    }

    async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
        e.preventDefault();
        setLoading(true);
        setError(null);

        const formData = new FormData(e.currentTarget);
        const name = formData.get("name")?.toString().trim() || "";
        const email = formData.get("email")?.toString().trim() || "";
        const dept = formData.get("dept")?.toString().trim() || "";
        const password = formData.get("password")?.toString().trim() || "";
        try {
            const res = await fetch("/api/signup", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    name: name || "",
                    email: email || "",
                    dept: dept || "",
                    password: password || "",
                })
            });

            if (res.ok) {
                router.replace("/");
            } else {
                const err = await res.json();
                setError(err.error || "An unknown error occurred");
            }
        } catch (err) {
            setError((err as Error).message || "An unknown error occurred");
        } finally {
            setLoading(false);
        }
    }

    return (
        <main className="min-h-screen grid place-items-center p-6">
            <form onSubmit={onSubmit} className="w-80 space-y-3">
                <h1 className="text-xl font-semibold text-center">Create account</h1>
                {error && <div role="alert" className="border border-red-300 bg-red-50 text-red-700 px-3 py-2 rounded">{error}</div>}
                <input name="name" type="text" className="w-full border rounded px-3 py-2" placeholder="Name (optional)" />
                <input name="email" type="email" required className="w-full border rounded px-3 py-2" placeholder="you@org.com" />
                <div className="max-h-55 overflow-y-auto border p-2 rounded">
                    {orgTree && orgTree.departments && renderOrganizationTree(orgTree.departments, "")}
                </div>
                <input name="password" type="password" required minLength={8} className="w-full border rounded px-3 py-2" placeholder="Password (min 8)" />
                <button disabled={loading} className="w-full bg-black text-white py-2 rounded">
                    {loading ? "Creating…" : "Create account"}
                </button>
                <Link href="/" className="block text-center text-sm text-gray-600 hover:underline">
                    Already have an account? Log in
                </Link>
            </form>
        </main>
    );
}
