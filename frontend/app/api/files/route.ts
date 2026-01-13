import { cookies } from 'next/headers'
import { getServerSession } from 'next-auth';
import { authOptions } from "@/lib/auth";
import { NextResponse } from 'next/server';
import { mintServiceToken, ServiceAuthError } from "@/lib/service-auth";
import { randomUUID } from 'crypto';

export const runtime = 'nodejs'
export async function GET(req: Request) {
    const session = await getServerSession(authOptions);
    if (!session) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    let token: string = "";
    try {
        token = mintServiceToken({ email: session?.user?.email, dept: session?.user?.dept });
    } catch (error) {
        if (error instanceof ServiceAuthError) {
            return NextResponse.json({ error: error.message }, { status: error.status });
        }
    }
    const r = await fetch(`${process.env.FASTAPI_URL}/files`, {
        headers: {
            'Authorization': `Bearer ${token}`,
            'X-Correlation-ID': randomUUID(),
        }
    })

    return new Response(r.body, { status: r.status, headers: { 'Content-Type': 'application/json' } })
}