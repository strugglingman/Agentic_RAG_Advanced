import { getServerSession } from 'next-auth';
import { authOptions } from "@/lib/auth";
import { NextResponse } from 'next/server';
import { mintServiceToken, ServiceAuthError } from "@/lib/service-auth";
import { randomUUID } from 'crypto';

export const runtime = 'nodejs'
export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
    const { id } = await params;
    const session = await getServerSession(authOptions);
    if (!session) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    let token: string = "";
    try {
        token = mintServiceToken({ email: session?.user?.email, dept: session?.user?.dept});
    } catch (error) {
        if (error instanceof ServiceAuthError) {
            return NextResponse.json({ error: error.message }, { status: error.status });
        }
    }
    const r = await fetch(`${process.env.FASTAPI_URL}/conversations/${id}`, {
        headers: {
            'Authorization': `Bearer ${token}`,
            'X-Correlation-ID': randomUUID(),
        }
    });

    return new Response(r.body, { status: r.status, headers: { 'Content-Type': 'application/json' } })
}

export async function PATCH(req: Request, { params }: { params: Promise<{ id: string }> }) {
    const { id } = await params;
    const sessions = await getServerSession(authOptions);
    if (!sessions) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    let token: string = "";
    try {
        token = mintServiceToken({ email: sessions?.user?.email, dept: sessions?.user?.dept});
    } catch (error) {
        if (error instanceof ServiceAuthError) {
            return NextResponse.json({ error: error.message }, { status: error.status });
        }
    }

    const payload = await req.json();
    const r = await fetch(`${process.env.FASTAPI_URL}/conversations/${id}`, {
        method: 'PATCH',
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
            'X-Correlation-ID': randomUUID(),
        },
        body: JSON.stringify(payload),
    });

    return new Response(r.body, { status: r.status, headers: { 'Content-Type': 'application/json' } });
}

export async function DELETE(req: Request, { params }: { params: Promise<{ id: string }> }) {
    const { id } = await params;
    const sessions = await getServerSession(authOptions);
    if (!sessions) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    let token: string = "";
    try {
        token = mintServiceToken({ email: sessions?.user?.email, dept: sessions?.user?.dept});
    } catch (error) {
        if (error instanceof ServiceAuthError) {
            return NextResponse.json({ error: error.message }, { status: error.status });
        }
    }

    const r = await fetch(`${process.env.FASTAPI_URL}/conversations/${id}`, {
        method: 'DELETE',
        headers: {
            'Authorization': `Bearer ${token}`,
            'X-Correlation-ID': randomUUID(),
        }
    });

    return new Response(r.body, { status: r.status, headers: { 'Content-Type': 'application/json' } })
}