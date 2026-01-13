import { NextRequest, NextResponse } from 'next/server';
import { authOptions } from "@/lib/auth";
import { getServerSession } from 'next-auth';
import { mintServiceToken, ServiceAuthError } from "@/lib/service-auth";
import { randomUUID } from 'crypto';

export const runtime = 'nodejs';


export async function POST(req: NextRequest) {
    const session = await getServerSession(authOptions);
    if (!session) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();

    let serviceToken: string;
    try {
        serviceToken = mintServiceToken({ email: session?.user?.email, dept: session?.user?.dept });
    } catch (error) {
        if (error instanceof ServiceAuthError) {
            return NextResponse.json({ error: error.message }, { status: error.status });
        }
        return NextResponse.json({ error: "Unable to mint service token" }, { status: 500 });
    }

    const r = await fetch(`${process.env.FASTAPI_URL}/chat/resume`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${serviceToken}`,
            'X-Correlation-ID': randomUUID(),
        },
        body: JSON.stringify(body)
    });
    // Pass through streaming body and content type
    const ct = r.headers.get('Content-Type') || 'text/plain; charset=utf-8';
    return new Response(r.body, { status: r.status, headers: { 'Content-Type': ct } });
}
