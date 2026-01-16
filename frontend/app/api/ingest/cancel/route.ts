import { getServerSession } from 'next-auth';
import { authOptions } from "@/lib/auth";
import { NextResponse } from 'next/server';
import { mintServiceToken, ServiceAuthError } from '@/lib/service-auth';
import { randomUUID } from 'crypto';

export const runtime = 'nodejs';

export async function POST(req: Request) {
    const session = await getServerSession(authOptions);
    if (!session) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    let token: string = "";
    try {
        token = mintServiceToken({ email: session?.user?.email, dept: session?.user?.dept });
    } catch (error) {
        if (error instanceof ServiceAuthError) {
            return NextResponse.json({ error: error.message }, { status: error.status });
        }
    }

    try {
        const r = await fetch(`${process.env.FASTAPI_URL}/ingest/cancel`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
                'X-Correlation-ID': randomUUID(),
            },
            body: JSON.stringify(body),
        });

        const data = await r.json();
        return NextResponse.json(data, { status: r.status });
    } catch (error: any) {
        return NextResponse.json({ error: error.message || 'Cancel failed' }, { status: 500 });
    }
}
