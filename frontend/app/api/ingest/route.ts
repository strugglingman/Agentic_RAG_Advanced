import { getServerSession } from 'next-auth';
import { authOptions } from "@/lib/auth";
import { NextResponse } from 'next/server';
import { mintServiceToken, ServiceAuthError } from '@/lib/service-auth';
import { randomUUID } from 'crypto';

export const runtime = 'nodejs';

// Ingestion can take a long time with semantic chunking - 10 minute timeout for SSE
const INGEST_TIMEOUT_MS = process.env.INGEST_TIMEOUT_MS ? parseInt(process.env.INGEST_TIMEOUT_MS) : 10 * 60 * 1000;

export async function POST(req: Request) {
    const session = await getServerSession(authOptions);
    if (!session) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.text();
    let token: string = "";
    try {
        token = mintServiceToken({ email: session?.user?.email, dept: session?.user?.dept });
    } catch (error) {
        if (error instanceof ServiceAuthError) {
            return NextResponse.json({ error: error.message }, { status: error.status });
        }
    }

    // Use AbortController for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), INGEST_TIMEOUT_MS);

    try {
        const r = await fetch(
            `${process.env.FASTAPI_URL}/ingest`,
            {
                method: 'POST',
                body: body,
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`,
                    'X-Correlation-ID': randomUUID(),
                    'Accept': 'text/event-stream',
                },
                signal: controller.signal,
            },
        );
        clearTimeout(timeoutId);

        // Forward SSE stream with job ID header
        const jobId = r.headers.get('X-Job-ID');
        const headers: Record<string, string> = {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        };
        if (jobId) {
            headers['X-Job-ID'] = jobId;
        }

        return new Response(r.body, {
            status: r.status,
            headers,
        });
    } catch (error: any) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') {
            return NextResponse.json(
                { error: 'Ingestion timed out. The process may still be running on the server.' },
                { status: 504 }
            );
        }
        throw error;
    }
}
