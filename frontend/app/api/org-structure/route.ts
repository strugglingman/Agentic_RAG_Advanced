import { NextResponse } from 'next/server';
import { randomUUID } from 'crypto';

export const runtime = 'nodejs';

export async function GET() {
    try {
        const r = await fetch(`${process.env.FLASK_URL}/org-structure`, {
            headers: {
                'X-Correlation-ID': randomUUID(),
            }
        });
        if (!r.ok) {
            return NextResponse.json({ error: "Failed to fetch organization structure" }, { status: r.status });
        }
        const data = await r.json();
        return NextResponse.json(data, { status: 200 });
    } catch (error) {
        return NextResponse.json({ error: "Failed to fetch organization structure" }, { status: 500 });
    }
}
