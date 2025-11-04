import { cookies } from 'next/headers';
import { NextRequest, NextResponse } from 'next/server';
import { authOptions } from "@/lib/auth";
import { getServerSession } from 'next-auth';
export const runtime = 'nodejs';

function getOrCreateSid() {
    const jar = cookies();
    let sid = jar.get('sid')?.value;
    if (!sid) {
        sid = crypto.randomUUID();
        jar.set('sid', sid, { httpOnly: true, sameSite: 'lax', path: '/', maxAge: 60 * 60 * 24 * 7 });
    }
    return sid;
}

export async function POST(req: NextRequest) {
    const session = await getServerSession(authOptions);
    if (!session) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const sid = getOrCreateSid();
    const r = await fetch(`${process.env.FLASK_URL}/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'x-session-id': sid,
            'x-user-id': session.user?.email || '',
            'x-dept-id': session.user?.dept || '',
        },
        body: JSON.stringify(body)
    });
    // Pass through streaming body and content type
    const ct = r.headers.get('Content-Type') || 'text/plain; charset=utf-8';
    return new Response(r.body, { status: r.status, headers: { 'Content-Type': ct } });
}