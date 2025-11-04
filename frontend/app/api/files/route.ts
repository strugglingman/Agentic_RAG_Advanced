import { cookies } from 'next/headers'
import { getServerSession } from 'next-auth';
import { authOptions } from "@/lib/auth";
import { NextResponse } from 'next/server';

export const runtime = 'nodejs'
export async function GET(req: Request) {
    const session = await getServerSession(authOptions);
    if (!session) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const jar = cookies();
    let sid = jar.get('sid')?.value;
    if (!sid) {
        sid = crypto.randomUUID();
        jar.set('sid', sid, { httpOnly: true, sameSite: 'lax', path: '/', maxAge: 60 * 60 * 24 * 7 });
    }

    const r = await fetch(`${process.env.FLASK_URL}/files`, {
        headers: {
            'x-session-id': sid,
            'x-dept-id': session.user?.dept || '',
            'x-user-id': session.user?.email || '',
        }
    })

    return new Response(r.body, { status: r.status, headers: { 'Content-Type': 'application/json' } })
}