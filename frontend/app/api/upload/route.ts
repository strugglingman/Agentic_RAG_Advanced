import { cookies } from 'next/headers'
import { getServerSession } from 'next-auth';
import { authOptions } from "@/lib/auth";
import { NextResponse } from 'next/server';
import { mintServiceToken, ServiceAuthError } from '@/lib/service-auth';

export const runtime = 'nodejs'

export async function POST(req: Request) {
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

    const form = await req.formData();
    let token: string = "";
    try {
        token = mintServiceToken({ email: session?.user?.email, dept: session?.user?.dept, sid });
    } catch (error) {
        if (error instanceof ServiceAuthError) {
            return NextResponse.json({ error: error.message }, { status: error.status });
        }
    }
    const r = await fetch(`${process.env.FLASK_URL}/upload`, {
        method: 'POST',
        body: form,
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })

    const ct = r.headers.get('Content-Type') ?? 'application/json'
    return new Response(r.body, { status: r.status, headers: { 'Content-Type': ct } })
}