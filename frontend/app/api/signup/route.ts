import { prisma } from '@/lib/prisma';
import { NextRequest, NextResponse } from 'next/server';
import bcrypt from 'bcryptjs';

export async function POST(req: NextRequest) {
    const { name, email, dept, password } = await req.json();
    if (!email || !dept || !password || password.length < 8) {
        return NextResponse.json({ error: 'Invalid input' }, { status: 400 });
    }

    let user = await prisma.user.findUnique({where: { email: email }});
    if (user) {
        return NextResponse.json({ error: 'User already exists' }, { status: 400 });
    }

    const passwordHash = await bcrypt.hash(password, 10);
    user = await prisma.user.create({
        data: {
            name: name,
            email: email,
            dept: dept,
            passwordHash: passwordHash,
        }
    });

    if (!user) {
        return NextResponse.json({ error: 'Failed to create user' }, { status: 500 });
    }

    return NextResponse.json({ message: 'User created successfully', id: user.id, email: user.email }, { status: 201 });
}