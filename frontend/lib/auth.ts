import { type NextAuthOptions } from "next-auth";
import Credentials from "next-auth/providers/credentials";
import bcrypt from "bcryptjs";
import { prisma } from "@/lib/prisma";

export const authOptions: NextAuthOptions = {
  secret: process.env.NEXTAUTH_SECRET,
  session: { strategy: "jwt" },
  pages: {
    signIn: "/",
  },
  providers: [
    Credentials({
      name: "Credentials",
      credentials: { email: {}, password: {} },
      async authorize(creds) {
        const { email, password } = creds as { email: string; password: string };
        const user = await prisma.user.findUnique({ where: { email: email } });
        if (!user || !user.passwordHash) {
          return null;
        }

        const match = await bcrypt.compare(password, user.passwordHash);
        if (user && match) {
          return { id: user.id, email: user.email, dept: user.dept };
        }

        return null;
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id;
        token.dept = (user as any).dept;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) { 
        (session.user as any).id = token.id;
        (session.user as any).dept = token.dept;
      }
      return session;
    },
  },
};
