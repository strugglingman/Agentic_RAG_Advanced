import { DefaultSession } from "next-auth";

declare module "next-auth" {
  interface Session {
    user: {
      id: string;
      dept?: string | null;
    } & DefaultSession["user"];
  }

  interface User {
    id: string;
    dept?: string | null;
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    id: string;
    dept?: string | null;
  }
}
