import { withAuth } from "next-auth/middleware";
import { NextResponse } from "next/server";

export default withAuth(
  // for authorized requests
  function middleware(req) {
    return NextResponse.next();
  },
  {
    callbacks: {
      // check if the route is authorized, redirect to sign in page if not, to middleware callback otherwise
      authorized: ({ token, req }) => {
        return !!token;
      },
    },
    pages: {
      signIn: "/",
    },
  }
);

export const config = {
  matcher: [
    // Protected pages
    "/chat/:path*",
    "/upload/:path*",
    // Protected API routes
    "/api/chat/:path*",
    "/api/upload/:path*",
    "/api/ingest/:path*",
    "/api/files/:path*",
    "/api/conversations/:path*",
    // /api/org-structure is NOT protected — it is needed by the user registration
  ],
};
