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
    // Protected routes
    "/chat/:path*",
    "/upload/:path*",
    "/api/chat",
    "/api/upload",
    "/api/ingest",
  ],
};
