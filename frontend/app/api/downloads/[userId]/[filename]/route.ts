import { getServerSession } from 'next-auth';
import { authOptions } from "@/lib/auth";
import { NextRequest, NextResponse } from 'next/server';
import { mintServiceToken, ServiceAuthError } from "@/lib/service-auth";
import { randomUUID } from 'crypto';

export const runtime = 'nodejs';

/**
 * GET /api/downloads/[userId]/[filename]
 * Proxy route to Flask backend's download endpoint
 *
 * This route:
 * 1. Authenticates the user via NextAuth session
 * 2. Mints a service token for backend authentication
 * 3. Proxies the request to Flask backend: /downloads/{userId}/{filename}
 * 4. Streams the file response back to the client
 *
 * Security:
 * - User must be authenticated (NextAuth session)
 * - Backend verifies user can only download their own files
 * - Service token ensures secure backend communication
 */
export async function GET(
  req: NextRequest,
  { params }: { params: { userId: string; filename: string } }
) {
  // Authenticate user
  const session = await getServerSession(authOptions);
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Mint service token for backend authentication
  let serviceToken: string;
  try {
    serviceToken = mintServiceToken({
      email: session?.user?.email,
      dept: session?.user?.dept
    });
  } catch (error) {
    if (error instanceof ServiceAuthError) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    return NextResponse.json({ error: "Unable to mint service token" }, { status: 500 });
  }

  // Extract route parameters
  const { userId, filename } = params;

  // Proxy request to Flask backend
  const backendUrl = `${process.env.FLASK_URL}/downloads/${userId}/${filename}`;

  try {
    const response = await fetch(backendUrl, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${serviceToken}`,
        'X-Correlation-ID': randomUUID(),
      },
    });

    // If backend returns error (JSON), pass it through
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Download failed' }));
      return NextResponse.json(errorData, { status: response.status });
    }

    // Stream file response back to client
    // Preserve content-type, content-disposition, and other headers
    const contentType = response.headers.get('Content-Type') || 'application/octet-stream';
    const contentDisposition = response.headers.get('Content-Disposition');

    const headers: Record<string, string> = {
      'Content-Type': contentType,
    };

    if (contentDisposition) {
      headers['Content-Disposition'] = contentDisposition;
    }

    return new Response(response.body, {
      status: response.status,
      headers,
    });
  } catch (error: any) {
    console.error('Download proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to download file from backend' },
      { status: 500 }
    );
  }
}
