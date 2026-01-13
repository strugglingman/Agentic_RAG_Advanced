import { getServerSession } from 'next-auth';
import { authOptions } from "@/lib/auth";
import { NextRequest, NextResponse } from 'next/server';
import { mintServiceToken, ServiceAuthError } from "@/lib/service-auth";
import { randomUUID } from 'crypto';

export const runtime = 'nodejs';

/**
 * GET /api/files/[fileId]
 * Proxy route to FastAPI backend's unified file download endpoint
 *
 * This route:
 * 1. Authenticates the user via NextAuth session
 * 2. Mints a service token for backend authentication
 * 3. Proxies the request to FastAPI backend: /files/{fileId}
 * 4. Streams the file response back to the client
 *
 * Unified file access - works for ALL file types:
 * - Uploaded RAG documents
 * - Chat attachments
 * - Downloaded files
 * - Created documents
 *
 * Security:
 * - User must be authenticated (NextAuth session)
 * - Backend verifies file ownership through FileRegistry
 * - Service token ensures secure backend communication
 */
export async function GET(
  req: NextRequest,
  { params }: { params: { fileId: string } }
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

  // Extract file ID from route parameter
  const { fileId } = params;

  // Proxy request to FastAPI backend's unified file endpoint
  const backendUrl = `${process.env.FASTAPI_URL}/files/${fileId}`;

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
      const errorData = await response.json().catch(() => ({ error: 'File download failed' }));
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
    console.error('File download proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to download file from backend' },
      { status: 500 }
    );
  }
}
