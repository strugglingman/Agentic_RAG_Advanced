import { getServerSession } from 'next-auth';
import { authOptions } from "@/lib/auth";
import { NextRequest, NextResponse } from 'next/server';
import { mintServiceToken, ServiceAuthError } from "@/lib/service-auth";
import { randomUUID } from 'crypto';

export const runtime = 'nodejs';

/**
 * POST /api/files/delete
 * Proxy route to FastAPI backend's batch file deletion endpoint
 *
 * Request body:
 * {
 *   "file_ids": ["id1", "id2", ...],
 *   "remove_vectors": boolean (optional, default false)
 * }
 *
 * Response:
 * {
 *   "success": boolean,
 *   "total_deleted": number,
 *   "total_chunks_deleted": number,
 *   "results": [{ file_id, success, message, chunks_deleted }, ...]
 * }
 */
export async function POST(req: NextRequest) {
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

  // Parse request body
  let body: { file_ids: string[]; remove_vectors?: boolean };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  // Validate file_ids
  if (!body.file_ids || !Array.isArray(body.file_ids) || body.file_ids.length === 0) {
    return NextResponse.json({ error: "file_ids array is required" }, { status: 400 });
  }

  // Proxy request to FastAPI backend
  const backendUrl = `${process.env.FASTAPI_URL}/files/delete`;

  try {
    const response = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${serviceToken}`,
        'Content-Type': 'application/json',
        'X-Correlation-ID': randomUUID(),
      },
      body: JSON.stringify({
        file_ids: body.file_ids,
        remove_vectors: body.remove_vectors ?? false,
      }),
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error: any) {
    console.error('Delete files proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to delete files from backend' },
      { status: 500 }
    );
  }
}
