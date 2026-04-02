import { beforeEach, describe, expect, it, vi } from "vitest";

const mockGetServerSession = vi.fn();
const mockMintServiceToken = vi.fn();

class MockServiceAuthError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

vi.mock("next-auth", () => ({
  getServerSession: mockGetServerSession,
}));

vi.mock("@/lib/auth", () => ({
  authOptions: {},
}));

vi.mock("@/lib/service-auth", () => ({
  mintServiceToken: mockMintServiceToken,
  ServiceAuthError: MockServiceAuthError,
}));

import { POST } from "../app/api/ingest/route";


function makeJsonRequest(body = `{"file_ids":["ALL"]}`): Request {
  return new Request("http://localhost/api/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
}


describe("/api/ingest route", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal("fetch", vi.fn());
  });

  it("returns 401 when session is missing", async () => {
    mockGetServerSession.mockResolvedValue(null);

    const res = await POST(makeJsonRequest());
    expect(res.status).toBe(401);
  });

  it("returns ServiceAuthError status and message", async () => {
    mockGetServerSession.mockResolvedValue({
      user: { email: "user@example.com", dept: "ENG" },
    });
    mockMintServiceToken.mockImplementation(() => {
      throw new MockServiceAuthError("Department is required", 400);
    });

    const res = await POST(makeJsonRequest());
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toContain("Department is required");
  });

  it("fails closed on unexpected token mint errors", async () => {
    mockGetServerSession.mockResolvedValue({
      user: { email: "user@example.com", dept: "ENG" },
    });
    mockMintServiceToken.mockImplementation(() => {
      throw new Error("unexpected mint error");
    });

    const fetchMock = global.fetch as unknown as ReturnType<typeof vi.fn>;
    fetchMock.mockResolvedValue(
      new Response("{}", {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    );

    const res = await POST(makeJsonRequest());
    expect(res.status).toBe(500);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
