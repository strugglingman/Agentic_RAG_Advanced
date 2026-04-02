import { mintServiceToken, ServiceAuthError } from "@/lib/service-auth";

const originalEnv = { ...process.env };

describe("mintServiceToken", () => {
  beforeEach(() => {
    process.env = { ...originalEnv };
    process.env.SERVICE_AUTH_SECRET = "test-secret";
    process.env.SERVICE_AUTH_ISSUER = "test-issuer";
    process.env.SERVICE_AUTH_AUDIENCE = "test-audience";
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  it("mints token when email and dept are provided", () => {
    const token = mintServiceToken({ email: "user@example.com", dept: "eng" });
    expect(typeof token).toBe("string");
    expect(token.length).toBeGreaterThan(20);
  });

  it("throws ServiceAuthError when email is missing", () => {
    expect(() => mintServiceToken({ dept: "eng" })).toThrow(ServiceAuthError);
  });

  it("throws ServiceAuthError when dept is missing", () => {
    expect(() => mintServiceToken({ email: "user@example.com" })).toThrow(ServiceAuthError);
  });
});
