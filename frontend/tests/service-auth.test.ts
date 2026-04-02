import jwt from "jsonwebtoken";
import { beforeEach, describe, expect, it } from "vitest";

import { ServiceAuthError, mintServiceToken } from "../lib/service-auth";

describe("mintServiceToken", () => {
  beforeEach(() => {
    process.env.SERVICE_AUTH_SECRET = "test-secret";
    process.env.SERVICE_AUTH_ISSUER = "test-issuer";
    process.env.SERVICE_AUTH_AUDIENCE = "test-audience";
  });

  it("throws ServiceAuthError when secret is missing", () => {
    delete process.env.SERVICE_AUTH_SECRET;

    expect(() =>
      mintServiceToken({ email: "user@example.com", dept: "ENG" })
    ).toThrow(ServiceAuthError);
  });

  it("throws ServiceAuthError when email is missing", () => {
    expect(() => mintServiceToken({ email: "", dept: "ENG" })).toThrow(
      ServiceAuthError
    );
  });

  it("throws ServiceAuthError when dept is missing", () => {
    expect(() =>
      mintServiceToken({ email: "user@example.com", dept: "" })
    ).toThrow(ServiceAuthError);
  });

  it("mints a token with expected identity claims", () => {
    const token = mintServiceToken({
      email: "alice@example.com",
      dept: "SALES",
    });

    const decoded = jwt.verify(token, process.env.SERVICE_AUTH_SECRET as string, {
      algorithms: ["HS256"],
      issuer: "test-issuer",
      audience: "test-audience",
    }) as { email: string; dept: string };

    expect(decoded.email).toBe("alice@example.com");
    expect(decoded.dept).toBe("SALES");
  });
});
