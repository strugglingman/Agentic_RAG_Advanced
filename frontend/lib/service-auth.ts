import jwt from "jsonwebtoken";

export class ServiceAuthError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ServiceAuthError";
    this.status = status;
  }
}

export function mintServiceToken(args: {
  email?: string | null;
  dept?: string | null;
}) {
  const secret = process.env.SERVICE_AUTH_SECRET;
  if (!secret) {
    throw new ServiceAuthError("SERVICE_AUTH_SECRET is not configured", 500);
  }
  if (!args.email) {
    throw new ServiceAuthError("Email is required to mint service token", 400);
  }
  if (!args.dept) {
    throw new ServiceAuthError("Department is required to mint service token", 400);
  }

  const issuer = process.env.SERVICE_AUTH_ISSUER ?? "chat-frontend";
  const audience = process.env.SERVICE_AUTH_AUDIENCE ?? "flask-backend";

  return jwt.sign(
    {
      email: args.email ?? null,
      dept: args.dept,
    },
    secret,
    {
      algorithm: "HS256",
      expiresIn: "5m",
      issuer,
      audience,
    }
  );
}
