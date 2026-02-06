# ---- Build stage ----
FROM node:20-alpine AS builder
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Generate Prisma client (needs DATABASE_URL format, not actual connection)
COPY prisma ./prisma
ENV DATABASE_URL="postgresql://user:pass@localhost:5432/db"
RUN npx prisma generate

# Copy source code and build
COPY . .
RUN npm run build

# ---- Production stage ----
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

# Copy standalone output (includes server + bundled node_modules)
COPY --from=builder /app/.next/standalone ./
# Copy static assets (not included in standalone by default)
COPY --from=builder /app/.next/static ./.next/static
# Copy Prisma schema + client (needed at runtime for DB access)
COPY --from=builder /app/node_modules/.prisma ./node_modules/.prisma
COPY --from=builder /app/node_modules/@prisma ./node_modules/@prisma
COPY --from=builder /app/prisma ./prisma

EXPOSE 3000

# standalone output produces server.js in the root
CMD ["node", "server.js"]
