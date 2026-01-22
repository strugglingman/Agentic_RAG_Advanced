FROM node:20-alpine
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Generate Prisma client (needs DATABASE_URL format, not actual connection)
COPY prisma ./prisma
ENV DATABASE_URL="postgresql://user:pass@localhost:5432/db"
RUN npx prisma generate

# Copy source code
COPY . .

# Build Next.js app
RUN npm run build

EXPOSE 3000
ENV NODE_ENV=production

# Start production server (backend handles migrations)
CMD ["npm", "run", "start"]
