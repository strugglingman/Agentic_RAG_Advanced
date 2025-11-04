FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY prisma ./prisma
RUN npx prisma generate
COPY . .
EXPOSE 3001
ENV NODE_ENV=development
CMD ["sh", "-c", "npx prisma migrate deploy && npm run dev"]
