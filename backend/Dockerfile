FROM python:3.12-slim
WORKDIR /app

# System packages (compilers, libs, PostgreSQL client)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ make git ca-certificates \
    libgomp1 libmagic1 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade Python build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.linux.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser (chromium only, ~150MB)
# Must be after pip install since playwright package is needed first
RUN playwright install chromium

# Install browser system dependencies (fonts, graphics libs, etc.)
# This runs apt-get internally to install what chromium needs
RUN playwright install-deps chromium

# Generate Prisma client (Python version)
# DATABASE_URL needed for prisma generate (just parses format, no actual connection)
COPY prisma ./prisma
ENV DATABASE_URL="postgresql://user:pass@localhost:5432/db"
RUN prisma generate

# Create necessary directories for runtime
RUN mkdir -p /app/uploads /app/downloads /app/logs /app/chroma_db

# Copy application code
COPY . .

EXPOSE 5001

# Run migrations then start server
CMD ["sh", "-c", "prisma migrate deploy && python run_fastapi.py"]
