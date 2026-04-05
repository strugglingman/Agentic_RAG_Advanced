#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env" ]]; then
  echo "Missing root .env file in $ROOT_DIR" >&2
  exit 1
fi

set -a
source .env
set +a

BACKUP_DIR="${1:-$ROOT_DIR/backups/postgres}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_FILE="$BACKUP_DIR/chatbot_${TIMESTAMP}.dump"

mkdir -p "$BACKUP_DIR"

echo "Creating PostgreSQL backup at: $BACKUP_FILE"
docker compose exec -T postgres pg_dump \
  -U "${POSTGRES_USER}" \
  -d "${POSTGRES_DB}" \
  -Fc > "$BACKUP_FILE"

echo "Backup completed: $BACKUP_FILE"
