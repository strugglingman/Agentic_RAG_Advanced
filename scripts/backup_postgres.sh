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
POSTGRES_BACKUP_MODE="${POSTGRES_BACKUP_MODE:-direct}"
POSTGRES_CLIENT_IMAGE="${POSTGRES_CLIENT_IMAGE:-postgres:15-alpine}"

mkdir -p "$BACKUP_DIR"

echo "Creating PostgreSQL backup at: $BACKUP_FILE"

case "$POSTGRES_BACKUP_MODE" in
  compose)
    docker compose exec -T postgres pg_dump \
      -U "${POSTGRES_USER}" \
      -d "${POSTGRES_DB}" \
      -Fc > "$BACKUP_FILE"
    ;;
  direct)
    : "${POSTGRES_HOST:?Missing POSTGRES_HOST in root .env}"
    : "${POSTGRES_PORT:?Missing POSTGRES_PORT in root .env}"
    : "${POSTGRES_USER:?Missing POSTGRES_USER in root .env}"
    : "${POSTGRES_PASSWORD:?Missing POSTGRES_PASSWORD in root .env}"
    : "${POSTGRES_DB:?Missing POSTGRES_DB in root .env}"

    docker run --rm \
      -e "PGPASSWORD=${POSTGRES_PASSWORD}" \
      "$POSTGRES_CLIENT_IMAGE" \
      pg_dump \
      -h "${POSTGRES_HOST}" \
      -p "${POSTGRES_PORT}" \
      -U "${POSTGRES_USER}" \
      -d "${POSTGRES_DB}" \
      -Fc \
      -w > "$BACKUP_FILE"
    ;;
  *)
    echo "Unsupported POSTGRES_BACKUP_MODE: $POSTGRES_BACKUP_MODE" >&2
    echo "Supported values: compose, direct" >&2
    exit 1
    ;;
esac

echo "Backup completed: $BACKUP_FILE"
