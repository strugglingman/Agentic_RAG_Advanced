#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env" ]]; then
  echo "Missing root .env file in $ROOT_DIR" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: ./scripts/restore_postgres.sh <backup.dump>" >&2
  exit 1
fi

BACKUP_FILE="$1"

if [[ ! -f "$BACKUP_FILE" ]]; then
  echo "Backup file not found: $BACKUP_FILE" >&2
  exit 1
fi

set -a
source .env
set +a

echo "This will replace data in database '${POSTGRES_DB}'."
read -r -p "Type RESTORE to continue: " CONFIRM

if [[ "$CONFIRM" != "RESTORE" ]]; then
  echo "Restore cancelled."
  exit 1
fi

echo "Restoring PostgreSQL backup from: $BACKUP_FILE"
docker compose exec -T postgres dropdb --if-exists -U "${POSTGRES_USER}" "${POSTGRES_DB}"
docker compose exec -T postgres createdb -U "${POSTGRES_USER}" "${POSTGRES_DB}"
cat "$BACKUP_FILE" | docker compose exec -T postgres pg_restore \
  -U "${POSTGRES_USER}" \
  -d "${POSTGRES_DB}" \
  --clean \
  --if-exists

echo "Restore completed."
