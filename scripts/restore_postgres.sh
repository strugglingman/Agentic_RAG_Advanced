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

POSTGRES_BACKUP_MODE="${POSTGRES_BACKUP_MODE:-direct}"
POSTGRES_CLIENT_IMAGE="${POSTGRES_CLIENT_IMAGE:-postgres:15-alpine}"

run_direct_client() {
  docker run --rm -i \
    -e "PGPASSWORD=${POSTGRES_PASSWORD}" \
    "$POSTGRES_CLIENT_IMAGE" \
    "$@"
}

echo "This will replace data in database '${POSTGRES_DB}'."
read -r -p "Type RESTORE to continue: " CONFIRM

if [[ "$CONFIRM" != "RESTORE" ]]; then
  echo "Restore cancelled."
  exit 1
fi

echo "Restoring PostgreSQL backup from: $BACKUP_FILE"

case "$POSTGRES_BACKUP_MODE" in
  compose)
    docker compose exec -T postgres dropdb --if-exists -U "${POSTGRES_USER}" "${POSTGRES_DB}"
    docker compose exec -T postgres createdb -U "${POSTGRES_USER}" "${POSTGRES_DB}"
    cat "$BACKUP_FILE" | docker compose exec -T postgres pg_restore \
      -U "${POSTGRES_USER}" \
      -d "${POSTGRES_DB}" \
      --clean \
      --if-exists
    ;;
  direct)
    : "${POSTGRES_HOST:?Missing POSTGRES_HOST in root .env}"
    : "${POSTGRES_PORT:?Missing POSTGRES_PORT in root .env}"
    : "${POSTGRES_USER:?Missing POSTGRES_USER in root .env}"
    : "${POSTGRES_PASSWORD:?Missing POSTGRES_PASSWORD in root .env}"
    : "${POSTGRES_DB:?Missing POSTGRES_DB in root .env}"

    run_direct_client dropdb \
      -h "${POSTGRES_HOST}" \
      -p "${POSTGRES_PORT}" \
      -U "${POSTGRES_USER}" \
      --if-exists \
      "${POSTGRES_DB}"

    run_direct_client createdb \
      -h "${POSTGRES_HOST}" \
      -p "${POSTGRES_PORT}" \
      -U "${POSTGRES_USER}" \
      "${POSTGRES_DB}"

    cat "$BACKUP_FILE" | run_direct_client pg_restore \
      -h "${POSTGRES_HOST}" \
      -p "${POSTGRES_PORT}" \
      -U "${POSTGRES_USER}" \
      -d "${POSTGRES_DB}" \
      --clean \
      --if-exists
    ;;
  *)
    echo "Unsupported POSTGRES_BACKUP_MODE: $POSTGRES_BACKUP_MODE" >&2
    echo "Supported values: compose, direct" >&2
    exit 1
    ;;
esac

echo "Restore completed."
