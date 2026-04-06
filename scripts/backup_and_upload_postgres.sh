#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BACKUP_DIR="${1:-$ROOT_DIR/backups/postgres}"

echo "Starting PostgreSQL backup..."
"$ROOT_DIR/scripts/backup_postgres.sh" "$BACKUP_DIR"

BACKUP_FILE="$(ls -1t "$BACKUP_DIR"/chatbot_*.dump 2>/dev/null | head -n 1 || true)"

if [[ -z "${BACKUP_FILE:-}" ]]; then
  echo "Backup completed but no dump file was found in $BACKUP_DIR" >&2
  exit 1
fi

echo "Uploading latest PostgreSQL backup..."
"$ROOT_DIR/scripts/upload_postgres_backup_to_s3.sh" "$BACKUP_FILE"

echo "Backup and upload completed: $BACKUP_FILE"
