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

if [[ -z "${S3_BACKUP_BUCKET:-}" ]]; then
  echo "Missing S3_BACKUP_BUCKET in root .env" >&2
  exit 1
fi

BACKUP_DIR="${BACKUP_DIR:-$ROOT_DIR/backups/postgres}"
S3_PREFIX="${S3_BACKUP_PREFIX:-postgres}"

if [[ $# -ge 1 ]]; then
  BACKUP_FILE="$1"
else
  BACKUP_FILE="$(ls -1t "$BACKUP_DIR"/chatbot_*.dump 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${BACKUP_FILE:-}" ]]; then
  echo "No PostgreSQL backup file found in $BACKUP_DIR" >&2
  exit 1
fi

if [[ ! -f "$BACKUP_FILE" ]]; then
  echo "Backup file not found: $BACKUP_FILE" >&2
  exit 1
fi

OBJECT_NAME="$(basename "$BACKUP_FILE")"
S3_URI="s3://${S3_BACKUP_BUCKET}/${S3_PREFIX}/${OBJECT_NAME}"

echo "Uploading PostgreSQL backup to: $S3_URI"
aws s3 cp "$BACKUP_FILE" "$S3_URI"
echo "Upload completed: $S3_URI"
