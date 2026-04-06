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

DOWNLOAD_DIR="${DOWNLOAD_DIR:-$ROOT_DIR/backups/postgres}"
S3_PREFIX="${S3_BACKUP_PREFIX:-postgres}"

mkdir -p "$DOWNLOAD_DIR"

if [[ $# -ge 1 ]]; then
  OBJECT_KEY="$1"
else
  OBJECT_KEY="$(aws s3 ls "s3://${S3_BACKUP_BUCKET}/${S3_PREFIX}/" | awk '{print $4}' | tail -n 1 || true)"
fi

if [[ -z "${OBJECT_KEY:-}" ]]; then
  echo "No PostgreSQL backup object found in s3://${S3_BACKUP_BUCKET}/${S3_PREFIX}/" >&2
  exit 1
fi

if [[ "$OBJECT_KEY" == */* ]]; then
  S3_URI="s3://${S3_BACKUP_BUCKET}/${OBJECT_KEY}"
  LOCAL_FILE="$DOWNLOAD_DIR/$(basename "$OBJECT_KEY")"
else
  S3_URI="s3://${S3_BACKUP_BUCKET}/${S3_PREFIX}/${OBJECT_KEY}"
  LOCAL_FILE="$DOWNLOAD_DIR/${OBJECT_KEY}"
fi

echo "Downloading PostgreSQL backup from: $S3_URI"
aws s3 cp "$S3_URI" "$LOCAL_FILE"
echo "Download completed: $LOCAL_FILE"
