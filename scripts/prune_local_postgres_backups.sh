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
KEEP_COUNT="${LOCAL_POSTGRES_BACKUP_KEEP_COUNT:-5}"

if [[ ! -d "$BACKUP_DIR" ]]; then
  echo "Backup directory does not exist: $BACKUP_DIR" >&2
  exit 1
fi

if ! [[ "$KEEP_COUNT" =~ ^[0-9]+$ ]]; then
  echo "LOCAL_POSTGRES_BACKUP_KEEP_COUNT must be an integer" >&2
  exit 1
fi

mapfile -t BACKUP_FILES < <(ls -1t "$BACKUP_DIR"/chatbot_*.dump 2>/dev/null || true)

FILE_COUNT="${#BACKUP_FILES[@]}"

if (( FILE_COUNT <= KEEP_COUNT )); then
  echo "No pruning needed. Found $FILE_COUNT backup file(s), keep count is $KEEP_COUNT."
  exit 0
fi

echo "Pruning local PostgreSQL backups in $BACKUP_DIR"
echo "Keeping newest $KEEP_COUNT file(s), deleting $((FILE_COUNT - KEEP_COUNT)) older file(s)."

for FILE in "${BACKUP_FILES[@]:KEEP_COUNT}"; do
  echo "Deleting old backup: $FILE"
  rm -f -- "$FILE"
done

echo "Local backup pruning completed."
