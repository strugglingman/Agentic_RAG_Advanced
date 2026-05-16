#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Running PostgreSQL backup maintenance..."
"$ROOT_DIR/scripts/backup_and_upload_postgres.sh"
"$ROOT_DIR/scripts/prune_local_postgres_backups.sh"
echo "PostgreSQL backup maintenance completed."
