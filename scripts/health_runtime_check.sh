
#!/usr/bin/env bash

set -euo pipefail



echo "== pwd =="

pwd



echo

echo "== docker compose =="

docker compose ps || true



echo

echo "== frontend =="

curl -I --max-time 5 http://localhost:3000 || true



echo

echo "== backend docs =="

curl -I --max-time 5 http://localhost:5001/docs || true



echo

echo "== qdrant health =="

curl --max-time 5 http://localhost:6333/healthz || true



echo

echo "== listening ports =="

ss -ltn 2>/dev/null | grep -E ":3000|:5001|:5433|:6333|:6379" || true



echo

echo "== done =="

