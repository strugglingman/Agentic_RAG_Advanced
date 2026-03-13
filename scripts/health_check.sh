#!/usr/bin/env bash
set -euo pipefail

echo "== pwd =="
pwd

echo
echo "== backend files =="
test -f backend/src/fastapi_app.py && echo "backend entry: ok" || echo "backend entry: missing"

echo
echo "== frontend files =="
test -f frontend/app/api/chat/route.ts && echo "frontend chat route: ok" || echo "frontend chat route: missing"

echo
echo "== prisma schema =="
test -f backend/prisma/schema.prisma && echo "prisma schema: ok" || echo "prisma schema: missing"

echo
echo "== qdrant retrieval files =="
test -f backend/src/services/vector_db_qdrant.py && echo "vector_db_qdrant.py: ok" || echo "vector_db_qdrant.py: missing"
test -f backend/src/services/retrieval_qdrant.py && echo "retrieval_qdrant.py: ok" || echo "retrieval_qdrant.py: missing"

echo
echo "== langgraph files =="
test -f backend/src/services/query_supervisor.py && echo "query_supervisor.py: ok" || echo "query_supervisor.py: missing"

echo
echo "== done =="