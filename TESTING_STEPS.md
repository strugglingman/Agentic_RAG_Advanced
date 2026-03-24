# Modern Testing Workflow

This project now uses lane-based testing for predictable feedback:

- `fast`: deterministic default checks, no external services.
- `integration`: requires local infra (Postgres/Redis/Qdrant/Prisma engine).
- `external`: requires internet or third-party APIs.
- `manual`: exploratory scripts (never CI-gating by default).

## Backend (pytest)

From `backend/`:

```powershell
# Fast lane (default)
python -m pytest tests

# Integration lane
python -m pytest -o addopts="-q -ra --strict-markers -m integration" tests

# External/API-dependent lane
python -m pytest -o addopts="-q -ra --strict-markers -m external" tests

# All automated lanes except manual
python -m pytest -o addopts="-q -ra --strict-markers -m not manual" tests
```

Integration prerequisite (Prisma-backed endpoints):

```powershell
prisma py fetch
```

If Prisma engine is missing locally, relevant integration tests now skip with an explicit message.

Windows helper:

```powershell
.\run_tests.bat fast
.\run_tests.bat integration
.\run_tests.bat external
.\run_tests.bat manual
.\run_tests.bat all
```

### Backend markers

Defined in `backend/pytest.ini`:

- `unit`
- `component`
- `integration`
- `external`
- `manual`
- `e2e`

Default run excludes: `integration`, `external`, `manual`, `e2e`.

Manual/external script-style tests are gated by:

```powershell
$env:RUN_MANUAL_TESTS="1"
python -m pytest -o addopts="-q -ra --strict-markers -m manual" tests
```

Additional manual knobs:

```powershell
# Retrieval evaluator smoke mode: fast | balanced | thorough
$env:MANUAL_REFLECTION_MODE="fast"

# API smoke target for manual conversation checks
$env:MANUAL_BACKEND_BASE_URL="http://localhost:5001"
```

Manual lane prerequisites:

- `test_conversations.py`: running backend API at `MANUAL_BACKEND_BASE_URL`.
- `test_agent_service.py`: reachable OpenAI endpoint + `OPENAI_API_KEY`.
- `test_langsmith_integration.py`: `OPENAI_API_KEY` + `LANGCHAIN_API_KEY` (and `RUN_LANGSMITH_UPLOAD=1` for upload check).

### Hardening guard

All backend tests must declare at least one lane marker (`unit`, `component`, `integration`, `external`, `manual`, or `e2e`).
Collection now fails if a test is unclassified.

## Frontend (Vitest + Playwright)

From `frontend/`:

```powershell
# Unit tests
npm run test

# Unit coverage
npm run test:coverage

# E2E test list (discovery only)
npm run test:e2e -- --list

# Run E2E
npm run test:e2e
```

## Recommended CI order

1. Backend fast lane.
2. Frontend unit lane.
3. Backend integration lane (with infra).
4. Frontend e2e lane (against deployed preview or compose stack).
