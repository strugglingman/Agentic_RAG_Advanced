@echo off
setlocal

REM Modern backend test runner with explicit lanes.
REM Usage:
REM   run_tests.bat               -> fast lane (default)
REM   run_tests.bat fast          -> fast lane
REM   run_tests.bat integration   -> integration lane
REM   run_tests.bat external      -> external/API-dependent lane
REM   run_tests.bat manual        -> manual scripts/tests (requires env + optional local services)
REM   run_tests.bat all           -> all automated lanes except manual

set MODE=%1
if "%MODE%"=="" set MODE=fast

pushd "%~dp0"

echo ================================================================
echo Backend test mode: %MODE%
echo ================================================================

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found in venv\ or .venv\
    exit /b 1
)

if /I "%MODE%"=="fast" (
    python -m pytest tests
    set EXIT_CODE=%ERRORLEVEL%
    popd
    exit /b %EXIT_CODE%
)

if /I "%MODE%"=="integration" (
    python -m pytest -o addopts="-q -ra --strict-markers -m \"integration and not manual and not external\"" tests
    set EXIT_CODE=%ERRORLEVEL%
    popd
    exit /b %EXIT_CODE%
)

if /I "%MODE%"=="external" (
    set RUN_MANUAL_TESTS=1
    python -m pytest -o addopts="-q -ra --strict-markers -m external" tests
    set EXIT_CODE=%ERRORLEVEL%
    popd
    exit /b %EXIT_CODE%
)

if /I "%MODE%"=="manual" (
    set RUN_MANUAL_TESTS=1
    python -m pytest -o addopts="-q -ra --strict-markers -m manual" tests
    set EXIT_CODE=%ERRORLEVEL%
    popd
    exit /b %EXIT_CODE%
)

if /I "%MODE%"=="all" (
    python -m pytest -o addopts="-q -ra --strict-markers" -m "not manual" tests
    set EXIT_CODE=%ERRORLEVEL%
    popd
    exit /b %EXIT_CODE%
)

echo ERROR: Unknown mode "%MODE%"
echo Valid modes: fast ^| integration ^| external ^| manual ^| all
popd
exit /b 2
