@echo off
REM Test runner for Week 2 & 3 standalone tests
REM Run this from backend directory

echo ======================================================================
echo WEEK 2 & 3 STANDALONE TESTS
echo ======================================================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    echo Please create one with: python -m venv venv
    pause
    exit /b 1
)

echo Virtual environment activated
echo.

REM Test 1: ClarificationHelper
echo ======================================================================
echo TEST 1: CLARIFICATION HELPER
echo ======================================================================
python -m src.services.clarification_helper
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] ClarificationHelper test failed
    pause
    exit /b 1
)
echo.
echo [PASSED] ClarificationHelper test passed
echo.
pause

REM Test 2: QueryRefiner
echo ======================================================================
echo TEST 2: QUERY REFINER
echo ======================================================================
python -m src.services.query_refiner
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] QueryRefiner test failed
    pause
    exit /b 1
)
echo.
echo [PASSED] QueryRefiner test passed
echo.
pause

REM Test 3: WebSearchService
echo ======================================================================
echo TEST 3: WEB SEARCH SERVICE
echo ======================================================================
python -m src.services.web_search
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] WebSearchService test failed
    pause
    exit /b 1
)
echo.
echo [PASSED] WebSearchService test passed
echo.

echo ======================================================================
echo ALL STANDALONE TESTS COMPLETE!
echo ======================================================================
echo.
echo Results:
echo   [PASS] ClarificationHelper
echo   [PASS] QueryRefiner
echo   [PASS] WebSearchService
echo.
echo Week 2 ^& 3 standalone tests: COMPLETE
echo.
pause
