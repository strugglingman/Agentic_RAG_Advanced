# Step-by-Step Testing Instructions

**Goal**: Complete Week 2 & Week 3 testing to verify all features work correctly

**Time Required**: 15-20 minutes

---

## 🚀 **Quick Start (Easiest Method)**

### **Option A: Automated Test Script (Windows)**

1. **Open PowerShell or Command Prompt**
   ```powershell
   cd d:\chatbot\backend
   ```

2. **Run the test script**
   ```powershell
   .\run_tests.bat
   ```

3. **Follow the prompts**
   - Script will activate virtual environment automatically
   - Runs all 3 tests sequentially
   - Press any key to continue between tests
   - All tests should pass with "[OK]" messages

4. **Expected Output**:
   - Test 1: ClarificationHelper ✅
   - Test 2: QueryRefiner ✅
   - Test 3: WebSearchService ✅

---

## 📋 **Option B: Manual Testing (Step-by-Step)**

If the script doesn't work, follow these manual steps:

### **Prerequisites**

1. **Open PowerShell** and navigate to backend:
   ```powershell
   cd d:\chatbot\backend
   ```

2. **Activate virtual environment**:

   **If using `venv` folder**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   **If using `.venv` folder**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   **If activation is blocked by execution policy**:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   Then try activating again.

3. **Verify activation** (you should see `(venv)` or `(.venv)` in prompt):
   ```powershell
   python --version
   # Should show: Python 3.10+ or 3.11+
   ```

---

### **Test 1: ClarificationHelper** (3-5 minutes)

1. **Run the test**:
   ```powershell
   python -m src.services.clarification_helper
   ```

2. **Expected Output**:
   ```
   ======================================================================
   CLARIFICATION HELPER TEST
   ======================================================================
   [OK] ClarificationHelper created

   ----------------------------------------------------------------------
   Test 1: No Results Message
   ----------------------------------------------------------------------
   I couldn't find any relevant information for: "What is quantum physics?".

   Suggestions to improve your query:
   - The information might not be in the uploaded documents.
   - Try using different keywords.
   - Be more specific about what you're looking for.

   Note: I have searched in HR documents.

   ----------------------------------------------------------------------
   Test 2: Ambiguous Query Message
   ----------------------------------------------------------------------
   Your query "How do I apply?" returned low-quality results.

   Issues found:
   - Low average relevance score: 0.20
   - Poor keyword match: 0.10

   Missing keywords: apply, application, process, submit

   Suggestions to improve your query:
   - Be more specific about what you're looking for.
   - Use different keywords.
   - Provide more context if possible.

   Note: I have searched in company policies.

   ----------------------------------------------------------------------
   Test 3: Max Attempts Message
   ----------------------------------------------------------------------
   After multiple search attempts, I couldn't find highly relevant information for: "Tell me about xyz".

   The best results I found may not fully answer your question.

   Suggestions to improve your query:
   - Try rephrasing your question.
   - Break it into smaller, more specific questions.
   - Check if the information exists in the uploaded documents.

   Keywords not found: apply, application, process, submit

   I have searched in: company documents.

   ----------------------------------------------------------------------
   Test 4: generate_clarification Routing
   ----------------------------------------------------------------------
     max_attempts_reached=True: _max_attempts_message
     empty issues: _no_results_message
     issues present: _ambiguous_query_message
     [OK] Routing logic verified

   ======================================================================
   ALL TESTS COMPLETE!
   ======================================================================
   ```

3. **Result**: If you see "ALL TESTS COMPLETE!" → ✅ **PASS**

---

### **Test 2: QueryRefiner** (3-5 minutes)

1. **Run the test**:
   ```powershell
   python -m src.services.query_refiner
   ```

2. **Expected Output**:
   ```
   ======================================================================
   QUERY REFINER TEST
   ======================================================================

   Test 1: QueryRefiner Initialization
   ----------------------------------------------------------------------
     OpenAI client: <OpenAI object at ...>
     Model: gpt-4o-mini
     Temperature: 0.1
   [OK] QueryRefiner initialized

   Test 2: Refinement (with LLM)
   ----------------------------------------------------------------------
     Original query: 'employee benefits policy'
     Calling OpenAI for query refinement...
     Refined query: 'employee benefits policy coverage health insurance dental vision'
     [OK] Query expanded with more keywords

   Test 3: Simple Refinement (fallback, no LLM)
   ----------------------------------------------------------------------
     Original query: 'employee benefits policy'
     Refined query: 'employee benefits policy employee coverage health insurance benefits'
     [OK] Fallback refinement working

   Test 4: should_refine() Logic
   ----------------------------------------------------------------------
     Test 1 (should_refine=True, confidence=0.6): True
     Test 2 (should_refine=False, confidence=0.85): False
     Test 3 (should_refine=False, confidence=0.4): False
   [OK] should_refine() logic working

   Test 5: track_refinement() Helper
   ----------------------------------------------------------------------
   [REFINEMENT] Attempt 1: 'original query' -> 'refined query'
   [OK] Refinement tracking working

   ======================================================================
   ALL TESTS COMPLETE!
   ======================================================================
   ```

3. **Note**: Test 2 makes an actual OpenAI API call, so it:
   - Takes 2-3 seconds
   - Uses ~$0.001 in API credits
   - Requires valid `OPENAI_API_KEY` in `.env`

4. **Result**: If you see "ALL TESTS COMPLETE!" → ✅ **PASS**

---

### **Test 3: WebSearchService** (3-5 minutes)

1. **Run the test**:
   ```powershell
   python -m src.services.web_search
   ```

2. **Expected Output**:
   ```
   ======================================================================
   WEB SEARCH SERVICE TEST
   ======================================================================

   Test 1: Service Initialization
     Provider: duckduckgo
     Max results: 5
   [OK] Service initialized

   Test 2: Query Validation
     Valid query: 'test query'
     Short query rejected: Query too short (minimum 3 characters)
     Long query truncated to 500 chars
   [OK] Query validation working

   Test 3: DuckDuckGo Search
     Query: 'Python programming language'
     Results found: 3
       - Python (programming language) - Wikipedia... (en.wikipedia.org)
       - Welcome to Python.org... (python.org)
       - Learn Python - Full Course for Beginners... (youtube.com)
   [OK] DuckDuckGo search working

   Test 4: Format for Agent
   Web search results for: "Python programming language"

   1. Python (programming language) - Wikipedia
      Source: en.wikipedia.org
      URL: https://en.wikipedia.org/wiki/Python_(programming_language)
      Python is a high-level, general-purpose programming language...

   2. Welcome to Python.org
      Source: python.org
      URL: https://www.python.org/
      The official home of the Python Programming Language...

   3. Learn Python - Full Course for Beginners
      Source: youtube.com
      URL: https://www.youtube.com/watch?v=...
      Learn Python programming in this comprehensive course...

   [Note: These results are from external web sources]
   [OK] Formatting working

   Test 5: Real-world Query
   Web search results for: "current US inflation rate 2024"

   1. Consumer Price Index Summary - Bureau of Labor Statistics
      Source: bls.gov
      URL: https://www.bls.gov/news.release/cpi.nr0.htm
      The Consumer Price Index for All Urban Consumers (CPI-U)...

   [... more results ...]

   [Note: These results are from external web sources]
   [OK] Real-world query working

   ======================================================================
   ALL TESTS COMPLETE!
   ======================================================================
   ```

3. **Note**: This test:
   - Makes real web searches (DuckDuckGo)
   - Requires internet connection
   - Takes 5-10 seconds

4. **Result**: If you see "ALL TESTS COMPLETE!" → ✅ **PASS**

---

## ✅ **Success Criteria**

All 3 tests should show:
- ✅ `[OK]` messages for each sub-test
- ✅ `ALL TESTS COMPLETE!` at the end
- ✅ No error messages or exceptions

---

## 🐛 **Troubleshooting**

### **Issue 1: "ModuleNotFoundError: No module named 'openai'"**

**Solution**: Virtual environment not activated or dependencies not installed

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

### **Issue 2: "PowerShell execution policy error"**

**Solution**: Allow script execution

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating venv again.

---

### **Issue 3: OpenAI API key error in QueryRefiner test**

**Solution**: Check `.env` file has valid API key

```bash
# In backend/.env, verify this line exists and has valid key:
OPENAI_API_KEY=sk-proj-...your-key-here...
```

If key is missing or invalid:
1. Get API key from https://platform.openai.com/api-keys
2. Add to `.env` file
3. Rerun test

---

### **Issue 4: DuckDuckGo returns no results**

**Possible causes**:
- No internet connection
- DuckDuckGo rate limiting (try again in 1 minute)
- Query too specific

**Solution**:
- Check internet connection
- Wait 1 minute and retry
- Test should still show `[OK]` even with 0 results (validates logic)

---

### **Issue 5: "ImportError" or module not found**

**Solution**: Wrong directory or virtual environment issue

```powershell
# Make sure you're in backend directory
pwd
# Should show: d:\chatbot\backend

# Make sure venv is activated
# Prompt should show: (venv) PS d:\chatbot\backend>

# If not, activate:
.\venv\Scripts\Activate.ps1
```

---

## 📊 **After All Tests Pass**

Once all 3 tests complete successfully:

### **Update Your Status**

**Week 2**: ✅ **COMPLETE**
- Query Refinement: ✅ Working
- Clarification: ✅ Working

**Week 3**: ✅ **COMPLETE**
- Web Search Service: ✅ Working
- EXTERNAL Detection: ✅ Implemented
- Agent Integration: ✅ Ready

### **What This Means**

You've successfully implemented and verified:
1. ✅ Self-reflection evaluation (Week 1)
2. ✅ Query refinement with progressive fallback (Week 2)
3. ✅ Clarification message generation (Week 2)
4. ✅ Web search fallback (Week 3)
5. ✅ EXTERNAL query detection (Week 3)

**Your RAG chatbot now has:**
- Self-evaluation
- Automatic query improvement
- Helpful clarification messages
- External knowledge integration

---

## 🚀 **Next Steps (Optional)**

### **Phase 2: Integration Tests** (30-60 minutes)

If you want even more confidence, create integration tests:

1. **Create test file**: `backend/tests/test_week2_integration.py`
2. **Copy code from**: [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) (Phase 2)
3. **Run**: `python tests/test_week2_integration.py`

### **Phase 3: End-to-End Testing** (30 minutes)

Test through the actual chat UI:

1. **Start backend**: `cd backend && flask run`
2. **Start frontend**: `cd frontend && npm run dev`
3. **Test scenarios**:
   - High confidence query → direct answer
   - Medium confidence → refinement
   - Low confidence → clarification
   - External query → EXTERNAL suggestion

---

## 📝 **Testing Results Template**

Use this to document your results:

```
# Week 2 & 3 Testing Results

**Date**: ___________
**Tester**: ___________

## Standalone Tests

| Test | Status | Notes |
|------|--------|-------|
| ClarificationHelper | ✅ Pass / ❌ Fail | |
| QueryRefiner | ✅ Pass / ❌ Fail | |
| WebSearchService | ✅ Pass / ❌ Fail | |

## Overall Status

- Week 2 (Refinement + Clarification): ✅ Complete / ❌ Incomplete
- Week 3 (Web Search): ✅ Complete / ❌ Incomplete

## Issues Found

(List any errors or unexpected behavior)

## Notes

(Any observations or improvements needed)
```

---

## 🎉 **Congratulations!**

Once all tests pass, you've completed Week 2 & Week 3 implementation!

Your project now has:
- ✅ Advanced self-reflection
- ✅ Automatic query refinement
- ✅ Progressive fallback chain
- ✅ External knowledge integration

**This is production-ready, cutting-edge RAG!** 🚀
