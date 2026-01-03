"""
Bot Adapters Module
===================

This module provides adapters for integrating external chat platforms
(Slack, Teams, etc.) with the existing backend API.

DESIGN PRINCIPLE:
-----------------
The adapters are "translation layers" - they DO NOT modify any backend logic.
They simply:
1. Receive events from external platforms (Slack, Teams)
2. Translate platform-specific data to your existing API format
3. Call your existing endpoints (same as frontend does)
4. Format responses back to platform-specific format

BACKEND STAYS 100% UNCHANGED:
- /chat/agent endpoint - no changes
- /files/upload endpoint - no changes
- /ingest endpoint - no changes
- All request/response formats - no changes

Available Adapters:
- slack: Slack bot adapter (see adapters/slack/)
- teams: Microsoft Teams adapter (see adapters/teams/) [future]
"""
