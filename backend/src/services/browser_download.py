"""
Browser-based download service using Browser-Use for complex downloads.

This module handles downloads that require:
- Login/authentication
- JavaScript rendering
- Button clicks to trigger downloads
- Navigation through multi-step processes

Uses Browser-Use (AI-powered Playwright wrapper) for intelligent web automation.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

from src.config.settings import Config
from src.observability.metrics import increment_error, MetricsErrorType

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading browser dependencies when not needed
_browser_use_available = None


def is_browser_use_available() -> bool:
    """Check if browser-use is installed and available."""
    global _browser_use_available
    if _browser_use_available is None:
        try:
            from browser_use import Agent
            from langchain_openai import ChatOpenAI

            _browser_use_available = True
        except ImportError:
            _browser_use_available = False
            logger.warning(
                "browser-use not installed. Run: pip install browser-use playwright && playwright install"
            )
    return _browser_use_available


async def browser_download(
    task: str,
    download_dir: str,
    url: Optional[str] = None,
    credentials: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    Use Browser-Use AI agent to navigate and download a file.

    Args:
        task: Natural language description of what to download
              e.g., "Download the December 2025 internet invoice"
        download_dir: Directory to save downloaded files
        url: Optional starting URL (if known)
        credentials: Optional dict with 'username' and 'password' keys
        timeout: Optional timeout in seconds (default: Config.BROWSER_TIMEOUT)

    Returns:
        Tuple of (success: bool, message: str, file_path: Optional[str])
    """
    if not Config.BROWSER_USE_ENABLED:
        return (
            False,
            "Browser automation is disabled. Set BROWSER_USE_ENABLED=true",
            None,
        )

    if not is_browser_use_available():
        return False, "browser-use library not installed", None

    # Import here to avoid loading when not needed
    from browser_use import Agent, Browser, BrowserConfig
    from langchain_openai import ChatOpenAI

    timeout = timeout or Config.BROWSER_TIMEOUT

    # Ensure download directory exists
    os.makedirs(download_dir, exist_ok=True)

    # Build the task prompt
    task_prompt = _build_task_prompt(task, url, credentials)

    try:
        # Initialize LLM for browser agent
        llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            api_key=Config.OPENAI_KEY,
            temperature=0,
        )

        # Configure browser
        browser_config = BrowserConfig(
            headless=Config.BROWSER_HEADLESS,
            disable_security=False,
        )

        browser = Browser(config=browser_config)

        # Create agent with download directory context
        # Enable conversation logging if path is configured
        log_path = Config.BROWSER_LOG_PATH if Config.BROWSER_LOG_PATH else None
        agent = Agent(
            task=task_prompt,
            llm=llm,
            browser=browser,
            save_conversation_path=log_path,
        )

        # Run the agent with timeout
        await asyncio.wait_for(
            agent.run(max_steps=Config.BROWSER_MAX_STEPS), timeout=timeout
        )

        # Check for downloaded files
        downloaded_file = _find_newest_file(download_dir)

        if downloaded_file:
            logger.info(
                f"[BROWSER_DOWNLOAD] Successfully downloaded: {downloaded_file}"
            )
            return True, f"Downloaded successfully", downloaded_file
        else:
            # Agent completed but no file found
            logger.warning(
                "[BROWSER_DOWNLOAD] Agent completed but no file was downloaded"
            )
            return (
                False,
                "Agent completed but no file was downloaded. The site may require different navigation.",
                None,
            )

    except asyncio.TimeoutError:
        logger.error(f"[BROWSER_DOWNLOAD] Timeout after {timeout}s")
        increment_error(MetricsErrorType.TIMEOUT)
        return False, f"Browser automation timed out after {timeout} seconds", None
    except Exception as e:
        logger.error(f"[BROWSER_DOWNLOAD] Error: {e}")
        return False, f"Browser automation failed: {str(e)}", None
    finally:
        # Cleanup browser
        try:
            if "browser" in locals():
                await browser.close()
        except Exception:
            pass


def _build_task_prompt(
    task: str,
    url: Optional[str] = None,
    credentials: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build a detailed task prompt for the browser agent.

    Args:
        task: User's download request
        url: Optional starting URL
        credentials: Optional login credentials

    Returns:
        Formatted task prompt for Browser-Use agent
    """
    prompt_parts = []

    # Main task
    prompt_parts.append(f"Task: {task}")

    # Starting URL if provided
    if url:
        prompt_parts.append(f"\nStart at URL: {url}")

    # Credentials if provided
    if credentials:
        username = credentials.get("username") or Config.BROWSER_TEST_USERNAME
        password = credentials.get("password") or Config.BROWSER_TEST_PASSWORD
        if username and password:
            prompt_parts.append(f"\nLogin credentials if needed:")
            prompt_parts.append(f"- Username: {username}")
            prompt_parts.append(f"- Password: {password}")

    # Instructions for download behavior
    prompt_parts.append(
        """

Instructions:
1. Navigate to find the requested file/document
2. If login is required, use the provided credentials
3. Look for download buttons, links, or PDF/document icons
4. Click to download the file
5. Wait for the download to complete
6. If you encounter errors or can't find the file, describe what you see

Important:
- Do NOT enter any information other than the provided credentials
- If asked for additional verification (2FA, CAPTCHA), stop and report
- Prefer direct download links over preview pages
"""
    )

    return "\n".join(prompt_parts)


def _find_newest_file(directory: str, max_age_seconds: int = 120) -> Optional[str]:
    """
    Find the most recently created/modified file in directory.

    Args:
        directory: Directory to search
        max_age_seconds: Only consider files modified within this time

    Returns:
        Path to newest file, or None if no recent files found
    """
    import time

    dir_path = Path(directory)
    if not dir_path.exists():
        return None

    newest_file = None
    newest_time = 0
    current_time = time.time()

    for file_path in dir_path.iterdir():
        if file_path.is_file():
            mtime = file_path.stat().st_mtime
            # Only consider files modified recently (during this download attempt)
            if current_time - mtime < max_age_seconds and mtime > newest_time:
                newest_time = mtime
                newest_file = str(file_path)

    return newest_file


async def browser_download_with_steps(
    url: str,
    steps: list[str],
    download_dir: str,
    credentials: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    Execute a predefined sequence of browser steps for known sites.

    This is more reliable than AI navigation for sites with known structure.

    Args:
        url: Starting URL
        steps: List of step descriptions for the agent
               e.g., ["Click 'Bills & Payments'", "Select December 2025", "Click Download"]
        download_dir: Directory to save downloaded files
        credentials: Optional login credentials
        timeout: Optional timeout in seconds

    Returns:
        Tuple of (success: bool, message: str, file_path: Optional[str])
    """
    # Combine steps into a single task
    steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
    task = f"""Navigate to {url} and follow these steps:
{steps_text}

Download any file that results from these steps."""

    return await browser_download(
        task=task,
        download_dir=download_dir,
        url=url,
        credentials=credentials,
        timeout=timeout,
    )
