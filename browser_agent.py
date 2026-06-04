"""
browser_agent.py — Browser automation with planning, execution, and verification
Provides autonomous browser interactions using Playwright.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Generator

try:
    from playwright.sync_api import sync_playwright, Page, Browser
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False


@dataclass
class BrowserAction:
    """Represents a single browser action to be performed."""
    action: str
    target: Optional[str] = None
    value: Optional[str] = None


class BrowserAgent:
    """Autonomous browser agent with planning and execution."""
    
    def __init__(self):
        self.available = _PLAYWRIGHT_AVAILABLE
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.history: list[str] = []
        self.screenshots: list[bytes] = []
        
        if not self.available:
            print("[BrowserAgent] Playwright not available - browser automation disabled")
    
    def start(self) -> bool:
        """Initialize the browser."""
        if not self.available:
            return False
        if self.browser is not None:
            return True
        
        try:
            playwright = sync_playwright().start()
            self.browser = playwright.chromium.launch(headless=False)
            self.page = self.browser.new_page()
            return True
        except Exception as e:
            print(f"[BrowserAgent] Failed to start browser: {e}")
            return False
    
    def stop(self):
        """Close the browser."""
        if self.browser:
            self.browser.close()
            self.browser = None
            self.page = None
    
    def execute_action(self, action: BrowserAction) -> str:
        """Execute a single browser action."""
        if not self.start():
            return "Browser not available"
        
        try:
            if action.action == "goto":
                return self._action_goto(action.target)
            elif action.action == "search":
                return self._action_search(action.target)
            elif action.action == "click":
                return self._action_click(action.target)
            elif action.action == "type":
                return self._action_type(action.target, action.value)
            elif action.action == "extract":
                return self._action_extract()
            elif action.action == "screenshot":
                return self._action_screenshot()
            else:
                return f"Unknown action: {action.action}"
        except Exception as e:
            return f"Browser error: {e}"
    
    def _action_goto(self, url: str) -> str:
        if not url:
            return "No URL provided"
        try:
            if "://" not in url:
                url = "https://" + url
            self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            self.history.append(url)
            return f"Navigated to {url}"
        except Exception as e:
            return f"Navigation failed: {e}"
    
    def _action_search(self, query: str) -> str:
        if not query:
            return "No search query provided"
        try:
            url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}"
            self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            self.history.append(url)
            return f"Searched for: {query}"
        except Exception as e:
            return f"Search failed: {e}"
    
    def _action_click(self, selector: str) -> str:
        if not selector:
            return "No selector provided"
        try:
            self.page.click(selector, timeout=5000)
            return f"Clicked element: {selector}"
        except Exception as e:
            return f"Click failed: {e}"
    
    def _action_type(self, selector: str, text: str) -> str:
        if not selector or not text:
            return "Selector and text required"
        try:
            self.page.fill(selector, text, timeout=5000)
            return f"Typed text in {selector}"
        except Exception as e:
            return f"Type failed: {e}"
    
    def _action_extract(self) -> str:
        try:
            text = self.page.inner_text("body")
            if len(text) > 8000:
                text = text[:8000] + "\n… [truncated]"
            return text
        except Exception as e:
            return f"Extraction failed: {e}"
    
    def _action_screenshot(self) -> str:
        try:
            screenshot_bytes = self.page.screenshot()
            self.screenshots.append(screenshot_bytes)
            return f"Screenshot taken ({len(screenshot_bytes)} bytes)"
        except Exception as e:
            return f"Screenshot failed: {e}"
    
    def plan_from_user_intent(self, intent: str) -> list[BrowserAction]:
        """Convert user intent into browser actions."""
        actions = []
        intent_lower = intent.lower()
        
        if "search" in intent_lower:
            query = intent.replace("search for", "").replace("search", "").strip()
            actions.append(BrowserAction("search", target=query))
        elif "open" in intent_lower or "go to" in intent_lower:
            url = intent.replace("open", "").replace("go to", "").strip()
            actions.append(BrowserAction("goto", target=url))
        elif "extract" in intent_lower or "read" in intent_lower:
            actions.append(BrowserAction("extract"))
        elif "screenshot" in intent_lower:
            actions.append(BrowserAction("screenshot"))
        
        return actions
    
    def execute_plan(self, actions: list[BrowserAction]) -> Generator[str, None, None]:
        """Execute a plan and yield results for each action."""
        for action in actions:
            result = self.execute_action(action)
            yield f"[{action.action}] {result}"
