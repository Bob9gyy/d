"""
core_verifier.py — Tool execution verification and correction layer

Implements verification of tool outputs, success detection, retry logic,
and failure recovery for autonomous agent operations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class VerificationResult:
    """Result of verifying a tool execution."""
    success: bool
    confidence: float
    error_message: Optional[str] = None
    recovery_action: Optional[str] = None
    should_retry: bool = False


class CoreVerifier:
    """Verifies tool execution results and determines if they meet success criteria."""
    
    def __init__(self):
        self.retry_count = 0
        self.max_retries = 3
    
    def verify_tool_result(self, tool_name: str, args: dict, result: str) -> VerificationResult:
        """Verify if a tool execution was successful."""
        error_keywords = {"error", "failed", "denied", "blocked", "timeout", "exception"}
        result_lower = result.lower()
        
        has_error = any(kw in result_lower for kw in error_keywords)
        is_empty = not result or len(result.strip()) == 0
        
        if has_error or is_empty:
            recovery = self._suggest_recovery(tool_name, args, result)
            should_retry = self.retry_count < self.max_retries
            
            return VerificationResult(
                success=False,
                confidence=0.0,
                error_message=result if has_error else "Empty result",
                recovery_action=recovery,
                should_retry=should_retry
            )
        
        if tool_name == "read_file":
            return self._verify_read_file(result)
        elif tool_name == "list_dir":
            return self._verify_list_dir(result)
        elif tool_name == "search_files":
            return self._verify_search_files(result)
        elif tool_name == "system_info":
            return self._verify_system_info(result)
        elif tool_name == "open_app":
            return self._verify_open_app(result)
        elif tool_name == "open_url":
            return self._verify_open_url(result)
        elif tool_name == "search_web":
            return self._verify_search_web(result)
        else:
            return VerificationResult(success=True, confidence=0.8)
    
    def _verify_read_file(self, result: str) -> VerificationResult:
        if len(result) > 20:
            return VerificationResult(success=True, confidence=0.95)
        return VerificationResult(success=False, confidence=0.2, 
                                error_message="File appears empty or unreadable")
    
    def _verify_list_dir(self, result: str) -> VerificationResult:
        if "[DIR]" in result or "[FILE]" in result:
            return VerificationResult(success=True, confidence=0.9)
        return VerificationResult(success=False, confidence=0.3,
                                error_message="Directory listing format invalid")
    
    def _verify_search_files(self, result: str) -> VerificationResult:
        if "No files found" in result:
            return VerificationResult(success=True, confidence=0.85)
        if "match]" in result or "found" in result.lower():
            return VerificationResult(success=True, confidence=0.95)
        return VerificationResult(success=False, confidence=0.2)
    
    def _verify_system_info(self, result: str) -> VerificationResult:
        if "CPU" in result or "RAM" in result or "DISK" in result:
            return VerificationResult(success=True, confidence=0.9)
        return VerificationResult(success=False, confidence=0.1)
    
    def _verify_open_app(self, result: str) -> VerificationResult:
        if "Launched" in result:
            return VerificationResult(success=True, confidence=0.85)
        return VerificationResult(success=False, confidence=0.2, should_retry=True)
    
    def _verify_open_url(self, result: str) -> VerificationResult:
        if "Opened" in result:
            return VerificationResult(success=True, confidence=0.9)
        return VerificationResult(success=False, confidence=0.2)
    
    def _verify_search_web(self, result: str) -> VerificationResult:
        if "Searched" in result:
            return VerificationResult(success=True, confidence=0.9)
        return VerificationResult(success=False, confidence=0.2)
    
    def _suggest_recovery(self, tool_name: str, args: dict, result: str) -> Optional[str]:
        if "Permission denied" in result:
            return "Check file permissions for path"
        elif "not found" in result.lower():
            return "Verify path or tool name exists"
        elif "timeout" in result.lower():
            return "Retry with increased timeout"
        elif "blocked" in result.lower():
            return "Action blocked by security policy"
        return None


class TaskPersistence:
    """Manages persistence of long-running tasks so they can be resumed after crashes."""
    
    def __init__(self, storage_dir: str = "./tasks"):
        self.storage_dir = storage_dir
        import os
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_task_state(self, task_id: str, state: dict) -> bool:
        """Save current task state to disk."""
        try:
            import json
            from pathlib import Path
            
            path = Path(self.storage_dir) / f"{task_id}.json"
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving task state: {e}")
            return False
    
    def load_task_state(self, task_id: str) -> Optional[dict]:
        """Load task state from disk."""
        try:
            import json
            from pathlib import Path
            
            path = Path(self.storage_dir) / f"{task_id}.json"
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading task state: {e}")
        return None
    
    def list_unfinished_tasks(self) -> list[dict]:
        """List all unfinished tasks."""
        try:
            import json
            from pathlib import Path
            
            tasks = []
            for task_file in Path(self.storage_dir).glob("*.json"):
                with open(task_file, 'r') as f:
                    state = json.load(f)
                    if state.get("status") in ("pending", "running"):
                        tasks.append(state)
            return tasks
        except Exception:
            return []
