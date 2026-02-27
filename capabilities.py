"""Capability implementations available to agents."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from agentlib.llm import LLMProvider, LLMResponse, make_provider

_provider: LLMProvider | None = None


def _extract_text(resp: LLMResponse, fallback: str) -> str:
    parts = [b.text for b in resp.content if b.type == "text"]
    return "\n".join(parts) if parts else fallback


def set_provider(p: LLMProvider):
    global _provider
    _provider = p


def _get_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        _provider = make_provider()
    return _provider


# ── API-only capabilities (no sandbox needed) ────────────────────────


async def web_search(query="", **kw):
    provider = _get_provider()
    if not provider.has_api_key():
        return f"No API key — cannot search for '{query}'."
    if provider.name != "anthropic":
        return await _web_search_generic(provider, query)
    # Anthropic-native web search tool
    try:
        resp = await asyncio.to_thread(
            provider.create_sync,
            messages=[{"role": "user", "content": f"Search the web for: {query}\n\nReturn a concise summary of what you find with URLs."}],
            max_tokens=1024,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        )
        return _extract_text(resp, f"No results found for '{query}'.")
    except Exception as e:
        return f"Search failed: {e}"


async def _web_search_generic(provider: LLMProvider, query: str) -> str:
    """Fallback web search for non-Anthropic providers (LLM-only, no live web)."""
    try:
        resp = await asyncio.to_thread(
            provider.create_sync,
            messages=[{"role": "user", "content": (
                f"The user wants to search the web for: {query}\n\n"
                "You do not have live web access. Provide the best answer you can from your training data. "
                "Clearly state that this is from training data, not a live search."
            )}],
            max_tokens=1024,
        )
        return _extract_text(resp, f"No results for '{query}'.")
    except Exception as e:
        return f"Search failed: {e}"


async def summarize_text(text="", **kw):
    provider = _get_provider()
    if not provider.has_api_key():
        return f"Summary: {text[:100]}..."
    resp = await asyncio.to_thread(
        provider.create_sync,
        messages=[{"role": "user", "content": f"Summarize concisely in 2-3 bullets:\n\n{text}"}],
        max_tokens=256,
    )
    return _extract_text(resp, f"Summary: {text[:100]}...")


async def analyze_image(image_url="", question="", **kw):
    """Analyze an image using the provider's vision capabilities."""
    provider = _get_provider()
    if not provider.has_api_key():
        return "No API key available for image analysis."

    question = question or "Describe this image in detail, noting key elements, text, and context."

    if provider.name == "anthropic":
        messages = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": image_url}},
            {"type": "text", "text": question},
        ]}]
    else:
        # OpenAI-compatible format
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": question},
        ]}]

    try:
        resp = await asyncio.to_thread(
            provider.create_sync,
            messages=messages,
            max_tokens=1024,
        )
        return _extract_text(resp, "Could not analyze the image.")
    except Exception as e:
        return f"Image analysis failed: {e}"


# ── Sandbox-required capabilities (CLI-powered) ─────────────────────


def _shell_quote(s: str) -> str:
    """Shell-quote a string for use in bash -c."""
    return "'" + s.replace("'", "'\\''") + "'"


async def claude_code(task="", **kw):
    """Run a task using the Claude Code CLI inside the sandbox.

    When running inside Docker, stderr is passed through to the outer process
    so that sandbox streaming can pick up progress updates.
    """
    if not task:
        return "Error: task description is required."

    try:
        # Detect if we're inside a Docker sandbox container
        in_sandbox = os.path.exists("/.dockerenv")

        result = subprocess.run(
            ["claude", "-p", task, "--output-format", "text"],
            stdout=subprocess.PIPE,
            # In sandbox: let stderr flow through to outer process for streaming
            # On host: capture stderr to include in result
            stderr=None if in_sandbox else subprocess.PIPE,
            text=True, timeout=120,
            env={**os.environ, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
        )
        output = result.stdout
        if not in_sandbox and result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        if result.returncode != 0 and not output.strip():
            output = f"claude_code exited with code {result.returncode}"
        output = output.strip() or "(no output)"
        if len(output) > 2000:
            output = output[:1997] + "..."
        return output
    except subprocess.TimeoutExpired:
        return "Error: claude_code timed out (120s limit)"
    except FileNotFoundError:
        return "Error: claude CLI not found. Is it installed in the sandbox?"
    except Exception as e:
        return f"claude_code error: {e}"


async def openai_code(task="", **kw):
    """Run a task using the OpenAI Codex CLI inside the sandbox.

    When running inside Docker, stderr is passed through for progress streaming.
    """
    if not task:
        return "Error: task description is required."

    try:
        in_sandbox = os.path.exists("/.dockerenv")

        result = subprocess.run(
            ["codex", "--full-auto", task],
            stdout=subprocess.PIPE,
            stderr=None if in_sandbox else subprocess.PIPE,
            text=True, timeout=120,
        )
        output = result.stdout
        if not in_sandbox and result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        if result.returncode != 0 and not output.strip():
            output = f"openai_code exited with code {result.returncode}"
        output = output.strip() or "(no output)"
        if len(output) > 2000:
            output = output[:1997] + "..."
        return output
    except subprocess.TimeoutExpired:
        return "Error: openai_code timed out (120s limit)"
    except FileNotFoundError:
        return "Error: codex CLI not found. Is it installed in the sandbox?"
    except Exception as e:
        return f"openai_code error: {e}"


# ── Registry ─────────────────────────────────────────────────────────
# 3-tuples: (description, function, sandbox_required)

CAPABILITY_REGISTRY = {
    "web_search":      ("Search the web", web_search, False),
    "summarize_text":  ("Summarize text", summarize_text, False),
    "analyze_image":   ("Analyze an image with vision AI", analyze_image, False),
    "claude_code":     ("Run a task using Claude Code CLI (can write code, execute it, read/write files, search the web, and more)", claude_code, True),
    "openai_code":     ("Run a task using OpenAI Codex CLI (can write code, execute it, read/write files, and more)", openai_code, True),
}
