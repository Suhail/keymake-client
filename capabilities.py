"""Capability implementations available to agents."""

import asyncio
import json
import os
import subprocess
import sys
import urllib.request
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


# ── Web hosting capabilities ──────────────────────────────────────────

_CONTENT_TYPE_MAP = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
    ".xml": "application/xml",
    ".webp": "image/webp",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
}

_HERE_NOW_API = "https://here.now/api/v1"


def _guess_content_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _CONTENT_TYPE_MAP.get(ext, "application/octet-stream")


async def publish_site(files=None, title="", **kw):
    """Publish files to here.now for instant web hosting.

    Uses the here.now 3-step API: create → upload → finalize.
    Returns the live URL on success.
    """
    if not files or not isinstance(files, list):
        return (
            "Error: files is required. Pass a list like: "
            '[{"path": "index.html", "content": "<html>...</html>"}]'
        )

    # Validate files
    for f in files:
        if not isinstance(f, dict) or "path" not in f or "content" not in f:
            return 'Error: each file must have "path" and "content" keys.'

    try:
        # Step 1: Declare the publish
        file_meta = []
        for f in files:
            content_bytes = f["content"].encode("utf-8")
            file_meta.append({
                "path": f["path"],
                "size": len(content_bytes),
                "contentType": _guess_content_type(f["path"]),
            })

        payload: dict = {"files": file_meta}
        if title:
            payload["viewer"] = {"title": title}

        req = urllib.request.Request(
            f"{_HERE_NOW_API}/publish",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        site_url = result.get("siteUrl", "")
        slug = result.get("slug", "")
        upload_info = result.get("upload", {})
        version_id = upload_info.get("versionId", "")
        uploads = upload_info.get("uploads", [])

        if not uploads or not version_id:
            return f"Error: unexpected API response — missing upload info. Got: {json.dumps(result)[:500]}"

        # Step 2: Upload each file to its presigned URL
        for i, upload_entry in enumerate(uploads):
            presigned_url = upload_entry.get("url", "")
            if not presigned_url:
                return f"Error: missing presigned URL for file {files[i]['path']}"

            content_bytes = files[i]["content"].encode("utf-8")
            ct = file_meta[i]["contentType"]

            put_req = urllib.request.Request(
                presigned_url,
                data=content_bytes,
                headers={"Content-Type": ct},
                method="PUT",
            )
            urllib.request.urlopen(put_req, timeout=60)

        # Step 3: Finalize the publish
        finalize_req = urllib.request.Request(
            f"{_HERE_NOW_API}/publish/{slug}/finalize",
            data=json.dumps({"versionId": version_id}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(finalize_req, timeout=30)

        return f"Site published: {site_url}"

    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500] if hasattr(e, "read") else ""
        if e.code == 429:
            return (
                "Rate limit reached: here.now allows 5 publishes per hour per IP "
                "for anonymous users. Do NOT retry — tell the user the rate limit "
                "has been hit and they should wait before publishing again."
            )
        return f"Publish failed (HTTP {e.code}): {body}"
    except urllib.error.URLError as e:
        return f"Publish failed (network error): {e.reason}"
    except Exception as e:
        return f"Publish failed: {e}"


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
    "publish_site":    ("Publish a website to here.now — pass files with path and content to get a live URL", publish_site, False),
    "claude_code":     ("Run a task using Claude Code CLI (can write code, execute it, read/write files, search the web, and more)", claude_code, True),
    "openai_code":     ("Run a task using OpenAI Codex CLI (can write code, execute it, read/write files, and more)", openai_code, True),
}

# JSON Schema for each capability's params object, so the LLM knows what to pass.
CAPABILITY_PARAMS = {
    "web_search": {
        "properties": {
            "query": {"type": "string", "description": "The search query"},
        },
        "required": ["query"],
    },
    "summarize_text": {
        "properties": {
            "text": {"type": "string", "description": "The text to summarize"},
        },
        "required": ["text"],
    },
    "analyze_image": {
        "properties": {
            "image_url": {"type": "string", "description": "URL of the image to analyze"},
            "question": {"type": "string", "description": "Question to ask about the image (optional)"},
        },
        "required": ["image_url"],
    },
    "publish_site": {
        "properties": {
            "files": {
                "type": "array",
                "description": "List of files to publish. Each file needs a relative path and its content as a string.",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative file path, e.g. index.html or css/style.css"},
                        "content": {"type": "string", "description": "The full file content as a string"},
                    },
                    "required": ["path", "content"],
                },
            },
            "title": {"type": "string", "description": "Optional site title for the viewer page"},
        },
        "required": ["files"],
    },
    "claude_code": {
        "properties": {
            "task": {"type": "string", "description": "The task description for Claude Code to execute"},
        },
        "required": ["task"],
    },
    "openai_code": {
        "properties": {
            "task": {"type": "string", "description": "The task description for Codex to execute"},
        },
        "required": ["task"],
    },
}
