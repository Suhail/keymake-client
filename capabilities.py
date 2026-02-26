"""Capability implementations available to agents."""

import asyncio
import sys
from datetime import datetime, timedelta
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


async def generate_code(task="", **kw):
    provider = _get_provider()
    if not provider.has_api_key():
        return "class RateLimitedScraper:\n    def __init__(self): pass"
    resp = await asyncio.to_thread(
        provider.create_sync,
        messages=[{"role": "user", "content": f"Write concise Python code (<30 lines, no markdown fences) for: {task}"}],
        max_tokens=512,
    )
    return _extract_text(resp, "# no code generated")


async def schedule_meeting(topic="", agenda="", **kw):
    when = datetime.now() + timedelta(days=1, hours=2)
    return (
        f"Meeting scheduled!\n"
        f"  Topic: {topic or 'Collaboration'}\n"
        f"  When: {when.strftime('%A %B %d, %I:%M %p')}\n"
        f"  Duration: 30 min\n"
        f"  Agenda: {(agenda or 'TBD')[:200]}\n"
        f"  Link: https://meet.example.com/alice-bob-{when.strftime('%Y%m%d')}"
    )


CAPABILITY_REGISTRY = {
    "web_search": ("Search the web", web_search),
    "summarize_text": ("Summarize text", summarize_text),
    "generate_code": ("Generate Python code", generate_code),
    "schedule_meeting": ("Schedule a meeting", schedule_meeting),
}
