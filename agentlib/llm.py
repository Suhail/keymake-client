"""LLM provider abstraction — Anthropic, OpenAI, and Ollama backends."""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ContentBlock:
    type: str  # "text" or "tool_use"
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: list[ContentBlock]


class LLMStream(ABC):
    @abstractmethod
    async def __aenter__(self):
        return self

    @abstractmethod
    async def __aexit__(self, *args):
        pass

    @abstractmethod
    async def text_chunks(self):
        """Async iterator yielding text delta strings."""

    @abstractmethod
    async def get_response(self) -> LLMResponse:
        """Return the final complete response after streaming."""


class LLMProvider(ABC):
    def __init__(self, model: str, auto_approve_model: str | None = None):
        self.model = model
        self.auto_approve_model = auto_approve_model or model

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def create(self, *, messages, max_tokens=1024, system=None, tools=None) -> LLMResponse: ...

    @abstractmethod
    def stream(self, *, messages, max_tokens=1024, system=None, tools=None) -> LLMStream: ...

    def wrap_assistant_content(self, response: LLMResponse):
        """Return the value to store as {"role": "assistant", "content": <this>}."""
        return response.content

    def wrap_tool_results(self, results: list[dict]) -> dict:
        """Return the message dict to append for tool results."""
        return {"role": "user", "content": results}

    def format_tool_result(self, tool_use_id: str, content: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}

    @abstractmethod
    def api_key_var(self) -> str: ...

    def has_api_key(self) -> bool:
        var = self.api_key_var()
        return not var or bool(os.environ.get(var))

    def api_key_for_sandbox(self) -> str | None:
        var = self.api_key_var()
        return os.environ.get(var) if var else None


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class _AnthropicStream(LLMStream):
    def __init__(self, ctx):
        self._ctx = ctx
        self._stream = None

    async def __aenter__(self):
        self._stream = await self._ctx.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self._ctx.__aexit__(*args)

    async def text_chunks(self):
        async for text in self._stream.text_stream:
            yield text

    async def get_response(self) -> LLMResponse:
        msg = await self._stream.get_final_message()
        return _anthropic_to_response(msg)


def _anthropic_to_response(msg) -> LLMResponse:
    blocks = []
    for b in msg.content:
        if b.type == "text":
            blocks.append(ContentBlock(type="text", text=b.text))
        elif b.type == "tool_use":
            blocks.append(ContentBlock(type="tool_use", id=b.id, name=b.name, input=b.input))
    return LLMResponse(content=blocks)


class AnthropicProvider(LLMProvider):
    @property
    def name(self) -> str:
        return "anthropic"

    def _client(self):
        import anthropic
        return anthropic.AsyncAnthropic()

    def _build_kw(self, messages, max_tokens, system=None, tools=None):
        kw = dict(model=self.model, max_tokens=max_tokens, messages=messages)
        if system:
            kw["system"] = system
        if tools:
            kw["tools"] = tools
        return kw

    async def create(self, *, messages, max_tokens=1024, system=None, tools=None) -> LLMResponse:
        resp = await self._client().messages.create(**self._build_kw(messages, max_tokens, system, tools))
        return _anthropic_to_response(resp)

    def stream(self, *, messages, max_tokens=1024, system=None, tools=None) -> _AnthropicStream:
        return _AnthropicStream(self._client().messages.stream(**self._build_kw(messages, max_tokens, system, tools)))

    def wrap_assistant_content(self, response: LLMResponse):
        import anthropic.types as at
        blocks = []
        for b in response.content:
            if b.type == "text":
                blocks.append(at.TextBlock(type="text", text=b.text))
            elif b.type == "tool_use":
                blocks.append(at.ToolUseBlock(type="tool_use", id=b.id, name=b.name, input=b.input))
        return blocks

    def api_key_var(self) -> str:
        return "ANTHROPIC_API_KEY"

    def create_sync(self, *, messages, max_tokens=1024, tools=None) -> LLMResponse:
        import anthropic
        resp = anthropic.Anthropic().messages.create(**self._build_kw(messages, max_tokens, tools=tools))
        return _anthropic_to_response(resp)


# ---------------------------------------------------------------------------
# OpenAI (also used for Ollama via base_url override)
# ---------------------------------------------------------------------------

def _openai_convert_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic-format tool defs to OpenAI function-calling format."""
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return out


def _block_text(block) -> str | None:
    if isinstance(block, ContentBlock):
        return block.text if block.type == "text" else None
    if isinstance(block, dict):
        return block.get("text") if block.get("type") == "text" else None
    return getattr(block, "text", None) if getattr(block, "type", None) == "text" else None


def _block_tool_call(block) -> dict | None:
    btype = block.type if isinstance(block, ContentBlock) else (
        block.get("type") if isinstance(block, dict) else getattr(block, "type", None))
    if btype != "tool_use":
        return None
    bid = block.id if isinstance(block, ContentBlock) else (
        block.get("id", "") if isinstance(block, dict) else getattr(block, "id", ""))
    bname = block.name if isinstance(block, ContentBlock) else (
        block.get("name", "") if isinstance(block, dict) else getattr(block, "name", ""))
    binput = block.input if isinstance(block, ContentBlock) else (
        block.get("input", {}) if isinstance(block, dict) else getattr(block, "input", {}))
    return {"id": bid, "type": "function", "function": {"name": bname, "arguments": json.dumps(binput)}}


def _openai_convert_messages(messages: list[dict]) -> list[dict]:
    """Convert Anthropic-format conversation history to OpenAI format."""
    out = []
    for m in messages:
        role = m["role"]
        content = m["content"]

        if role == "assistant" and isinstance(content, list):
            text_parts = []
            tool_calls = []
            for block in content:
                txt = _block_text(block)
                if txt is not None:
                    text_parts.append(txt)
                tc = _block_tool_call(block)
                if tc:
                    tool_calls.append(tc)
            msg = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)

        elif role == "user" and isinstance(content, list):
            has_tool_results = any(
                isinstance(item, dict) and item.get("type") == "tool_result"
                for item in content
            )
            if has_tool_results:
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        out.append({
                            "role": "tool",
                            "tool_call_id": item["tool_use_id"],
                            "content": item.get("content", ""),
                        })
                    else:
                        out.append({"role": "user", "content": str(item)})
            else:
                # Multimodal content (text + images) — pass through as-is
                out.append({"role": "user", "content": content})
        else:
            out.append({"role": role, "content": content if isinstance(content, str) else str(content)})
    return out


def _openai_to_response(choice) -> LLMResponse:
    blocks = []
    msg = choice.message
    if msg.content:
        blocks.append(ContentBlock(type="text", text=msg.content))
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            blocks.append(ContentBlock(type="tool_use", id=tc.id, name=tc.function.name, input=args))
    return LLMResponse(content=blocks)


class _OpenAIStream(LLMStream):
    def __init__(self, client, kw):
        self._client = client
        self._kw = kw
        self._stream = None

    async def __aenter__(self):
        self._stream = await self._client.chat.completions.create(**self._kw, stream=True)
        return self

    async def __aexit__(self, *args):
        pass

    async def text_chunks(self):
        full_content = ""
        tool_calls_map = {}
        async for chunk in self._stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            if delta.content:
                full_content += delta.content
                yield delta.content
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_calls_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_map[idx]["arguments"] += tc_delta.function.arguments

        blocks = []
        if full_content:
            blocks.append(ContentBlock(type="text", text=full_content))
        for _idx in sorted(tool_calls_map):
            tc = tool_calls_map[_idx]
            try:
                args = json.loads(tc["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            blocks.append(ContentBlock(type="tool_use", id=tc["id"], name=tc["name"], input=args))
        self._final = LLMResponse(content=blocks)

    async def get_response(self) -> LLMResponse:
        return self._final


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, auto_approve_model: str | None = None, base_url: str | None = None):
        super().__init__(model, auto_approve_model)
        self.base_url = base_url

    @property
    def name(self) -> str:
        return "openai"

    def _client(self):
        from openai import AsyncOpenAI
        kw = {}
        if self.base_url:
            kw["base_url"] = self.base_url
        return AsyncOpenAI(**kw)

    def _client_sync(self):
        from openai import OpenAI
        kw = {}
        if self.base_url:
            kw["base_url"] = self.base_url
        return OpenAI(**kw)

    def _build_kw(self, messages, max_tokens, system=None, tools=None):
        msgs = _openai_convert_messages(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        kw = dict(model=self.model, max_completion_tokens=max_tokens, messages=msgs)
        if tools:
            kw["tools"] = _openai_convert_tools(tools)
        return kw

    async def create(self, *, messages, max_tokens=1024, system=None, tools=None) -> LLMResponse:
        resp = await self._client().chat.completions.create(**self._build_kw(messages, max_tokens, system, tools))
        return _openai_to_response(resp.choices[0])

    def stream(self, *, messages, max_tokens=1024, system=None, tools=None) -> _OpenAIStream:
        return _OpenAIStream(self._client(), self._build_kw(messages, max_tokens, system, tools))

    def api_key_var(self) -> str:
        return "OPENAI_API_KEY"

    def create_sync(self, *, messages, max_tokens=1024, tools=None) -> LLMResponse:
        resp = self._client_sync().chat.completions.create(**self._build_kw(messages, max_tokens, tools=tools))
        return _openai_to_response(resp.choices[0])


class OllamaProvider(OpenAIProvider):
    def __init__(self, model: str, auto_approve_model: str | None = None,
                 base_url: str = "http://localhost:11434/v1"):
        super().__init__(model, auto_approve_model, base_url=base_url)

    @property
    def name(self) -> str:
        return "ollama"

    def api_key_var(self) -> str:
        return ""

    def _client(self):
        from openai import AsyncOpenAI
        return AsyncOpenAI(base_url=self.base_url, api_key="ollama")

    def _client_sync(self):
        from openai import OpenAI
        return OpenAI(base_url=self.base_url, api_key="ollama")

    def _build_kw(self, messages, max_tokens, system=None, tools=None):
        kw = super()._build_kw(messages, max_tokens, system, tools)
        kw["max_tokens"] = kw.pop("max_completion_tokens")
        return kw


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-5.2",
    "ollama": "llama3.2",
}

DEFAULT_AUTO_APPROVE_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-5.2-mini",
    "ollama": "llama3.2",
}


def make_provider(
    provider_name: str = "anthropic",
    model: str | None = None,
    auto_approve_model: str | None = None,
    base_url: str | None = None,
) -> LLMProvider:
    provider_name = provider_name.lower()
    model = model or DEFAULT_MODELS.get(provider_name, "claude-sonnet-4-6")
    auto_approve_model = auto_approve_model or DEFAULT_AUTO_APPROVE_MODELS.get(provider_name)

    if provider_name == "anthropic":
        return AnthropicProvider(model, auto_approve_model)
    elif provider_name == "openai":
        return OpenAIProvider(model, auto_approve_model, base_url=base_url)
    elif provider_name == "ollama":
        url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OllamaProvider(model, auto_approve_model, base_url=url)
    elif provider_name == "custom":
        if not base_url:
            raise ValueError("Custom provider requires base_url")
        return OpenAIProvider(model, auto_approve_model, base_url=base_url)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
