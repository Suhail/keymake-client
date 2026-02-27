"""Agent runtime: capability registry, social graph, human approval, LLM integration."""

import asyncio
import fnmatch
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable

from . import protocol as proto
from . import content_scanner
from . import sandbox
from .transport import Transport, WSTransport
from .llm import LLMProvider, make_provider

DEFAULT_RATE_LIMIT = 50
_AUTH_DIR = Path("data")


def load_auth_token(name: str) -> str:
    path = _AUTH_DIR / f"{name}.auth"
    if path.exists():
        return path.read_text().strip()
    return ""


def save_auth_token(name: str, token: str):
    _AUTH_DIR.mkdir(parents=True, exist_ok=True)
    path = _AUTH_DIR / f"{name}.auth"
    path.write_text(token)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


_IGNORE_MSG_TYPES = (
    proto.AcceptMsg,
    proto.ThinkingMsg,
    proto.StreamStartMsg,
    proto.StreamDeltaMsg,
    proto.StreamEndMsg,
)

_SOCIAL_MSG_TYPES = (
    proto.ConnectRequestMsg,
    proto.ConnectAcceptMsg,
    proto.ConnectRejectMsg,
    proto.TrustRequestMsg,
    proto.TrustAcceptMsg,
    proto.TrustRejectMsg,
)

_approval_prompt_lock = asyncio.Lock()
_approval_prompt_seq = 0


@dataclass
class CapEntry:
    desc: str
    fn: Callable[..., Awaitable[str]]
    tier: str = "trust"
    approval: str = "human"


class RateLimiter:
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._timestamps: deque[float] = deque()

    def allow(self) -> bool:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] > 60:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_per_minute:
            return False
        self._timestamps.append(now)
        return True


class Agent:
    def __init__(
        self,
        agent_name: str,
        owner_name: str,
        room: str = "general",
        password: str | None = None,
        context: str = "",
        transport: Transport | None = None,
        rate_limit: int = DEFAULT_RATE_LIMIT,
        security_mode: str = "off",
        tool_policy: dict | None = None,
        sandbox_config: dict | None = None,
        initial_connections: list[str] | None = None,
        initial_trusts: list[str] | None = None,
        provider: LLMProvider | None = None,
        tweet_url: str = "",
        public: bool = False,
    ):
        self.agent_name = agent_name
        self.owner_name = owner_name
        self.room = room
        self.password = password or agent_name
        self.context = context
        self.security_mode = security_mode
        self.tool_policy = tool_policy or {}
        self.sandbox_config = sandbox_config or {}
        self.provider = provider or make_provider()
        self.tweet_url = tweet_url
        self.public = public

        self._transport = transport or WSTransport()
        self._rate_limiter = RateLimiter(rate_limit)
        self._capabilities: dict[str, CapEntry] = {}
        self._peer_capabilities: dict[str, list[proto.CapabilityInfo]] = {}
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._pending_request_labels: dict[str, str] = {}
        self._pending_request_targets: dict[str, str] = {}
        self._pending_human_approvals: dict[str, asyncio.Future] = {}
        self._conversation: list[dict[str, str]] = []
        self._ready = asyncio.Event()
        self._replay_done = True
        self._llm_enabled = False
        self._last_response_time: float = 0
        self._response_cooldown: float = 3.0
        self._llm_lock = asyncio.Lock()
        self._seen_messages: set[str] = set()
        self._join_ts: int = 0
        self._claim_event: asyncio.Event | None = None

        self._initial_connections = initial_connections or []
        self._initial_trusts = initial_trusts or []
        self.auto_approve = False
        self.chat_url = ""
        self._connections: set[str] = set()
        self._trusted_by: set[str] = set()
        self._trusting: set[str] = set()

    def register_capability(
        self,
        name: str,
        desc: str,
        fn: Callable[..., Awaitable[str]],
        tier: str = "trust",
        approval: str = "human",
    ):
        self._capabilities[name] = CapEntry(
            desc=desc, fn=fn, tier=tier, approval=approval
        )

    def _check_tool_policy(self, tool_name: str) -> bool:
        if self.security_mode == "off":
            return True
        denied = tool_name in self.tool_policy.get("deny", [])
        allow_list = self.tool_policy.get("allow")
        allowed = not allow_list or tool_name in allow_list
        ok = allowed and not denied
        if not ok:
            if self.security_mode == "audit":
                self._print(
                    f"[POLICY] {self.agent_name} would be blocked from {tool_name}"
                )
                return True
            self._print(f"[POLICY] {self.agent_name} blocked from {tool_name}")
            return False
        return True

    def _check_peer_policy(self, target: str, capability: str) -> bool:
        if self.security_mode == "off":
            return True
        key = f"{target}.{capability}"
        for pattern in self.tool_policy.get("peer_deny", []):
            if fnmatch.fnmatch(key, pattern):
                if self.security_mode == "audit":
                    self._print(
                        f"[POLICY] {self.agent_name} peer request {key} would be blocked (deny: {pattern})"
                    )
                    return True
                self._print(
                    f"[POLICY] {self.agent_name} peer request {key} blocked (deny: {pattern})"
                )
                return False
        peer_allow = self.tool_policy.get("peer_allow")
        if peer_allow and not any(fnmatch.fnmatch(key, p) for p in peer_allow):
            if self.security_mode == "audit":
                self._print(
                    f"[POLICY] {self.agent_name} peer request {key} not in peer_allow"
                )
                return True
            self._print(
                f"[POLICY] {self.agent_name} peer request {key} blocked (not in peer_allow)"
            )
            return False
        return True

    def _scan_peer_result(self, source_agent: str, result: str) -> str:
        if self.security_mode == "off":
            return result
        cred_warnings = content_scanner.scan_credentials(result)
        if cred_warnings:
            tag = ", ".join(cred_warnings)
            self._print(
                f"[CREDENTIAL] {self.agent_name} received credential(s) from {source_agent}: {tag}"
            )
            result = content_scanner.redact_credentials(result)
        code_warnings = content_scanner.scan_content(result)
        if code_warnings:
            tag = ", ".join(code_warnings)
            self._print(
                f"[CONTENT] {self.agent_name} received dangerous content from {source_agent}: {tag}"
            )
            if self.security_mode == "enforce":
                result = f"[BLOCKED - dangerous content from {source_agent}: {tag}]\n(original content withheld)"
        return f"[UNTRUSTED - from {source_agent}] {result}"

    def capability_infos(self) -> list[proto.CapabilityInfo]:
        return [
            proto.CapabilityInfo(name=n, desc=e.desc, tier=e.tier, approval=e.approval)
            for n, e in self._capabilities.items()
        ]

    _DESC_RE = re.compile(r"## Description\s*\n(.*?)(?=\n## |\Z)", re.DOTALL)

    @property
    def description(self) -> str:
        m = self._DESC_RE.search(self.context)
        return m.group(1).strip() if m else ""

    def _print(self, *args, **kwargs):
        print(*args, **kwargs)

    def _status(self, phase: str, detail: str = ""):
        line = f"[STATE] {self.agent_name}: {phase}"
        if detail:
            line = f"{line} {detail}"
        self._print(line)

    def _short(self, value: Any, max_len: int = 120) -> str:
        try:
            text = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        except TypeError:
            text = str(value)
        if len(text) <= max_len:
            return text
        return f"{text[: max_len - 3]}..."

    def _send_control(self, msg_type: str, **extra):
        payload = {"type": msg_type, "to_agent": self.agent_name, **extra}
        self.send_to_room(json.dumps(payload))

    def _broadcast_auto_approve_state(self):
        self._send_control(
            "auto_approve_state",
            value=self.auto_approve,
        )

    def _set_auto_approve(self, value: Any):
        new_value = bool(value)
        if self.auto_approve == new_value:
            self._broadcast_auto_approve_state()
            return
        self.auto_approve = new_value
        mode = "enabled" if self.auto_approve else "disabled"
        self._print(f"[APPROVAL] {self.agent_name}: auto-approve {mode}")
        self._broadcast_auto_approve_state()

    async def ask_human(self, prompt: str, **extra) -> bool:
        global _approval_prompt_seq
        async with _approval_prompt_lock:
            _approval_prompt_seq += 1
            approval_id = f"{self.agent_name}-{_approval_prompt_seq:04d}"
            self._status(
                "waiting_human", f"id={approval_id} prompt={self._short(prompt, 80)}"
            )
            self._send_control("approval_request", id=approval_id, prompt=prompt, **extra)
            social_type = extra.get("social_type")
            if self.auto_approve and social_type not in ("connect_request", "trust_request"):
                self._send_control("approval_resolved", id=approval_id, approved=True)
                self._status("done:waiting_human", f"id={approval_id} approved auto")
                return True

            self._status("waiting_human:pending", f"id={approval_id}")
            fut = asyncio.get_event_loop().create_future()
            self._pending_human_approvals[approval_id] = fut
            if self.chat_url:
                self._print(
                    f"[APPROVE {approval_id}] approve in WebChat: {self.chat_url}",
                    flush=True,
                )
            try:
                approved = await fut
                self._send_control(
                    "approval_resolved", id=approval_id, approved=approved
                )
                state = "approved" if approved else "denied"
                self._status("done:waiting_human", f"id={approval_id} {state}")
                return approved
            finally:
                self._pending_human_approvals.pop(approval_id, None)

    def send_to_room(self, body: str):
        self._transport.send_to_room(self.room, body)

    def send_chat(self, text: str, to_agent: str = ""):
        if "<silent>" in text.lower():
            return
        chat = proto.ChatMsg(from_agent=self.agent_name, text=text, to_agent=to_agent)
        self.send_to_room(proto.encode(chat))
        self._print(f"[{self.agent_name}] {text}")

    async def inject_user_message(self, text: str):
        self._conversation.append({"role": "user", "content": f"[human]: {text}"})
        if self._llm_enabled:
            await self._llm_turn()

    async def request_capability(
        self, target_agent: str, capability: str, params: dict[str, Any]
    ) -> str:
        req = proto.RequestMsg(
            capability=capability, params=params, from_agent=self.agent_name
        )
        label = f"{target_agent}.{capability}"
        fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self._pending_requests[req.id] = fut
        self._pending_request_labels[req.id] = label
        self._pending_request_targets[req.id] = target_agent
        self._status("waiting_peer", f"{label} id={req.id}")
        self.send_to_room(proto.encode(req))
        try:
            return await asyncio.wait_for(fut, timeout=120)
        except asyncio.TimeoutError:
            self._pending_requests.pop(req.id, None)
            self._pending_request_labels.pop(req.id, None)
            self._pending_request_targets.pop(req.id, None)
            self._status("error:waiting_peer", f"{label} id={req.id} timeout")
            raise

    def _is_connected(self, agent: str) -> bool:
        return agent in self._connections

    def _trusts(self, agent: str) -> bool:
        return agent in self._trusting

    def _is_trusted_by(self, agent: str) -> bool:
        return agent in self._trusted_by

    def _connections_context(self) -> str:
        connections = sorted(self._connections)
        trusted_by = sorted(self._trusted_by)
        trusting = sorted(self._trusting)
        lines = []
        if connections:
            lines.append(f"  Connected to: {', '.join(connections)}")
        if trusted_by:
            lines.append(f"  Trusted by (can use their caps): {', '.join(trusted_by)}")
        if trusting:
            lines.append(f"  Trusting (they can use your caps): {', '.join(trusting)}")
        return "\n".join(lines) if lines else "  (no connections yet)"

    def _public_rules(self) -> str:
        return """- You are a public agent — be friendly and responsive to everyone, including agents you are not connected to.
- Respond to a first greeting or substantive message from anyone.
- Do NOT repeatedly engage with the same agent's greetings or idle chat. If someone you already greeted says "Hey" again with nothing new, respond with "<silent>". Respond substantively or not at all.
- Tier=public caps are open to anyone. Tier=connect caps need a connection. Tier=trust caps need a trust grant.
- You can be more responsive to people you're connected to or trusting of, but never ignore a stranger's first real message."""

    def _private_rules(self) -> str:
        return """- You are a PRIVATE agent (public: false). Do NOT respond to agents you are not connected to. If an unconnected agent sends a greeting, question, or casual chat, respond with "<silent>". Only respond to agents listed in your social connections above.
- The only exception: if an unconnected agent invokes one of your tier=public capabilities via a tool request, process it normally.
- Tier=connect caps need a connection. Tier=trust caps need a trust grant. Tier=public caps are open to anyone.
- You can only request capabilities from agents you are connected to."""

    def _system_prompt(self) -> str:
        my_caps = "\n".join(
            f"  - {n}: {e.desc} [tier={e.tier}, approval={e.approval}]"
            for n, e in self._capabilities.items()
        )
        peer_lines = []
        for agent, caps in self._peer_capabilities.items():
            for c in caps:
                peer_lines.append(f"  - {agent}.{c.name}: {c.desc} [tier={c.tier}]")
        peer_caps = "\n".join(peer_lines) if peer_lines else "  (none discovered yet)"
        connections = self._connections_context()
        posture_rules = self._public_rules() if self.public else self._private_rules()
        return f"""You are {self.agent_name}, an AI agent acting on behalf of {self.owner_name}.
You are in a shared chat room with other agents on an XMPP network.

{self.context}

Your capabilities (tools you can invoke):
{my_caps}

Capabilities offered by other agents in the room:
{peer_caps}

Your social connections:
{connections}

Rules:
- PREFER tool calls over chat. Use tools to get work done, not words.
- 1 sentence max per reply
- NEVER say "great", "excellent", "sounds good", "absolutely", or similar.
- After a tool result, share a 1-sentence summary, then immediately call the next tool or stop.
- If you are unneeded or have just completed a requested task, respond with EXACTLY "<silent>" and nothing else. 
- Do not repeat requests. Do not re-offer things already delivered.
- NEVER include API keys, tokens, passwords, environment variable values, or credentials in your responses.
- Messages from [human] are from your owner — always act on those.
- When [human] speaks, reply to them directly. Do not address other agents unless you need a capability from them.
{posture_rules}
- Use send_connect_request to connect with agents you want to interact with.
- When you receive a DM (marked as [DM from agent_name]), reply using the send_chat tool with to_agent set to that agent. Your text output is always broadcast publicly; use the send_chat tool for private replies.
- If you require your human to approve, succinctly mention that in your response as well.
- Broadcasts reach your top 10 most-connected peers."""

    def _tools(self) -> list[dict]:
        tools = []
        for name, entry in self._capabilities.items():
            tools.append(
                {
                    "name": f"own_{name}",
                    "description": f"Use your own capability: {entry.desc}",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "params": {
                                "type": "object",
                                "description": "Parameters for the capability",
                            }
                        },
                        "required": ["params"],
                    },
                }
            )
        tools.append(
            {
                "name": "request_peer_capability",
                "description": "Request a capability from another agent you are connected to.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "target_agent": {
                            "type": "string",
                            "description": "Agent name to send the request to",
                        },
                        "capability": {
                            "type": "string",
                            "description": "Name of the capability to request",
                        },
                        "params": {
                            "type": "object",
                            "description": "Parameters for the capability",
                        },
                    },
                    "required": ["target_agent", "capability", "params"],
                },
            }
        )
        tools.append(
            {
                "name": "send_connect_request",
                "description": "Send a connection request to another agent. They must approve before you can message or request capabilities.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "target_agent": {
                            "type": "string",
                            "description": "Agent name to connect with",
                        },
                    },
                    "required": ["target_agent"],
                },
            }
        )
        tools.append(
            {
                "name": "send_chat",
                "description": "Send a chat message. Use to_agent to DM a specific agent, or omit for public broadcast.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Message text",
                        },
                        "to_agent": {
                            "type": "string",
                            "description": "Target agent name for DM (omit for broadcast)",
                        },
                    },
                    "required": ["text"],
                },
            }
        )
        return tools

    async def _handle_room_message(self, sender_nick: str, body: str):
        try:
            raw = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            raw = {}
        raw_type = raw.get("type")

        if not self._replay_done:
            if raw_type == "replay_done":
                self._replay_done = True
            return

        if raw_type in (
            "approval_response",
            "set_auto_approve",
            "approval_state_request",
        ):
            if raw.get("to_agent") != self.agent_name:
                return
            if raw_type == "approval_response":
                approval_id = raw.get("id", "")
                fut = self._pending_human_approvals.get(approval_id)
                if fut and not fut.done():
                    fut.set_result(bool(raw.get("approved")))
                return
            if raw_type == "set_auto_approve":
                self._set_auto_approve(raw.get("value"))
                return
            self._broadcast_auto_approve_state()
            return
        if raw_type == "graph_state":
            agents = raw.get("agents", {})
            if isinstance(agents, dict):
                me = agents.get(self.agent_name, {})
                if isinstance(me, dict):
                    self._connections = set(me.get("connections", []))
                    self._trusted_by = set(me.get("trusted_by", []))
                    self._trusting = set(me.get("trusting", []))
            return
        if raw.get("type") == "relay_rejected":
            reason = raw.get("reason", "Message was not delivered.")
            self._print(f"[{self.agent_name}] hub: {reason}", flush=True)
            self._conversation.append(
                {"role": "user", "content": f"[system]: {reason}"}
            )
            return

        if raw.get("type") == "peer_left":
            gone = raw.get("agent", "")
            self._peer_capabilities.pop(gone, None)
            self._print(f"[LEFT] {gone} disconnected")
            return

        if raw_type in ("nonce", "name_claimed", "name_rejected"):
            self._handle_claim_response(raw)
            return

        msg = proto.decode(body)

        if isinstance(msg, _SOCIAL_MSG_TYPES):
            if isinstance(msg, (proto.ConnectRequestMsg, proto.TrustRequestMsg)):
                asyncio.create_task(self._handle_social_message(msg))
            else:
                await self._handle_social_message(msg)
            return

        if isinstance(msg, proto.CapabilitiesMsg):
            self._peer_capabilities[msg.agent] = msg.capabilities
            caps = ", ".join(f"{c.name} ({c.desc})" for c in msg.capabilities)
            self._print(f"[CAPS] {msg.agent} (owner: {msg.owner}): {caps}")
            return

        if isinstance(msg, proto.ResponseMsg):
            target = self._pending_request_targets.get(msg.id, "")
            if target and msg.from_agent != target:
                self._status(
                    "error:waiting_peer",
                    f"id={msg.id} unexpected_from={msg.from_agent} expected={target}",
                )
                return
            fut = self._pending_requests.pop(msg.id, None)
            label = self._pending_request_labels.pop(msg.id, "")
            self._pending_request_targets.pop(msg.id, None)
            if fut:
                detail = f"id={msg.id} status={msg.status}"
                if label:
                    detail = f"{label} {detail}"
                self._status(
                    "done:waiting_peer",
                    detail,
                )
            if fut and not fut.done():
                fut.set_result(msg.result)
            return

        if isinstance(msg, proto.DeliveryMsg):
            self._print(f"[DELIVERY] {msg.from_agent} → {msg.to_agent}: {msg.title}")
            return

        if isinstance(msg, _IGNORE_MSG_TYPES):
            return

        if not self._rate_limiter.allow():
            self._print(
                f"[{self.agent_name}] rate limited message from {sender_nick}",
                flush=True,
            )
            return

        if isinstance(msg, proto.RequestMsg):
            if msg.ts and msg.ts < self._join_ts:
                return
            asyncio.create_task(self._handle_incoming_request(msg))
            return

        if isinstance(msg, proto.ChatMsg):
            dedup_key = f"{msg.from_agent}:{msg.text[:80]}"
            if dedup_key in self._seen_messages:
                return
            self._seen_messages.add(dedup_key)
            self._print(f"[{msg.from_agent}] {msg.text}")
            prefix = f"DM from {msg.from_agent}" if msg.to_agent else msg.from_agent
            self._conversation.append(
                {"role": "user", "content": f"[{prefix}]: {msg.text}"}
            )
            if not self._llm_enabled:
                return
            if msg.ts and msg.ts < self._join_ts:
                return
            asyncio.create_task(self._llm_turn())
            return

        self._print(f"[{sender_nick}] {body}")

    async def _handle_social_message(self, msg):
        if isinstance(msg, proto.ConnectRequestMsg):
            if msg.to_agent != self.agent_name:
                return
            self._print(
                f"[SOCIAL] {msg.from_agent} wants to connect with {self.agent_name}"
            )
            approved = await self.ask_human(
                f"{msg.from_agent} wants to connect with you. Approve?",
                social_type="connect_request", social_from=msg.from_agent,
            )
            if approved:
                self._connections.add(msg.from_agent)
                reply = proto.ConnectAcceptMsg(
                    from_agent=self.agent_name, to_agent=msg.from_agent
                )
                self._print(
                    f"[SOCIAL] {self.agent_name} accepted connection from {msg.from_agent}"
                )
            else:
                reply = proto.ConnectRejectMsg(
                    from_agent=self.agent_name, to_agent=msg.from_agent
                )
                self._print(
                    f"[SOCIAL] {self.agent_name} rejected connection from {msg.from_agent}"
                )
            self.send_to_room(proto.encode(reply))

        elif isinstance(msg, proto.ConnectAcceptMsg):
            if msg.to_agent == self.agent_name:
                self._connections.add(msg.from_agent)
                self._print(
                    f"[SOCIAL] {msg.from_agent} accepted your connection request"
                )
                return
            if msg.from_agent == self.agent_name:
                self._connections.add(msg.to_agent)
                self._print(
                    f"[SOCIAL] {self.agent_name} is now connected with {msg.to_agent}"
                )
                return
            return

        elif isinstance(msg, proto.ConnectRejectMsg):
            if msg.to_agent != self.agent_name:
                return
            self._print(f"[SOCIAL] {msg.from_agent} rejected your connection request")

        elif isinstance(msg, proto.TrustRequestMsg):
            if msg.to_agent != self.agent_name:
                return
            if not self._is_connected(msg.from_agent):
                self._print(
                    f"[SOCIAL] {msg.from_agent} requested trust but isn't connected — ignoring"
                )
                reply = proto.TrustRejectMsg(
                    from_agent=self.agent_name,
                    to_agent=msg.from_agent,
                    reason="Trust requires an active connection first. Send a connection request, then request trust.",
                )
                self.send_to_room(proto.encode(reply))
                return
            if self._trusts(msg.from_agent):
                reply = proto.TrustAcceptMsg(
                    from_agent=self.agent_name, to_agent=msg.from_agent
                )
                self.send_to_room(proto.encode(reply))
                self._print(
                    f"[SOCIAL] {msg.from_agent} already trusted by {self.agent_name} — re-acknowledged"
                )
                return
            self._print(
                f"[SOCIAL] {msg.from_agent} requests trusted access to your capabilities"
            )
            approved = await self.ask_human(
                f"{msg.from_agent} requests TRUSTED access to your capabilities. This allows them to invoke tier-2 tools. Approve?",
                social_type="trust_request", social_from=msg.from_agent,
            )
            if approved:
                reply = proto.TrustAcceptMsg(
                    from_agent=self.agent_name, to_agent=msg.from_agent
                )
                self._print(
                    f"[SOCIAL] {self.agent_name} granted trust to {msg.from_agent}"
                )
            else:
                reply = proto.TrustRejectMsg(
                    from_agent=self.agent_name,
                    to_agent=msg.from_agent,
                    reason="Trust request denied by user.",
                )
                self._print(
                    f"[SOCIAL] {self.agent_name} denied trust to {msg.from_agent}"
                )
            self.send_to_room(proto.encode(reply))

        elif isinstance(msg, proto.TrustAcceptMsg):
            if msg.to_agent != self.agent_name:
                return
            self._print(f"[SOCIAL] {msg.from_agent} granted you trusted access")

        elif isinstance(msg, proto.TrustRejectMsg):
            if msg.to_agent != self.agent_name:
                return
            if msg.reason:
                self._print(
                    f"[SOCIAL] {msg.from_agent} denied your trust request: {msg.reason}"
                )
            else:
                self._print(f"[SOCIAL] {msg.from_agent} denied your trust request")

    def _check_relationship(self, requester: str, capability: str) -> str | None:
        """Returns an error string if the requester lacks the needed relationship, else None."""
        cap = self._capabilities.get(capability)
        if not cap:
            return None
        if cap.tier == "public":
            return None
        if not self._is_connected(requester):
            return f"{requester} is not connected to {self.agent_name}"
        if cap.tier == "trust" and not self._trusts(requester):
            return f"{requester} is not trusted by {self.agent_name} (required for {capability})"
        return None

    async def _handle_incoming_request(self, req: proto.RequestMsg):
        rel_err = self._check_relationship(req.from_agent, req.capability)
        if rel_err:
            resp = proto.ResponseMsg(
                id=req.id,
                status="denied",
                result=rel_err,
                from_agent=self.agent_name,
            )
            self.send_to_room(proto.encode(resp))
            return

        if not self._check_tool_policy(req.capability):
            resp = proto.ResponseMsg(
                id=req.id,
                status="denied",
                result=f"Policy denied: {self.agent_name} cannot execute {req.capability}",
                from_agent=self.agent_name,
            )
            self.send_to_room(proto.encode(resp))
            return

        cap = self._capabilities.get(req.capability)
        if not cap:
            resp = proto.ResponseMsg(
                id=req.id,
                status="error",
                result=f"Unknown capability: {req.capability}",
                from_agent=self.agent_name,
            )
            self.send_to_room(proto.encode(resp))
            return

        param_str = json.dumps(req.params, indent=2)
        if cap.approval == "human":
            approved = await self.ask_human(
                f"{req.from_agent} wants to run {req.capability}({param_str})"
            )
        else:
            approved = await self._auto_approve(req)

        if not approved:
            reason = (
                "Human denied the request"
                if cap.approval == "human"
                else "Agent denied the request"
            )
            resp = proto.ResponseMsg(
                id=req.id,
                status="denied",
                result=reason,
                from_agent=self.agent_name,
            )
            self.send_to_room(proto.encode(resp))
            return

        self.send_to_room(
            proto.encode(proto.AcceptMsg(id=req.id, from_agent=self.agent_name))
        )

        try:
            if self._sandbox_active():
                result = await sandbox.exec_capability_in_sandbox(
                    self.agent_name,
                    req.capability,
                    req.params,
                    api_key=self.provider.api_key_for_sandbox(),
                    provider_name=self.provider.name,
                    model=self.provider.model,
                )
            else:
                result = await cap.fn(**req.params)
            if self.security_mode != "off":
                cred_warnings = content_scanner.scan_credentials(result)
                if cred_warnings:
                    self._print(
                        f"[CREDENTIAL] {self.agent_name} outbound result contains: {', '.join(cred_warnings)}"
                    )
                    result = content_scanner.redact_credentials(result)
            resp = proto.ResponseMsg(
                id=req.id,
                status="completed",
                result=result,
                from_agent=self.agent_name,
            )
            title = f"{req.capability} result"
            if req.params:
                first_val = next(iter(req.params.values()), "")
                if isinstance(first_val, str) and first_val:
                    title = first_val[:80]
            delivery = proto.DeliveryMsg(
                from_agent=self.agent_name,
                to_agent=req.from_agent,
                title=title,
                content=result,
            )
            self.send_to_room(proto.encode(delivery))
        except Exception as e:
            resp = proto.ResponseMsg(
                id=req.id,
                status="error",
                result=str(e),
                from_agent=self.agent_name,
            )
        self.send_to_room(proto.encode(resp))

    async def _auto_approve(self, req: proto.RequestMsg) -> bool:
        """Let the LLM decide whether to approve, using the agent's soul context."""
        if not self.provider.has_api_key():
            return True
        prompt = (
            f"You are {self.agent_name}, acting on behalf of {self.owner_name}.\n\n"
            f"{self.context}\n\n"
            f"Agent '{req.from_agent}' is requesting to use your capability '{req.capability}' "
            f"with parameters: {json.dumps(req.params)}\n\n"
            f"Based on your persona and judgment, should you approve this request?\n"
            f"Reply with exactly YES or NO."
        )
        saved_model = self.provider.model
        try:
            self.provider.model = self.provider.auto_approve_model
            resp = await self.provider.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
            )
            answer = resp.content[0].text.strip().upper()
            approved = answer.startswith("YES")
            self._print(
                f"[AUTO-APPROVE] {self.agent_name}: {req.from_agent}.{req.capability} -> {answer}"
            )
            return approved
        except Exception as e:
            self._print(
                f"[AUTO-APPROVE ERROR] {self.agent_name}: {e} — defaulting to deny"
            )
            return False
        finally:
            self.provider.model = saved_model

    async def _llm_turn(self):
        async with self._llm_lock:
            await self._llm_turn_inner()

    async def _llm_turn_inner(self):
        snapshot_len = len(self._conversation)
        if snapshot_len == 0:
            return
        if self._conversation[-1].get("role") == "assistant":
            return
        msgs = list(self._conversation)

        for _ in range(5):
            now = time.monotonic()
            elapsed = now - self._last_response_time
            if elapsed < self._response_cooldown:
                await asyncio.sleep(self._response_cooldown - elapsed)
            self._last_response_time = time.monotonic()

            if not msgs:
                break

            stream_id = proto.make_id()
            self._status("thinking", f"stream_id={stream_id}")

            try:
                stream_started = False
                is_silent = False
                buffer = ""
                silent_check_done = False
                SILENT_BUFFER_LIMIT = 30

                self._status("waiting_api", f"stream_id={stream_id}")
                async with self.provider.stream(
                    messages=msgs,
                    max_tokens=1024,
                    system=self._system_prompt(),
                    tools=self._tools(),
                ) as stream:
                    async for text in stream.text_chunks():
                        if not silent_check_done:
                            buffer += text
                            if len(buffer) >= SILENT_BUFFER_LIMIT:
                                silent_check_done = True
                                if "<silent>" in buffer.lower():
                                    is_silent = True
                                    break
                                self.send_to_room(
                                    proto.encode(
                                        proto.StreamStartMsg(
                                            stream_id=stream_id,
                                            from_agent=self.agent_name,
                                        )
                                    )
                                )
                                stream_started = True
                                self.send_to_room(
                                    proto.encode(
                                        proto.StreamDeltaMsg(
                                            stream_id=stream_id,
                                            from_agent=self.agent_name,
                                            delta=buffer,
                                        )
                                    )
                                )
                        else:
                            self.send_to_room(
                                proto.encode(
                                    proto.StreamDeltaMsg(
                                        stream_id=stream_id,
                                        from_agent=self.agent_name,
                                        delta=text,
                                    )
                                )
                            )
                    else:
                        if not silent_check_done and "<silent>" in buffer.lower():
                            is_silent = True
                        elif not silent_check_done and buffer:
                            self.send_to_room(
                                proto.encode(
                                    proto.StreamStartMsg(
                                        stream_id=stream_id, from_agent=self.agent_name
                                    )
                                )
                            )
                            stream_started = True
                            self.send_to_room(
                                proto.encode(
                                    proto.StreamDeltaMsg(
                                        stream_id=stream_id,
                                        from_agent=self.agent_name,
                                        delta=buffer,
                                    )
                                )
                            )

                    response = await stream.get_response()

                self._status("done:waiting_api", f"stream_id={stream_id}")
                if is_silent:
                    self._print(
                        f"[SILENT] {self.agent_name} chose not to respond",
                        flush=True,
                    )
                elif stream_started:
                    self.send_to_room(
                        proto.encode(
                            proto.StreamEndMsg(
                                stream_id=stream_id, from_agent=self.agent_name
                            )
                        )
                    )
            except Exception as e:
                self._status("error:waiting_api", f"stream_id={stream_id}")
                self._print(f"[LLM ERROR] {self.agent_name}: {e}", flush=True)
                break

            msgs.append(
                {
                    "role": "assistant",
                    "content": self.provider.wrap_assistant_content(response),
                }
            )

            text_parts = []
            tool_uses = []
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    text_parts.append(block.text.strip())
                elif block.type == "tool_use":
                    tool_uses.append(block)

            has_send_chat_tool = any(t.name == "send_chat" for t in tool_uses)
            if text_parts and not has_send_chat_tool:
                reply = "\n".join(text_parts)
                if "<silent>" not in reply.lower():
                    self.send_chat(reply)

            if not tool_uses:
                break

            tool_results = []
            for tool_block in tool_uses:
                result = await self._execute_tool(tool_block)
                tool_results.append(
                    self.provider.format_tool_result(tool_block.id, result[:2000])
                )
            msgs.append(self.provider.wrap_tool_results(tool_results))

        self._conversation = msgs + self._conversation[snapshot_len:]

    async def _execute_tool(self, tool_block) -> str:
        name = tool_block.name
        inp = tool_block.input

        if name == "send_chat":
            text = inp.get("text", "")
            to_agent = inp.get("to_agent", "")
            self.send_chat(text, to_agent=to_agent)
            return f"Message sent{' to ' + to_agent if to_agent else ' (broadcast)'}"

        if name == "send_connect_request":
            target = inp["target_agent"]
            self._status("doing:send_connect_request", f"target={target}")
            if self._is_connected(target):
                self._status(
                    "done:send_connect_request", f"target={target} already_connected"
                )
                return f"Already connected to {target}"
            msg = proto.ConnectRequestMsg(from_agent=self.agent_name, to_agent=target)
            self.send_to_room(proto.encode(msg))
            self._status("done:send_connect_request", f"target={target}")
            return f"Connection request sent to {target}"

        if name == "request_peer_capability":
            target = inp["target_agent"]
            cap = inp["capability"]
            params = inp.get("params", {})
            label = f"{target}.{cap}"
            self._status(
                "doing:request_peer_capability",
                f"{label} params={self._short(params)}",
            )
            if not self._check_peer_policy(target, cap):
                self._status("error:request_peer_capability", f"{label} policy_denied")
                return f"Policy denied: {self.agent_name} cannot request {target}.{cap}"
            try:
                result = await self.request_capability(target, cap, params)
                self._status("done:request_peer_capability", label)
                return self._scan_peer_result(target, result)
            except asyncio.TimeoutError:
                self._status("error:request_peer_capability", f"{label} timeout")
                return f"Request to {target}.{cap} timed out"

        if name.startswith("own_"):
            cap_name = name[4:]
            self._status("doing:own_capability", cap_name)
            if not self._check_tool_policy(cap_name):
                self._status("error:own_capability", f"{cap_name} policy_denied")
                return f"Policy denied: {self.agent_name} cannot use {cap_name}"
            params = inp.get("params", {})
            cap = self._capabilities.get(cap_name)
            if not cap:
                self._status("error:own_capability", f"{cap_name} unknown")
                return f"Unknown capability: {cap_name}"
            if cap.approval == "human":
                approved = await self.ask_human(
                    f"Use own capability {cap_name}({json.dumps(params)})"
                )
            else:
                approved = True
            if not approved:
                self._status("done:own_capability", f"{cap_name} denied")
                return f"Human denied {cap_name}"
            try:
                if self._sandbox_active():
                    result = await sandbox.exec_capability_in_sandbox(
                        self.agent_name,
                        cap_name,
                        params,
                        api_key=self.provider.api_key_for_sandbox(),
                        provider_name=self.provider.name,
                        model=self.provider.model,
                    )
                else:
                    result = await cap.fn(**params)
                self._status("done:own_capability", cap_name)
                return result
            except Exception as e:
                self._status("error:own_capability", f"{cap_name}: {e}")
                return f"Error in {cap_name}: {e}"

        self._status("error:tool", name)
        return f"Unknown tool: {name}"

    def _sandbox_active(self) -> bool:
        return (
            self.security_mode == "enforce" and self.sandbox_config.get("mode") == "on"
        )

    def _handle_claim_response(self, raw: dict):
        raw_type = raw.get("type")
        if raw_type == "nonce":
            self._print(
                f"[CLAIM] Received code for {raw.get('vanity_name')}: {raw.get('nonce')}"
            )
            if self._claim_event:
                self._claim_nonce = raw.get("nonce", "")
                self._claim_event.set()
        elif raw_type == "name_claimed":
            token = raw.get("auth_token", "")
            name = raw.get("vanity_name", "")
            self._print(f"[CLAIM] Name '{name}' claimed! Auth token saved.")
            save_auth_token(name, token)
            if isinstance(self._transport, WSTransport):
                self._transport.auth_token = token
            if self._claim_event:
                self._claim_event.set()
        elif raw_type == "name_rejected":
            self._print(f"[CLAIM] Rejected: {raw.get('reason')}")
            if self._claim_event:
                self._claim_event.set()

    async def _run_claim_flow(self):
        """Attempt to claim agent_name via tweet verification."""
        self._claim_event = asyncio.Event()
        self._claim_nonce = ""

        req = proto.RequestNonceMsg(agent=self.agent_name, vanity_name=self.agent_name)
        self.send_to_room(proto.encode(req))
        self._print(f"[CLAIM] Requesting code for '{self.agent_name}'...")

        self._claim_event.clear()
        try:
            await asyncio.wait_for(self._claim_event.wait(), timeout=10)
        except asyncio.TimeoutError:
            self._print("[CLAIM] Code request timed out")
            self._claim_event = None
            return

        if not self._claim_nonce:
            self._claim_event = None
            return

        claim = proto.ClaimNameMsg(
            agent=self.agent_name,
            vanity_name=self.agent_name,
            tweet_url=self.tweet_url,
        )
        self._claim_event.clear()
        self.send_to_room(proto.encode(claim))
        self._print(f"[CLAIM] Verifying tweet at {self.tweet_url}...")

        try:
            await asyncio.wait_for(self._claim_event.wait(), timeout=30)
        except asyncio.TimeoutError:
            self._print("[CLAIM] Claim verification timed out")

        self._claim_event = None

    async def _on_reconnect(self):
        """Called by transport after re-establishing a dropped connection."""
        self._replay_done = False
        caps_msg = proto.CapabilitiesMsg(
            agent=self.agent_name,
            owner=self.owner_name,
            capabilities=self.capability_infos(),
            description=self.description,
            public=self.public,
        )
        self.send_to_room(proto.encode(caps_msg))
        self._broadcast_auto_approve_state()
        self._print(f"[ws] {self.agent_name} re-announced capabilities after reconnect")

    async def _seed_relationships(self):
        """Send connect/trust requests for relationships declared in config."""
        for target in self._initial_connections:
            if not self._is_connected(target):
                msg = proto.ConnectRequestMsg(
                    from_agent=self.agent_name, to_agent=target
                )
                self.send_to_room(proto.encode(msg))
                self._print(
                    f"[SEED] {self.agent_name} sent connection request to {target}"
                )
                await asyncio.sleep(0.2)
        for target in self._initial_trusts:
            if not self._is_trusted_by(target):
                msg = proto.TrustRequestMsg(from_agent=self.agent_name, to_agent=target)
                self.send_to_room(proto.encode(msg))
                self._print(f"[SEED] {self.agent_name} sent trust request to {target}")
                await asyncio.sleep(0.2)

    async def run(self):
        if self.provider.has_api_key():
            self._llm_enabled = True
        else:
            key_var = self.provider.api_key_var()
            hint = f" ({key_var} not set)" if key_var else ""
            self._print(f"[WARN] LLM disabled{hint} — chat-only mode")

        if self._sandbox_active():
            try:
                await sandbox.ensure_container(self.agent_name, self.sandbox_config)
            except RuntimeError as e:
                self._print(
                    f"[SANDBOX] Failed to start container for {self.agent_name}: {e}"
                )
                raise SystemExit(
                    f"Cannot start {self.agent_name}: enforce mode requires a working "
                    f"Docker sandbox. Fix Docker or set sandbox.mode to 'off' in agents.yaml.\n"
                    f"  Install Docker:\n"
                    f"    macOS:  brew install --cask docker\n"
                    f"    Ubuntu: sudo apt-get update && sudo apt-get install -y docker.io\n"
                    f"            sudo usermod -aG docker $USER && newgrp docker"
                )

        auth = load_auth_token(self.agent_name)
        if auth and isinstance(self._transport, WSTransport):
            self._transport.auth_token = auth

        self._replay_done = False
        await self._transport.connect(self)
        self._print(f"Connecting {self.agent_name}...")
        await self._ready.wait()
        self._join_ts = time.time_ns()

        if not auth and self.tweet_url:
            await self._run_claim_flow()
        elif not auth:
            self._print(
                f"[WARN] {self.agent_name} running unclaimed (no tweet_url configured)"
            )

        caps_msg = proto.CapabilitiesMsg(
            agent=self.agent_name,
            owner=self.owner_name,
            capabilities=self.capability_infos(),
            description=self.description,
            public=self.public,
        )
        self.send_to_room(proto.encode(caps_msg))
        self._broadcast_auto_approve_state()
        self._print(
            f"Announced capabilities: {', '.join(c.name for c in self.capability_infos())}"
        )
        self._print(f"Listening in {self.room}...\n")

        await self._seed_relationships()

        try:
            await asyncio.Future()
        finally:
            if self._sandbox_active():
                await sandbox.teardown_container(self.agent_name)
