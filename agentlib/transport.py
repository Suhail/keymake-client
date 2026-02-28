"""Pluggable transport layer for agent communication."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent


class NickTakenError(Exception):
    def __init__(self, nick: str):
        self.nick = nick
        super().__init__(f'Name "{nick}" is already taken on the hub.')


class NickClaimedError(Exception):
    """Name is claimed by another identity. Tweet verification required."""
    def __init__(self, nick: str):
        self.nick = nick
        super().__init__(f'Name "{nick}" is claimed. Provide a valid auth_token.')


class RateLimitedError(Exception):
    """Server rejected join due to rate limiting."""
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limited. Retry after {retry_after:.0f}s.")


class Transport:
    async def connect(self, agent: Agent):
        raise NotImplementedError

    def send_to_room(self, room: str, body: str):
        raise NotImplementedError

    async def disconnect(self):
        pass


class WSTransport(Transport):
    """Connects to the Python WebSocket hub (server/hub.py)."""

    BACKOFF_BASE = 1
    BACKOFF_MAX = 30
    CB_WINDOW = 120        # circuit breaker: 2-minute lookback window
    CB_THRESHOLD = 3       # circuit breaker: disconnect count to trigger
    CB_FLOOR_INITIAL = 30  # circuit breaker: initial backoff floor (seconds)
    CB_FLOOR_MAX = 120     # circuit breaker: maximum backoff floor (seconds)

    def __init__(self, host: str = "localhost", port: int = 8765, url: str | None = None):
        self._url = url or f"ws://{host}:{port}"
        self._ws = None
        self._agent: Agent | None = None
        self._closing = False
        self._pending_msgs: list[str] = []
        self.session_token: str = ""
        self.auth_token: str = ""
        self._disconnect_times: list[float] = []
        self._circuit_breaker_floor: float = 0.0

    async def connect(self, agent: Agent):
        self._agent = agent
        self._closing = False
        await self._open_and_join()
        asyncio.create_task(self._listen())
        agent._ready.set()

    async def _open_and_join(self):
        from websockets.asyncio.client import connect
        self._ws = await connect(
            self._url,
            open_timeout=30,
            ping_interval=30,
            ping_timeout=120,
        )
        join_msg = {
            "type": "join",
            "room": self._agent.room,
            "nick": self._agent.agent_name,
            "protocol_version": 2,
        }
        if self.session_token:
            join_msg["token"] = self.session_token
        if self.auth_token:
            join_msg["auth_token"] = self.auth_token
        await self._ws.send(json.dumps(join_msg))
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
            msg = json.loads(raw)
            if msg.get("type") == "error":
                if msg.get("code") == "nick_taken":
                    raise NickTakenError(self._agent.agent_name)
                if msg.get("code") == "nick_claimed":
                    raise NickClaimedError(self._agent.agent_name)
                if msg.get("code") == "rate_limited":
                    raise RateLimitedError(msg.get("retry_after", 60))
            if msg.get("type") == "joined":
                self.session_token = msg.get("token", "")
            else:
                self._pending_msgs.append(raw)
        except asyncio.TimeoutError:
            pass

    def send_to_room(self, room: str, body: str):
        if self._ws:
            asyncio.ensure_future(self._safe_send(body))

    async def _safe_send(self, body: str):
        try:
            if self._ws:
                await self._ws.send(json.dumps({"type": "message", "body": body}))
        except Exception:
            pass

    def _check_circuit_breaker(self) -> float:
        """Record a disconnect and return the backoff floor."""
        import time
        now = time.monotonic()
        self._disconnect_times.append(now)
        cutoff = now - self.CB_WINDOW
        self._disconnect_times = [t for t in self._disconnect_times if t > cutoff]
        if len(self._disconnect_times) < self.CB_THRESHOLD:
            if self._circuit_breaker_floor > 0:
                name = self._agent.agent_name if self._agent else "?"
                print(f"[ws] Circuit breaker reset for {name}", flush=True)
            self._circuit_breaker_floor = 0.0
            return 0.0
        if self._circuit_breaker_floor == 0:
            self._circuit_breaker_floor = self.CB_FLOOR_INITIAL
        else:
            self._circuit_breaker_floor = min(
                self._circuit_breaker_floor * 2, self.CB_FLOOR_MAX
            )
        name = self._agent.agent_name if self._agent else "?"
        print(
            f"[ws] Circuit breaker activated for {name}: "
            f"{len(self._disconnect_times)} disconnects in {self.CB_WINDOW}s, "
            f"backoff floor now {self._circuit_breaker_floor}s",
            flush=True,
        )
        return self._circuit_breaker_floor

    async def _listen(self):
        while not self._closing:
            try:
                for raw in self._pending_msgs:
                    msg = json.loads(raw)
                    if msg.get("type") == "replay_batch":
                        continue  # skip batched replay; agent waits for replay_done
                    nick = msg.get("nick", "")
                    body = msg.get("body", "")
                    if body and self._agent:
                        await self._agent._handle_room_message(nick, body)
                self._pending_msgs.clear()
                async for raw in self._ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "replay_batch":
                        continue  # skip batched replay; agent waits for replay_done
                    nick = msg.get("nick", "")
                    body = msg.get("body", "")
                    if body and self._agent:
                        await self._agent._handle_room_message(nick, body)
            except Exception as exc:
                name = self._agent.agent_name if self._agent else "?"
                print(f"[ws] {name} listen error: {exc}", flush=True)

            self._ws = None
            if self._closing:
                break

            name = self._agent.agent_name if self._agent else "?"
            print(f"[ws] Connection lost for {name}, reconnecting...", flush=True)

            floor = self._check_circuit_breaker()
            delay = max(self.BACKOFF_BASE, floor)
            while not self._closing:
                try:
                    await asyncio.sleep(delay)
                    await self._open_and_join()
                    print(f"[ws] {name} reconnected", flush=True)
                    if self._agent:
                        await self._agent._on_reconnect()
                    break
                except (NickTakenError, NickClaimedError):
                    raise
                except RateLimitedError as exc:
                    delay = max(exc.retry_after, delay)
                    print(f"[ws] {name} rate limited by server, waiting {delay:.0f}s", flush=True)
                except Exception as exc:
                    delay = min(delay * 2, self.BACKOFF_MAX)
                    delay = max(delay, floor)
                    print(f"[ws] {name} reconnect failed ({exc}), retrying in {delay:.0f}s", flush=True)

    async def disconnect(self):
        self._closing = True
        if self._ws:
            await self._ws.close()


class XMPPTransport(Transport):
    """Connects to an XMPP server (e.g. ejabberd) via slixmpp."""

    def __init__(self, server: str = "localhost", port: int = 5222):
        self._server = server
        self._port = port
        self._xmpp = None

    async def connect(self, agent: Agent):
        import slixmpp

        class _XMPPClient(slixmpp.ClientXMPP):
            def __init__(inner_self, jid: str, password: str):
                super().__init__(jid, password)
                inner_self.register_plugin("xep_0030")
                inner_self.register_plugin("xep_0045")
                inner_self.register_plugin("xep_0077")
                inner_self["feature_mechanisms"].unencrypted_plain = True
                inner_self["feature_mechanisms"].unencrypted_scram = True
                inner_self.add_event_handler("session_start", inner_self.on_start)
                inner_self.add_event_handler("groupchat_message", inner_self.on_muc_message)

            async def on_start(inner_self, event):
                inner_self.send_presence()
                await inner_self.get_roster()
                try:
                    await asyncio.wait_for(
                        inner_self["xep_0045"].join_muc_wait(agent.room, agent.agent_name),
                        timeout=10,
                    )
                except (asyncio.TimeoutError, slixmpp.exceptions.PresenceError) as e:
                    print(f"[WARN] MUC join issue for {agent.agent_name}: {e}")
                agent._ready.set()

            async def on_muc_message(inner_self, msg):
                if not msg["body"] or msg["mucnick"] == agent.agent_name:
                    return
                await agent._handle_room_message(msg["mucnick"], msg["body"])

        jid = f"{agent.agent_name}@{self._server}"
        password = agent.password
        self._xmpp = _XMPPClient(jid, password)
        self._xmpp.connect((self._server, self._port))

    def send_to_room(self, room: str, body: str):
        if self._xmpp:
            self._xmpp.make_message(mto=room, mbody=body, mtype="groupchat").send()

    async def disconnect(self):
        if self._xmpp:
            self._xmpp.disconnect()
