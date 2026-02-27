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

    def __init__(self, host: str = "localhost", port: int = 8765, url: str | None = None):
        self._url = url or f"ws://{host}:{port}"
        self._ws = None
        self._agent: Agent | None = None
        self._closing = False
        self._pending_msgs: list[str] = []
        self.session_token: str = ""
        self.auth_token: str = ""

    async def connect(self, agent: Agent):
        self._agent = agent
        self._closing = False
        await self._open_and_join()
        asyncio.create_task(self._listen())
        agent._ready.set()

    async def _open_and_join(self):
        from websockets.asyncio.client import connect
        self._ws = await connect(self._url, ping_interval=20, ping_timeout=20)
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
            raw = await asyncio.wait_for(self._ws.recv(), timeout=2)
            msg = json.loads(raw)
            if msg.get("type") == "error":
                if msg.get("code") == "nick_taken":
                    raise NickTakenError(self._agent.agent_name)
                if msg.get("code") == "nick_claimed":
                    raise NickClaimedError(self._agent.agent_name)
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
            except Exception:
                pass

            self._ws = None
            if self._closing:
                break

            name = self._agent.agent_name if self._agent else "?"
            print(f"[ws] Connection lost for {name}, reconnecting...", flush=True)

            delay = self.BACKOFF_BASE
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
                except Exception as exc:
                    print(f"[ws] {name} reconnect failed ({exc}), retrying in {min(delay * 2, self.BACKOFF_MAX)}s", flush=True)
                    delay = min(delay * 2, self.BACKOFF_MAX)

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
