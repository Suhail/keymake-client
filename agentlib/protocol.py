"""JSON message protocol for agent-to-agent communication."""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


def make_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class CapabilityInfo:
    name: str
    desc: str
    tier: str = "trust"
    approval: str = "human"


@dataclass
class CapabilitiesMsg:
    agent: str
    owner: str
    capabilities: list[CapabilityInfo]
    description: str = ""
    public: bool = False
    type: str = "capabilities"


@dataclass
class RequestMsg:
    capability: str
    params: dict[str, Any]
    from_agent: str
    id: str = field(default_factory=make_id)
    type: str = "request"


@dataclass
class ResponseMsg:
    id: str
    status: str  # "completed" | "denied" | "error"
    result: str
    from_agent: str
    type: str = "response"


@dataclass
class AcceptMsg:
    id: str
    from_agent: str
    type: str = "accept"


@dataclass
class ThinkingMsg:
    from_agent: str
    id: str = field(default_factory=make_id)
    type: str = "thinking"


@dataclass
class StreamStartMsg:
    stream_id: str
    from_agent: str
    type: str = "stream_start"


@dataclass
class StreamDeltaMsg:
    stream_id: str
    from_agent: str
    delta: str
    type: str = "stream_delta"


@dataclass
class StreamEndMsg:
    stream_id: str
    from_agent: str
    type: str = "stream_end"


@dataclass
class DeliveryMsg:
    from_agent: str
    to_agent: str
    title: str
    content: str
    id: str = field(default_factory=make_id)
    type: str = "delivery"


@dataclass
class ChatMsg:
    from_agent: str
    text: str
    to_agent: str = ""
    type: str = "chat"


# -- Social graph messages --
# Hub is authoritative for social state. Clients send social intents and
# consume hub graph_state snapshots; they no longer sync local social state.

@dataclass
class ConnectRequestMsg:
    from_agent: str
    to_agent: str
    type: str = "connect_request"


@dataclass
class ConnectAcceptMsg:
    from_agent: str
    to_agent: str
    type: str = "connect_accept"


@dataclass
class ConnectRejectMsg:
    from_agent: str
    to_agent: str
    type: str = "connect_reject"


@dataclass
class TrustRequestMsg:
    from_agent: str
    to_agent: str
    type: str = "trust_request"


@dataclass
class TrustAcceptMsg:
    from_agent: str
    to_agent: str
    type: str = "trust_accept"


@dataclass
class TrustRejectMsg:
    from_agent: str
    to_agent: str
    reason: str = ""
    type: str = "trust_reject"


@dataclass
class RequestNonceMsg:
    agent: str
    vanity_name: str
    type: str = "request_nonce"


@dataclass
class NonceMsg:
    agent: str
    vanity_name: str
    nonce: str
    tweet_text: str
    type: str = "nonce"


@dataclass
class ClaimNameMsg:
    agent: str
    vanity_name: str
    tweet_url: str
    type: str = "claim_name"


@dataclass
class NameClaimedMsg:
    agent: str
    vanity_name: str
    auth_token: str
    type: str = "name_claimed"


@dataclass
class NameRejectedMsg:
    agent: str
    vanity_name: str
    reason: str
    type: str = "name_rejected"


def encode(msg) -> str:
    d = asdict(msg)
    d["ts"] = time.time_ns()
    return json.dumps(d)


Msg = (
    CapabilitiesMsg | RequestMsg | ResponseMsg | AcceptMsg
    | ThinkingMsg | StreamStartMsg | StreamDeltaMsg | StreamEndMsg
    | DeliveryMsg | ChatMsg
    | ConnectRequestMsg | ConnectAcceptMsg | ConnectRejectMsg
    | TrustRequestMsg | TrustAcceptMsg | TrustRejectMsg
    | RequestNonceMsg | NonceMsg | ClaimNameMsg | NameClaimedMsg | NameRejectedMsg
)

def _stamp(msg: Msg, d: dict) -> Msg:
    msg.ts = d.get("ts", 0)
    return msg


def decode(raw: str) -> Msg | None:
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    t = d.get("type")
    if t == "capabilities":
        return _stamp(CapabilitiesMsg(
            agent=d["agent"],
            owner=d["owner"],
            capabilities=[
                CapabilityInfo(
                    name=c["name"], desc=c.get("desc", ""),
                    tier=c.get("tier", "trust"), approval=c.get("approval", "human"),
                )
                for c in d["capabilities"]
            ],
            description=d.get("description", ""),
        ), d)
    if t == "request":
        return _stamp(RequestMsg(
            id=d["id"],
            capability=d["capability"],
            params=d["params"],
            from_agent=d["from_agent"],
        ), d)
    if t == "response":
        return _stamp(ResponseMsg(
            id=d["id"],
            status=d["status"],
            result=d["result"],
            from_agent=d["from_agent"],
        ), d)
    if t == "accept":
        return _stamp(AcceptMsg(id=d["id"], from_agent=d["from_agent"]), d)
    if t == "thinking":
        return _stamp(ThinkingMsg(id=d["id"], from_agent=d["from_agent"]), d)
    if t == "stream_start":
        return _stamp(StreamStartMsg(stream_id=d["stream_id"], from_agent=d["from_agent"]), d)
    if t == "stream_delta":
        return _stamp(StreamDeltaMsg(stream_id=d["stream_id"], from_agent=d["from_agent"], delta=d["delta"]), d)
    if t == "stream_end":
        return _stamp(StreamEndMsg(stream_id=d["stream_id"], from_agent=d["from_agent"]), d)
    if t == "delivery":
        return _stamp(DeliveryMsg(
            id=d["id"],
            from_agent=d["from_agent"],
            to_agent=d["to_agent"],
            title=d["title"],
            content=d["content"],
        ), d)
    if t == "chat":
        return _stamp(ChatMsg(from_agent=d["from_agent"], text=d["text"], to_agent=d.get("to_agent", "")), d)
    if t == "connect_request":
        return _stamp(
            ConnectRequestMsg(from_agent=d["from_agent"], to_agent=d["to_agent"]), d
        )
    if t == "connect_accept":
        return _stamp(
            ConnectAcceptMsg(from_agent=d["from_agent"], to_agent=d["to_agent"]), d
        )
    if t == "connect_reject":
        return _stamp(
            ConnectRejectMsg(from_agent=d["from_agent"], to_agent=d["to_agent"]), d
        )
    if t == "trust_request":
        return _stamp(
            TrustRequestMsg(from_agent=d["from_agent"], to_agent=d["to_agent"]), d
        )
    if t == "trust_accept":
        return _stamp(
            TrustAcceptMsg(from_agent=d["from_agent"], to_agent=d["to_agent"]), d
        )
    if t == "trust_reject":
        return _stamp(
            TrustRejectMsg(
                from_agent=d["from_agent"],
                to_agent=d["to_agent"],
                reason=d.get("reason", ""),
            ),
            d,
        )
    if t == "request_nonce":
        return _stamp(RequestNonceMsg(agent=d["agent"], vanity_name=d["vanity_name"]), d)
    if t == "nonce":
        return _stamp(NonceMsg(agent=d["agent"], vanity_name=d["vanity_name"], nonce=d["nonce"], tweet_text=d.get("tweet_text", "")), d)
    if t == "claim_name":
        return _stamp(ClaimNameMsg(agent=d["agent"], vanity_name=d["vanity_name"], tweet_url=d["tweet_url"]), d)
    if t == "name_claimed":
        return _stamp(NameClaimedMsg(agent=d["agent"], vanity_name=d["vanity_name"], auth_token=d["auth_token"]), d)
    if t == "name_rejected":
        return _stamp(NameRejectedMsg(agent=d["agent"], vanity_name=d["vanity_name"], reason=d["reason"]), d)
    return None
