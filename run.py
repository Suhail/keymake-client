"""Config-driven multi-agent runner."""

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VENV_PYTHON = _HERE / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if sys.prefix == sys.base_prefix and _VENV_PYTHON.exists():
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import asyncio

import yaml
from dotenv import load_dotenv
load_dotenv()

from agentlib.agent import Agent, DEFAULT_RATE_LIMIT
from agentlib.llm import make_provider
from agentlib.transport import WSTransport, XMPPTransport, NickTakenError, NickClaimedError
from capabilities import CAPABILITY_REGISTRY, set_provider


def load_config(path=None):
    path = path or Path(__file__).parent / "agents.yaml"
    return yaml.safe_load(Path(path).read_text())


def make_transport(config):
    if config.get("transport", "ws") == "xmpp":
        return XMPPTransport(
            server=config.get("xmpp_server", "localhost"),
            port=config.get("xmpp_port", 5222),
        )
    ws_url = os.environ.get("WS_URL") or config.get("ws_url")
    return WSTransport(
        host=config.get("ws_host", "localhost"),
        port=config.get("ws_port", 8765),
        url=ws_url,
    )


def load_soul(soul_path: str) -> str:
    return (Path(__file__).parent / soul_path).read_text()


def _parse_cap(entry) -> tuple[str, str, str]:
    """Parse a capability entry (string or dict) into (name, tier, approval)."""
    if isinstance(entry, str):
        return entry, "trust", "human"
    return entry["name"], entry.get("tier", "trust"), entry.get("approval", "human")


def make_provider_from_config(config):
    provider_name = config.get("provider") or os.environ.get("LLM_PROVIDER", "anthropic")
    model = config.get("model")
    auto_approve_model = config.get("auto_approve_model")
    base_url = config.get("base_url") or config.get("ollama_base_url")
    return make_provider(
        provider_name=provider_name,
        model=model,
        auto_approve_model=auto_approve_model,
        base_url=base_url,
    )


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run agents from agents.yaml")
    parser.add_argument("--agent", action="append", dest="agents",
                        help="Run only the named agent(s). Can be repeated. Default: all.")
    parser.add_argument("--config", help="Path to agents.yaml")
    return parser.parse_args()


async def main():
    args = _parse_args()
    config = load_config(args.config)
    room = config.get("room", "general")
    default_rate_limit = config.get("rate_limit", DEFAULT_RATE_LIMIT)
    security_mode = config.get("security_mode", "off")
    agent_filter = set(args.agents) if args.agents else None

    if security_mode not in ("off", "audit", "enforce"):
        print(f"[WARN] Unknown security_mode '{security_mode}', defaulting to 'off'")
        security_mode = "off"

    if security_mode != "off":
        print(f"[SECURITY] mode={security_mode}")

    provider = make_provider_from_config(config)
    set_provider(provider)
    print(f"[LLM] provider={provider.name} model={provider.model}")

    all_names = [a["name"] for a in config["agents"]]
    if agent_filter:
        unknown = agent_filter - set(all_names)
        if unknown:
            print(f"[ERROR] Unknown agent(s): {', '.join(unknown)}")
            print(f"[ERROR] Available: {', '.join(all_names)}")
            return

    agents = []
    for agent_cfg in config["agents"]:
        if agent_filter and agent_cfg["name"] not in agent_filter:
            continue
        transport = make_transport(config)
        context = load_soul(agent_cfg["soul"]) if "soul" in agent_cfg else agent_cfg.get("context", "")
        rate_limit = agent_cfg.get("rate_limit", default_rate_limit)
        agent = Agent(
            agent_name=agent_cfg["name"],
            owner_name=agent_cfg["owner"],
            room=room,
            context=context,
            transport=transport,
            rate_limit=rate_limit,
            security_mode=security_mode,
            tool_policy=agent_cfg.get("tools"),
            sandbox_config=agent_cfg.get("sandbox"),
            initial_connections=agent_cfg.get("connections"),
            initial_trusts=agent_cfg.get("trusts"),
            provider=provider,
            tweet_url=agent_cfg.get("tweet_url", ""),
            public=agent_cfg.get("public", False),
        )
        for cap_entry in agent_cfg.get("capabilities", []):
            cap_name, tier, approval = _parse_cap(cap_entry)
            if cap_name not in CAPABILITY_REGISTRY:
                print(f"[WARN] Unknown capability: {cap_name}")
                continue
            desc, fn = CAPABILITY_REGISTRY[cap_name]
            agent.register_capability(cap_name, desc, fn, tier=tier, approval=approval)
        agents.append(agent)

    ws_url = config.get("ws_url") or os.environ.get("WS_URL") or ""

    def _chat_url(agent):
        transport = agent._transport
        token = getattr(transport, "session_token", "")
        if not token or not ws_url:
            return None
        if os.environ.get("FRONTEND_URL"):
            frontend = os.environ["FRONTEND_URL"]
        elif "keymake.ai" in ws_url:
            frontend = "https://keymake.ai"
        else:
            frontend = "http://localhost:3000"
        return f"{frontend}/chat?agent={agent.agent_name}&token={token}"

    async def _run_agent(agent):
        try:
            await agent.run()
        except NickTakenError:
            print(f'[ERROR] Name "{agent.agent_name}" is already taken on the hub.')
            print(f'        Edit agents.yaml and change the agent name, then re-run.')
        except NickClaimedError:
            print(f'[ERROR] Name "{agent.agent_name}" is claimed. Provide a valid auth_token or claim via /claim page.')

    async def _print_chat_url(agent):
        await agent._ready.wait()
        url = _chat_url(agent)
        if url:
            agent.chat_url = url
            R = "\033[0m"
            BL = "\033[38;5;27m"   # brand blue
            GR = "\033[38;5;48m"   # brand green
            B = "\033[1m"
            M = "\033[38;5;245m"
            bar = f"{BL}{'━' * 60}{R}"
            print(f"\n{bar}")
            print(f"  {GR}{B}◆ Web UI for {agent.agent_name}{R}")
            print(f"  {BL}{B}{url}{R}")
            print(f"  {M}Open this URL to chat, interact with other agents,")
            print(f"  or approve requests for your agent.{R}")
            print(f"  {M}To run again later: cd {Path(__file__).resolve().parent} && source .venv/bin/activate && python run.py{R}")
            print(f"{bar}\n")

    tasks = []
    for agent in agents:
        tasks.append(asyncio.create_task(_run_agent(agent)))
        asyncio.create_task(_print_chat_url(agent))
        await asyncio.sleep(2)

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
