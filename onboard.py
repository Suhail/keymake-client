#!/usr/bin/env python3
"""Interactive onboarding wizard for Agent Chat client.

Sets up LLM provider, API keys, agent souls, and agents.yaml.
All paths are relative to this script's directory.
"""

import argparse
import json
import os
import random
import shutil
import sys
import threading
import time
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE / ".env"
YAML_PATH = HERE / "agents.yaml"
SOULS_DIR = HERE / "souls"

from agentlib.llm import DEFAULT_MODELS

PRODUCTION_HUB = "wss://api.keymake.ai"

# ── Color palette ─────────────────────────────────────────────────────

_NO_COLOR = os.environ.get("NO_COLOR") or not sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if _NO_COLOR:
        return text
    return f"{code}{text}\033[0m"


def accent(t):    return _c("\033[38;5;202m", t)
def success(t):   return _c("\033[38;5;78m", t)
def warn(t):      return _c("\033[38;5;214m", t)
def error(t):     return _c("\033[38;5;167m", t)
def muted(t):     return _c("\033[38;5;245m", t)
def bold(t):      return _c("\033[1m", t)

BAR_CHAR = "│"
S_BAR = muted(BAR_CHAR)


# ── Clack-style primitives ───────────────────────────────────────────

def _term_width() -> int:
    return min(shutil.get_terminal_size((80, 24)).columns, 90)


def intro(title: str):
    w = _term_width()
    bar = accent("┌") + accent("─" * (w - 2)) + accent("┐")
    pad = (w - 4 - len(title)) // 2
    line = accent("│") + " " * max(pad, 1) + bold(title) + " " * max(w - 4 - pad - len(title), 1) + accent("│")
    bot = accent("└") + accent("─" * (w - 2)) + accent("┘")
    print(f"\n{bar}\n{line}\n{bot}\n")


def outro(message: str):
    w = _term_width()
    bar = success("─") * w
    print(f"\n{bar}")
    print(f"  {success('●')} {bold(message)}")
    print(f"{bar}\n")


def note(message: str, title: str | None = None):
    w = _term_width() - 4
    lines = []
    for raw_line in message.split("\n"):
        if not raw_line.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(raw_line, width=w, break_on_hyphens=False, break_long_words=False) or [""])
    top = muted("╭─") + (f" {accent(title)} " + muted("─" * max(0, w - len(title) - 3)) if title else muted("─" * w)) + muted("─╮")
    bot = muted("╰") + muted("─" * (w + 2)) + muted("╯")
    print(top)
    for ln in lines:
        padding = w - len(ln)
        print(f"{S_BAR}  {ln}{' ' * max(padding, 0)} {S_BAR}")
    print(bot)
    print()


def select(message: str, options: list[tuple[str, str, str]]) -> str:
    print(f"  {accent('◆')} {bold(message)}")
    for i, (value, label, hint) in enumerate(options, 1):
        hint_str = f"  {muted(hint)}" if hint else ""
        marker = accent(f"  {i}.")
        print(f"  {S_BAR} {marker} {label}{hint_str}")
    while True:
        choice = input(f"  {S_BAR} {muted(f'Enter [1-{len(options)}]')}: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                picked = options[idx]
                print(f"  {S_BAR} {success('✓')} {picked[1]}")
                print(f"  {S_BAR}")
                return picked[0]
        except ValueError:
            pass
        print(f"  {S_BAR} {error('Invalid choice.')}")


def prompt_text(message: str, default: str | None = None, password: bool = False) -> str:
    suffix = f" {muted(f'({default})')}" if default else ""
    prompt_str = f"  {accent('◆')} {bold(message)}{suffix}\n  {S_BAR} "
    if password:
        import getpass
        val = getpass.getpass(prompt_str)
    else:
        val = input(prompt_str)
    val = val.strip()
    result = val or default or ""
    print(f"  {S_BAR}")
    return result


def confirm(message: str, default: bool = True) -> bool:
    hint = muted("(Y/n)") if default else muted("(y/N)")
    val = input(f"  {accent('◆')} {bold(message)} {hint} ").strip().lower()
    result = (val in ("y", "yes")) if val else default
    status = success("Yes") if result else muted("No")
    print(f"  {S_BAR} {status}")
    print(f"  {S_BAR}")
    return result


class Spinner:
    _FRAMES = ["◒", "◐", "◓", "◑"]

    def __init__(self, label: str):
        self._label = label
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self):
        i = 0
        while self._running:
            frame = self._FRAMES[i % len(self._FRAMES)]
            sys.stdout.write(f"\r  {accent(frame)} {self._label}")
            sys.stdout.flush()
            time.sleep(0.12)
            i += 1

    def update(self, label: str):
        self._label = label

    def stop(self, message: str | None = None):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        msg = message or self._label
        sys.stdout.write(f"\r  {success('✓')} {msg}          \n")
        sys.stdout.flush()


# ── Capabilities catalog ─────────────────────────────────────────────

AVAILABLE_CAPS = {
    "web_search":       ("Search the web",           "trust",   "human"),
    "summarize_text":   ("Summarize text",            "public",  "auto"),
    "generate_code":    ("Generate Python code",      "trust",   "human"),
    "schedule_meeting": ("Schedule a meeting",        "connect", "human"),
}

_NAME_ADJECTIVES = [
    "swift", "bright", "cosmic", "gentle", "bold", "vivid", "calm",
    "clever", "daring", "eager", "fancy", "happy", "jolly", "keen",
    "lively", "merry", "noble", "plucky", "quiet", "ready", "sunny",
    "witty", "zesty", "agile", "brave", "crisp", "deft", "fair",
    "grand", "wise",
]

_NAME_NOUNS = [
    "raccoon", "penguin", "falcon", "otter", "panda", "fox", "owl",
    "dolphin", "hawk", "lynx", "robin", "wolf", "bear", "crane",
    "finch", "heron", "koala", "lemur", "moose", "newt", "quail",
    "raven", "seal", "tiger", "viper", "wren", "zebra", "badger",
    "coyote", "elk",
]


def _random_agent_name() -> str:
    return f"{random.choice(_NAME_ADJECTIVES)}-{random.choice(_NAME_NOUNS)}"


DEFAULT_AGENTS = [
    {
        "name": "alice", "owner": "Alice",
        "soul_file": "alice.md",
        "description": "A research-focused agent specializing in web search and text summarization.",
        "style": "Gen-Z speak. Chill.",
        "caps": ["web_search", "summarize_text"],
    },
    {
        "name": "bob", "owner": "Bob",
        "soul_file": "bob.md",
        "description": "A developer agent that writes code and manages schedules.",
        "style": "Direct. Concise. Succinct.",
        "caps": ["generate_code", "schedule_meeting"],
    },
]

DEFAULT_PEER_DENY = ["*.exec", "*.eval", "*.shell*", "*.post_*", "*.rm_*"]


# ── Wizard helpers ───────────────────────────────────────────────────

def _risk_acknowledgement():
    note(
        "Agent Chat runs AI agents that can:\n"
        "  • Execute tool calls (web search, code generation)\n"
        "  • Communicate with other agents autonomously\n"
        "  • Take actions requiring human approval\n"
        "\n"
        "Recommended baseline:\n"
        "  • Use 'audit' security mode (logs policy violations)\n"
        "  • Keep API keys out of agent-reachable paths\n"
        "  • Review tool policies in agents.yaml",
        "Security",
    )
    if not confirm("I understand the risks and want to continue"):
        outro("Setup cancelled.")
        sys.exit(0)


def _select_provider_grouped() -> str:
    groups = [
        ("anthropic", "Anthropic", "Claude models — API key"),
        ("openai", "OpenAI", "GPT models — API key"),
        ("ollama", "Ollama", "Local open-source models — no key needed"),
        ("custom", "Custom endpoint", "Any OpenAI-compatible API"),
    ]

    print(f"  {accent('◆')} {bold('Model provider')}")
    for i, (_, label, hint) in enumerate(groups, 1):
        print(f"  {S_BAR} {accent(f'  {i}.')} {label}  {muted(hint)}")
    print(f"  {S_BAR} {accent(f'  {len(groups) + 1}.')} {muted('Skip for now')}")

    while True:
        choice = input(f"  {S_BAR} {muted(f'Enter [1-{len(groups) + 1}]')}: ").strip()
        try:
            idx = int(choice) - 1
            if idx == len(groups):
                print(f"  {S_BAR} {success('✓')} Skipped")
                print(f"  {S_BAR}")
                return "skip"
            if 0 <= idx < len(groups):
                print(f"  {S_BAR} {success('✓')} {groups[idx][1]}")
                print(f"  {S_BAR}")
                return groups[idx][0]
        except ValueError:
            pass
        print(f"  {S_BAR} {error('Invalid choice.')}")


def _validate_api_key(provider_name: str, key: str) -> bool:
    spin = Spinner(f"Validating {provider_name} API key…").start()
    try:
        if provider_name == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            client.models.list(limit=1)
        elif provider_name == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=key)
            client.models.list()
        spin.stop(f"{provider_name} API key validated.")
        return True
    except Exception as e:
        spin.stop(f"Validation failed: {e}")
        return False


def _prompt_api_key(provider_name: str) -> str | None:
    placeholder = "sk-ant-..." if provider_name == "anthropic" else "sk-..."
    key = prompt_text(f"{provider_name.capitalize()} API key ({placeholder})")
    if not key:
        note("No key provided. You can set the env var later.", "Skipped")
        return None
    if _validate_api_key(provider_name, key):
        return key
    if confirm("Key validation failed. Use it anyway?", default=False):
        return key
    return None


def _list_ollama_models(base_url: str) -> list[str]:
    try:
        import urllib.request
        api_url = base_url.rstrip("/").replace("/v1", "") + "/api/tags"
        with urllib.request.urlopen(urllib.request.Request(api_url, method="GET"), timeout=3) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _prompt_ollama() -> tuple[str, str]:
    url = prompt_text("Ollama base URL", "http://localhost:11434")
    spin = Spinner("Checking Ollama…").start()
    models = _list_ollama_models(url)
    if models:
        spin.stop(f"Ollama running — {len(models)} model(s) available.")
        options = [(m, m, "") for m in models[:10]]
        model = select("Pick a model", options)
        return url, model
    spin.stop("Could not reach Ollama.")
    note(f"Make sure Ollama is running at {url}.\nYou can start it later.", "Ollama")
    model = prompt_text("Model name", "llama3.2")
    return url, model


def _prompt_custom() -> tuple[str, str, str]:
    url = prompt_text("Base URL (OpenAI-compatible)")
    key = prompt_text("API key (leave empty if none)", "")
    model = prompt_text("Model ID")
    return url, key, model


def _select_model(provider_name: str) -> str:
    models = {
        "anthropic": [
            ("claude-sonnet-4-6", "Claude Sonnet 4.6", "balanced speed and quality — recommended"),
            ("claude-haiku-4-20250414", "Claude Haiku 4", "fastest, cheapest"),
        ],
        "openai": [
            ("gpt-5.2", "GPT-5.2", "most capable"),
            ("gpt-5.2-mini", "GPT-5.2 Mini", "fast, cheap"),
            ("o3-mini", "o3-mini", "reasoning"),
        ],
    }
    options = models.get(provider_name)
    if not options:
        return prompt_text("Model name")
    return select("Default model", options)


def _select_security_mode() -> str:
    return select("Security mode", [
        ("audit", "Audit", "log policy violations — recommended"),
        ("enforce", "Enforce", "block violations + Docker sandbox"),
        ("off", "Off", "no security checks"),
    ])


# ── Soul / agent generation ──────────────────────────────────────────

def _generate_soul(name: str, owner: str, description: str, style: str,
                   caps: list[str], peer_names: list[str]) -> str:
    cap_names = ", ".join(caps)
    lines = [
        f"You are {owner}'s assistant. You have {cap_names} capabilities.",
        "",
        "## Description",
        description,
        "",
        "## Communication style",
        style,
        "",
        "## Coordination",
    ]
    other_caps = [c for c in AVAILABLE_CAPS if c not in caps]
    for c in other_caps:
        if peer_names:
            desc = c.replace("_", " ")
            lines.append(f"- If you need {desc}, request {c} from another agent.")
    lines.append("- Use your own tools when possible.")
    if peer_names:
        lines.append(f"- Coordinate with {', '.join(peer_names)} as needed.")
    lines.append("- When you finish a task, do not announce completion — just stop responding.")
    return "\n".join(lines) + "\n"


def _ensure_default_souls():
    SOULS_DIR.mkdir(parents=True, exist_ok=True)
    created = []
    for agent in DEFAULT_AGENTS:
        path = SOULS_DIR / agent["soul_file"]
        if not path.exists():
            peers = [a["name"] for a in DEFAULT_AGENTS if a["name"] != agent["name"]]
            content = _generate_soul(
                agent["name"], agent["owner"], agent["description"],
                agent["style"], agent["caps"], peers,
            )
            path.write_text(content)
            created.append(agent["name"])
    return created


def _prompt_starter_agent(existing_names: list[str]) -> dict | None:
    note(
        "Each agent needs a soul — a markdown file defining its\n"
        "personality, description, and coordination rules.\n"
        "Let's create your agent.",
        "Create an agent",
    )

    name = prompt_text("Agent name (lowercase, no spaces)", "alice")
    name = name.lower().replace(" ", "_")

    if name in existing_names:
        soul_path = SOULS_DIR / f"{name}.md"
        if soul_path.exists():
            note(f"Soul file already exists: souls/{name}.md", name)
            return None

    owner = prompt_text("Owner name (display name)", name.capitalize())
    description = prompt_text("Short description", "A helpful AI assistant.")
    style = prompt_text("Communication style", "Concise and direct.")

    print(f"  {accent('◆')} {bold('Capabilities')}")
    cap_options = []
    for cap_name, (desc, tier, _) in AVAILABLE_CAPS.items():
        cap_options.append((cap_name, f"{cap_name}", f"{desc} [tier={tier}]"))
    for i, (_, label, hint) in enumerate(cap_options, 1):
        print(f"  {S_BAR} {accent(f'  {i}.')} {label}  {muted(hint)}")
    print(f"  {S_BAR}")
    raw = prompt_text("Pick capabilities (comma-separated numbers)", "1,2")
    chosen_caps = []
    for part in raw.split(","):
        part = part.strip()
        try:
            idx = int(part) - 1
            if 0 <= idx < len(cap_options):
                chosen_caps.append(cap_options[idx][0])
        except ValueError:
            if part in AVAILABLE_CAPS:
                chosen_caps.append(part)
    if not chosen_caps:
        chosen_caps = ["web_search", "summarize_text"]
        note("No valid selection — defaulting to web_search, summarize_text.", "Capabilities")

    personality = select("Agent personality", [
        ("public", "Public", "responds to greetings and casual chat from anyone"),
        ("locked", "Locked down", "only responds to connected agents and capability requests"),
    ])
    is_public = personality == "public"

    peers = [n for n in existing_names if n != name]
    soul_content = _generate_soul(name, owner, description, style, chosen_caps, peers)

    SOULS_DIR.mkdir(parents=True, exist_ok=True)
    (SOULS_DIR / f"{name}.md").write_text(soul_content)
    note(soul_content, f"souls/{name}.md")

    return {"name": name, "owner": owner, "soul_file": f"{name}.md", "caps": chosen_caps, "public": is_public}


# ── Config writers ───────────────────────────────────────────────────

def _build_agents_list(custom_agents: list[dict]) -> list[dict]:
    agents = []
    all_names = [a["name"] for a in custom_agents]
    for agent in custom_agents:
        entry = {
            "name": agent["name"],
            "owner": agent["owner"],
            "soul": f"souls/{agent['soul_file']}",
            "public": agent.get("public", False),
            "capabilities": [],
            "tools": {"peer_deny": DEFAULT_PEER_DENY},
        }
        for cap_name in agent["caps"]:
            _, tier, approval = AVAILABLE_CAPS[cap_name]
            entry["capabilities"].append({"name": cap_name, "tier": tier, "approval": "human"})
        peers = [n for n in all_names if n != agent["name"]]
        if peers:
            entry["connections"] = peers
        agents.append(entry)
    return agents


def _read_env() -> dict:
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def _write_env(values: dict):
    existing = _read_env()
    existing.update(values)
    lines = [f"{k}={v}" for k, v in existing.items() if v]
    ENV_PATH.write_text("\n".join(lines) + "\n")


def _write_yaml(provider_name, model, security_mode, base_url=None, agents_list=None):
    import yaml
    cfg = yaml.safe_load(YAML_PATH.read_text()) if YAML_PATH.exists() else {}
    cfg["provider"] = provider_name
    cfg["model"] = model
    cfg["ws_url"] = PRODUCTION_HUB
    cfg["security_mode"] = security_mode
    if base_url:
        cfg["base_url"] = base_url
    elif "base_url" in cfg:
        del cfg["base_url"]
    if agents_list is not None:
        cfg["agents"] = agents_list
    YAML_PATH.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


# ── Main flows ───────────────────────────────────────────────────────

def run_interactive():
    intro("Agent Chat — Client Setup")
    _risk_acknowledgement()

    has_env = ENV_PATH.exists()
    if has_env:
        if not confirm("Existing .env detected. Overwrite?"):
            outro("Keeping existing configuration.")
            return

    flow = select("Setup mode", [
        ("quick", "QuickStart", "sensible defaults, just enter your API key"),
        ("manual", "Manual", "choose provider, model, security mode"),
    ])

    env_values = {}
    provider_name = "anthropic"
    model = "claude-sonnet-4-6"
    security_mode = "audit"
    base_url = None
    custom_agents = None

    if flow == "quick":
        provider_name = select("Provider", [
            ("anthropic", "Anthropic", "Claude — recommended"),
            ("openai", "OpenAI", "GPT models"),
            ("ollama", "Ollama", "local models, no key needed"),
        ])
        model = DEFAULT_MODELS.get(provider_name, "claude-sonnet-4-6")

        if provider_name == "ollama":
            base_url, model = _prompt_ollama()
            base_url = base_url.rstrip("/") + "/v1"
            env_values["OLLAMA_BASE_URL"] = base_url
        else:
            key = _prompt_api_key(provider_name)
            if key:
                env_key = "ANTHROPIC_API_KEY" if provider_name == "anthropic" else "OPENAI_API_KEY"
                env_values[env_key] = key

        note(
            f"Provider:  {provider_name} ({model})\n"
            "Security:  Audit mode\n"
            f"Hub:       {PRODUCTION_HUB}",
            "QuickStart",
        )

        name = prompt_text("Name your agent", _random_agent_name()).lower().replace(" ", "-")
        owner = name.replace("-", " ").title()
        caps = ["web_search", "summarize_text"]
        soul_file = f"{name}.md"

        personality = select("Agent personality", [
            ("public", "Public", "responds to greetings and casual chat from anyone"),
            ("locked", "Locked down", "only responds to connected agents and capability requests"),
        ])
        is_public = personality == "public"

        SOULS_DIR.mkdir(parents=True, exist_ok=True)
        soul_content = _generate_soul(
            name, owner,
            "A versatile assistant with research and summarization skills.",
            "Friendly and helpful.", caps, [],
        )
        (SOULS_DIR / soul_file).write_text(soul_content)
        note(f"Created soul: souls/{soul_file}", "Agent")

        custom_agents = _build_agents_list([
            {"name": name, "owner": owner, "soul_file": soul_file, "caps": caps, "public": is_public}
        ])
    else:
        provider_name = _select_provider_grouped()

        if provider_name == "skip":
            note("Set environment variables manually before launching.", "Skipped")
            _write_yaml("anthropic", "claude-sonnet-4-6", "audit")
            outro("Setup complete.")
            return

        if provider_name == "anthropic":
            key = _prompt_api_key("anthropic")
            if key:
                env_values["ANTHROPIC_API_KEY"] = key
            model = _select_model("anthropic")
        elif provider_name == "openai":
            key = _prompt_api_key("openai")
            if key:
                env_values["OPENAI_API_KEY"] = key
            model = _select_model("openai")
        elif provider_name == "ollama":
            base_url, model = _prompt_ollama()
            base_url = base_url.rstrip("/") + "/v1"
            env_values["OLLAMA_BASE_URL"] = base_url
        elif provider_name == "custom":
            base_url, key, model = _prompt_custom()
            if key:
                env_values["OPENAI_API_KEY"] = key

        security_mode = _select_security_mode()

        agent_choice = select("Starter agents", [
            ("defaults", "Use defaults", "alice (research) + bob (coding)"),
            ("create", "Create a new agent", "custom name, personality, capabilities"),
            ("skip", "Skip", "keep existing agents.yaml entries"),
        ])

        if agent_choice == "defaults":
            created = _ensure_default_souls()
            if created:
                note(f"Created soul(s): {', '.join(created)}", "Agents")
            custom_agents = _build_agents_list([
                {"name": a["name"], "owner": a["owner"], "soul_file": a["soul_file"], "caps": a["caps"]}
                for a in DEFAULT_AGENTS
            ])
        elif agent_choice == "create":
            existing_names = []
            agents_created = []
            first = _prompt_starter_agent(existing_names)
            if first:
                existing_names.append(first["name"])
                agents_created.append(first)
            while confirm("Add another agent?", default=False):
                extra = _prompt_starter_agent(existing_names)
                if extra:
                    existing_names.append(extra["name"])
                    agents_created.append(extra)
            if agents_created:
                custom_agents = _build_agents_list(agents_created)

    env_values["LLM_PROVIDER"] = provider_name

    spin = Spinner("Writing configuration…").start()
    _write_env(env_values)
    _write_yaml(provider_name, model, security_mode, base_url, custom_agents)
    time.sleep(0.3)
    spin.stop("Configuration saved.")

    note(
        f"Provider:  {provider_name}\n"
        f"Model:     {model}\n"
        f"Security:  {security_mode}\n"
        f"Hub:       {PRODUCTION_HUB}\n"
        f"Config:    agents.yaml\n"
        f"Env:       .env",
        "Configuration",
    )

    hub_http = PRODUCTION_HUB.replace("wss://", "https://").replace("ws://", "http://")
    note(
        "To start your agents:\n\n"
        "  python run.py\n\n"
        f"Hub status:  {hub_http}\n"
        f"Browser chat: a personalized link is printed when your agent connects.\n\n"
        "Back up your .env — it contains your API keys.\n"
        "Review tool policies in agents.yaml before production use.",
        "Next steps",
    )

    outro("Setup complete. Happy hacking!")


def run_non_interactive(args):
    provider_name = args.provider or os.environ.get("LLM_PROVIDER", "anthropic")
    api_key = args.api_key or ""
    model = args.model or DEFAULT_MODELS.get(provider_name, "claude-sonnet-4-6")
    security_mode = args.security_mode or "audit"
    base_url = args.base_url

    env_values = {"LLM_PROVIDER": provider_name}
    if provider_name == "anthropic" and api_key:
        env_values["ANTHROPIC_API_KEY"] = api_key
    elif provider_name == "openai" and api_key:
        env_values["OPENAI_API_KEY"] = api_key
    elif provider_name == "ollama":
        env_values["OLLAMA_BASE_URL"] = base_url or "http://localhost:11434/v1"

    _write_env(env_values)

    _ensure_default_souls()
    agents_list = _build_agents_list([
        {"name": a["name"], "owner": a["owner"], "soul_file": a["soul_file"], "caps": a["caps"]}
        for a in DEFAULT_AGENTS
    ])
    _write_yaml(provider_name, model, security_mode, base_url, agents_list)
    print(f"Configured: provider={provider_name} model={model} security={security_mode}")


def main():
    parser = argparse.ArgumentParser(description="Agent Chat client setup")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--provider", choices=["anthropic", "openai", "ollama", "custom"])
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--security-mode", choices=["off", "audit", "enforce"])
    args = parser.parse_args()

    if args.non_interactive:
        run_non_interactive(args)
    else:
        try:
            run_interactive()
        except (KeyboardInterrupt, EOFError):
            print()
            outro("Setup cancelled.")
            sys.exit(1)


if __name__ == "__main__":
    main()
