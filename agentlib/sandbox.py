"""Docker sandbox: container lifecycle and exec routing for agent isolation."""

import asyncio
import json
import os
import shutil
from pathlib import Path

DEFAULT_IMAGE = "agentchat-sandbox:latest"
CONTAINER_PREFIX = "agentchat-sbx"
WORKSPACE_DIR = Path(__file__).resolve().parent.parent


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _container_name(agent_name: str) -> str:
    return f"{CONTAINER_PREFIX}-{agent_name}"


async def _run(cmd: list[str], check: bool = True) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{stderr.decode()}")
    return stdout.decode().strip()


async def container_running(agent_name: str) -> bool:
    name = _container_name(agent_name)
    out = await _run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name], check=False,
    )
    return out == "true"


async def ensure_container(agent_name: str, config: dict) -> str:
    """Create and start a sandbox container if not already running. Returns container name."""
    if not _docker_available():
        raise RuntimeError("Docker not found. Install Docker or set sandbox.mode to 'off'.")

    name = _container_name(agent_name)

    if await container_running(agent_name):
        return name

    # Remove stale container if it exists but isn't running
    await _run(["docker", "rm", "-f", name], check=False)

    image = config.get("image", DEFAULT_IMAGE)
    workspace_access = config.get("workspace_access", "none")

    cmd = [
        "docker", "create",
        "--name", name,
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
    ]

    if workspace_access == "rw":
        cmd += ["-v", f"{WORKSPACE_DIR}:/workspace:rw"]
    elif workspace_access == "ro":
        cmd += ["-v", f"{WORKSPACE_DIR}:/workspace:ro"]

    env_vars = config.get("env", {})
    for k, v in env_vars.items():
        expanded = os.path.expandvars(v) if v.startswith("$") else v
        cmd += ["-e", f"{k}={expanded}"]

    cmd.append(image)

    await _run(cmd)
    await _run(["docker", "start", name])
    print(f"[SANDBOX] Started container {name} (image={image}, workspace={workspace_access})")
    return name


async def exec_in_sandbox(agent_name: str, command: str) -> str:
    """Run a command inside the agent's sandbox container."""
    name = _container_name(agent_name)
    if not await container_running(agent_name):
        raise RuntimeError(f"Sandbox container {name} is not running")
    return await _run(["docker", "exec", name, "bash", "-c", command])


async def exec_capability_in_sandbox(
    agent_name: str, cap_name: str, params: dict, api_key: str | None = None,
) -> str:
    """Serialize a capability call as a Python script and run it in the sandbox."""
    params_json = json.dumps(params)
    script = (
        "import json, sys\n"
        f"params = json.loads({params_json!r})\n"
        f"print(f'[SANDBOX] Running {cap_name} with {{params}}')\n"
        f"print(json.dumps({{'status': 'ok', 'capability': {cap_name!r}, 'params': params}}))\n"
    )
    env_cmd = ""
    if api_key:
        env_cmd = f"ANTHROPIC_API_KEY={api_key} "
    return await exec_in_sandbox(agent_name, f"{env_cmd}python3 -c {_shell_quote(script)}")


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


async def teardown_container(agent_name: str):
    """Stop and remove the agent's sandbox container."""
    name = _container_name(agent_name)
    await _run(["docker", "rm", "-f", name], check=False)
    print(f"[SANDBOX] Removed container {name}")


async def teardown_all():
    """Remove all agentchat sandbox containers."""
    out = await _run(
        ["docker", "ps", "-a", "--filter", f"name={CONTAINER_PREFIX}", "--format", "{{.Names}}"],
        check=False,
    )
    for name in out.splitlines():
        if name.strip():
            await _run(["docker", "rm", "-f", name.strip()], check=False)
            print(f"[SANDBOX] Removed container {name.strip()}")
