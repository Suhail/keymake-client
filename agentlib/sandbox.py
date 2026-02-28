"""Docker sandbox: container lifecycle and exec routing for agent isolation."""

import asyncio
import json
import os
import re
import shutil
from pathlib import Path

DEFAULT_IMAGE = "agentchat-sandbox:latest"
CONTAINER_PREFIX = "agentchat-sbx"
CLIENT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = CLIENT_DIR


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def image_exists(image: str = DEFAULT_IMAGE) -> bool:
    """Check if a Docker image exists locally."""
    if not _docker_available():
        return False
    import subprocess
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def build_image(image: str = DEFAULT_IMAGE) -> tuple[bool, str]:
    """Build the sandbox Docker image. Returns (success, output_or_error).

    Auto-detects whether we're in the monorepo (client/ is a subdirectory) or
    the standalone keymake-client repo (client/ contents are the repo root) and
    adjusts the build context and skills path accordingly.
    """
    if not _docker_available():
        return False, "Docker is not installed or not in PATH."
    import subprocess

    parent_dir = CLIENT_DIR.parent
    is_monorepo = (parent_dir / "client").is_dir()

    if is_monorepo:
        build_context = parent_dir
        skills_path = "client/skills"
    else:
        build_context = CLIENT_DIR
        skills_path = "skills"

    dockerfile = CLIENT_DIR / "Dockerfile.sandbox"
    try:
        result = subprocess.run(
            ["docker", "build", "-f", str(dockerfile), "-t", image,
             "--build-arg", f"SKILLS_PATH={skills_path}", "."],
            cwd=str(build_context),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Docker build timed out after 5 minutes."
    except Exception as e:
        return False, str(e)


def _container_name(agent_name: str) -> str:
    return f"{CONTAINER_PREFIX}-{agent_name}"


def _sanitize_error(err: str) -> str:
    """Strip host-system details from error messages before returning to agents."""
    # Redact absolute host paths (keep only basename)
    err = re.sub(r'/(?:Users|home|root)/\S+', '<host-path>', err)
    # Redact API keys that might appear in error output
    err = re.sub(r'sk-ant-api\w{2}-[\w-]{20,}', '[REDACTED]', err)
    err = re.sub(r'sk-[a-zA-Z0-9]{32,}', '[REDACTED]', err)
    # Truncate to avoid overly long errors
    if len(err) > 500:
        err = err[:497] + "..."
    return err


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


async def _run(cmd: list[str], check: bool = True) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if check and proc.returncode != 0:
        # Only show the command name (not args which may contain paths/secrets)
        cmd_name = cmd[0] if cmd else "unknown"
        err_text = _sanitize_error(stderr.decode())
        raise RuntimeError(f"{cmd_name} failed (exit {proc.returncode}): {err_text}")
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
        "--memory=512m",
        "--cpus=1",
    ]

    # Always mount client code read-only so capabilities can be imported
    cmd += ["-v", f"{CLIENT_DIR}:/app:ro"]

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


async def exec_cli_streaming(
    agent_name: str,
    command: str,
    env_vars: dict[str, str] | None = None,
    on_stdout_line: "Callable[[str], None] | None" = None,
    timeout: int = 120,
) -> str:
    """Run a CLI command in the sandbox, streaming stdout lines for progress.

    Each stdout line is passed to on_stdout_line (if provided) and also
    collected. Returns the full stdout output. Designed for CLI tools like
    Claude Code and Codex that produce streaming JSON output.

    Env vars are passed via `docker exec -e` flags to avoid leaking secrets
    in the process command line.
    """
    name = _container_name(agent_name)
    if not await container_running(agent_name):
        raise RuntimeError(f"Sandbox container {name} is not running")

    # Build docker exec command with -e flags for env vars (not shell prefix)
    docker_cmd = ["docker", "exec"]
    for k, v in (env_vars or {}).items():
        docker_cmd += ["-e", f"{k}={v}"]
    docker_cmd += [name, "bash", "-c", command]

    proc = await asyncio.create_subprocess_exec(
        *docker_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    lines: list[str] = []
    assert proc.stdout is not None

    try:
        async with asyncio.timeout(timeout):
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode()
                lines.append(text)
                if on_stdout_line and text.strip():
                    on_stdout_line(text.strip())

            await proc.wait()
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return "Error: command timed out"

    return "".join(lines).strip()


async def exec_capability_in_sandbox(
    agent_name: str,
    cap_name: str,
    params: dict,
    api_key: str | None = None,
    provider_name: str = "anthropic",
    model: str | None = None,
) -> str:
    """Import and run a capability function inside the sandbox container.

    API keys are passed via docker exec -e flags to avoid leaking them
    in shell command strings or process listings.
    """
    name = _container_name(agent_name)
    if not await container_running(agent_name):
        raise RuntimeError(f"Sandbox container {name} is not running")

    params_json = json.dumps(params)
    script = (
        "import asyncio, json, sys, os\n"
        "sys.path.insert(0, '/app')\n"
        f"params = json.loads({params_json!r})\n"
        "from agentlib.llm import make_provider\n"
        "from capabilities import CAPABILITY_REGISTRY, set_provider\n"
        f"provider = make_provider({provider_name!r}, model={model!r})\n"
        "set_provider(provider)\n"
        f"entry = CAPABILITY_REGISTRY.get({cap_name!r})\n"
        "if not entry:\n"
        f"    print('Unknown capability: {cap_name}')\n"
        "    sys.exit(1)\n"
        "desc, fn, _sbx = entry\n"
        "result = asyncio.run(fn(**params))\n"
        "print(result)\n"
    )

    # Pass secrets via -e flags instead of shell command string
    docker_cmd = ["docker", "exec"]
    if api_key:
        docker_cmd += ["-e", f"ANTHROPIC_API_KEY={api_key}"]
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        docker_cmd += ["-e", f"OPENAI_API_KEY={openai_key}"]
    docker_cmd += [name, "bash", "-c", f"python3 -c {_shell_quote(script)}"]

    proc = await asyncio.create_subprocess_exec(
        *docker_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        # Sanitize error output — don't leak host paths or env details
        err_msg = stderr.decode().strip()
        err_msg = _sanitize_error(err_msg)
        raise RuntimeError(f"Sandbox execution failed: {err_msg}")
    return stdout.decode().strip()


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
