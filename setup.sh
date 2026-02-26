#!/usr/bin/env bash
set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────────
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  ORANGE='\033[38;5;202m' GREEN='\033[38;5;78m' RED='\033[38;5;167m'
  GRAY='\033[38;5;245m'   BOLD='\033[1m'        RESET='\033[0m'
else
  ORANGE='' GREEN='' RED='' GRAY='' BOLD='' RESET=''
fi

step() { printf "\n${ORANGE}▸${RESET} ${BOLD}%s${RESET}\n" "$1"; }
ok()   { printf "  ${GREEN}✓${RESET} %s\n" "$1"; }
fail() { printf "  ${RED}✗${RESET} %s\n" "$1"; exit 1; }
info() { printf "  ${GRAY}%s${RESET}\n" "$1"; }

# ── Locate script directory ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

printf "\n${ORANGE}┌──────────────────────────────────────────────────┐${RESET}\n"
printf "${ORANGE}│${RESET}${BOLD}          Agent Chat — Client Setup               ${RESET}${ORANGE}│${RESET}\n"
printf "${ORANGE}└──────────────────────────────────────────────────┘${RESET}\n\n"

# ── 1. Check Python ──────────────────────────────────────────────────
step "Checking Python"

PYTHON=""
for cmd in python3 python; do
  if command -v "$cmd" &>/dev/null; then
    ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    major=$("$cmd" -c 'import sys; print(sys.version_info.major)')
    minor=$("$cmd" -c 'import sys; print(sys.version_info.minor)')
    if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
      PYTHON="$cmd"
      ok "Found $cmd $ver"
      break
    else
      info "$cmd $ver found but need >= 3.11"
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  fail "Python >= 3.11 required. Install from https://python.org"
fi

# ── 2. Create venv ───────────────────────────────────────────────────
step "Setting up virtual environment"

if [ ! -d ".venv" ]; then
  "$PYTHON" -m venv .venv
  ok "Created .venv"
else
  ok ".venv already exists"
fi

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate
ok "Activated .venv"

# ── 3. Install dependencies ──────────────────────────────────────────
step "Installing dependencies"

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "Dependencies installed"

# ── 4. Run onboarding ────────────────────────────────────────────────
step "Running setup wizard"

python onboard.py "$@"
