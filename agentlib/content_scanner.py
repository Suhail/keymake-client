"""Scan agent-to-agent content for dangerous patterns and credential leaks."""

import re

DANGEROUS_CODE_PATTERNS = [
    (r'\bos\.system\s*\(', "os.system() call"),
    (r'\bsubprocess\.\w+\s*\(', "subprocess call"),
    (r'\bshutil\.rmtree\s*\(', "recursive directory deletion"),
    (r'\beval\s*\(', "eval() call"),
    (r'\bexec\s*\(', "exec() call"),
    (r'rm\s+-rf?\s+/', "rm -rf on root"),
    (r'__import__\s*\(', "dynamic import"),
    (r'\bopen\s*\([^)]*["\']\/(?:etc|proc|sys)', "sensitive file access"),
    (r'\bos\.remove\s*\(', "file deletion"),
    (r'\bos\.rmdir\s*\(', "directory deletion"),
    (r'\bos\.environ\b', "environment variable access"),
]

CREDENTIAL_PATTERNS = [
    (r'sk-ant-api\w{2}-[\w-]{20,}', "Anthropic API key"),
    (r'sk-[a-zA-Z0-9]{32,}', "OpenAI API key"),
    (r'ghp_[a-zA-Z0-9]{36}', "GitHub PAT"),
    (r'gho_[a-zA-Z0-9]{36}', "GitHub OAuth token"),
    (r'xox[bporas]-[\w-]{10,}', "Slack token"),
    (r'(?i)bearer\s+[a-zA-Z0-9._~+/=-]{20,}', "Bearer token"),
    (r'AKIA[0-9A-Z]{16}', "AWS access key"),
]


def _scan(content: str, patterns: list[tuple[str, str]]) -> list[str]:
    return [desc for pattern, desc in patterns if re.search(pattern, content)]


def scan_content(content: str) -> list[str]:
    return _scan(content, DANGEROUS_CODE_PATTERNS)


def scan_credentials(content: str) -> list[str]:
    return _scan(content, CREDENTIAL_PATTERNS)


def redact_credentials(content: str) -> str:
    """Replace detected credential values with redacted placeholders."""
    result = content
    for pattern, desc in CREDENTIAL_PATTERNS:
        result = re.sub(pattern, f"[REDACTED {desc}]", result)
    return result
