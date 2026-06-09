"""Local secrets file. NOT TRACKED IN GIT. Delete manually after use.

Fill in your keys below, then ``import llm_client.secrets`` (once, anywhere)
before calling any LLM. The import puts the values on ``os.environ`` so any
helper that looks up ``ANTHROPIC_API_KEY`` / ``DNABOT_TOKEN`` finds them.
"""
import os

# ----------------------------------------------------------------------
# Fill in the key you actually use; leave the others empty / delete.
# ----------------------------------------------------------------------
ANTHROPIC_API_KEY: str = ""   # e.g. "sk-ant-..."
OPENAI_API_KEY: str = ""   # e.g. "sk-..."  # e.g. "sk-..."
DNABOT_TOKEN: str = ""   # internal BCG token

# ----------------------------------------------------------------------
# Push them onto the environment (only if non-empty, only if not already set).
# ----------------------------------------------------------------------
for _name, _value in [
    ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
    ("OPENAI_API_KEY", OPENAI_API_KEY),
    ("DNABOT_TOKEN", DNABOT_TOKEN),
]:
    if _value and _name not in os.environ:
        os.environ[_name] = _value
