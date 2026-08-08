"""Persistent cognitive agent for the Unitree G1.

This package extends the existing prototypes under ``g1/modules/scripts``
(``llm_client`` for the OpenAI/Anthropic tool-calling transport,
``ollama_ai/scene_executor.py`` for the deterministic skill-dispatch
pattern, ``ollama_ai/nav_bot.py`` for local-knowledge retrieval,
``ollama_ai/chatbot_with_tactile_dex3.py`` for the gesture scheduler, and
``g1_approval_ros/policy.py`` for the safety/approval policy shape) into a
single, typed, persistent agent loop rather than adding a second parallel
architecture.

See ``g1/modules/scripts/agent/README.md`` for the module map and the
explicit list of what is a real integration versus a documented stub in
this phase.
"""

from __future__ import annotations

__version__ = "0.1.0"
