"""Disable GLM "thinking" in OpenEvolve's OpenAI client.

GLM models reason by default; reasoning tokens count against `max_tokens` and can truncate the
program the Challenger is asked to write.  OpenEvolve builds its request by hand, so the patch
injects `extra_body={"thinking": {"type": "disabled"}}` for models whose name starts with "glm".
Other models are left untouched.
"""
from __future__ import annotations

from openevolve.llm.openai import OpenAILLM

_PATCHED_ATTR = "_glm_thinking_disabled"


def disable_glm_thinking() -> None:
    """Idempotently patch OpenAILLM._call_api."""
    if getattr(OpenAILLM, _PATCHED_ATTR, False):
        return
    _orig_call_api = OpenAILLM._call_api

    async def _call_api(self, params):  # type: ignore[no-untyped-def]
        if str(getattr(self, "model", "")).lower().startswith("glm"):
            extra = dict(params.get("extra_body") or {})
            extra.setdefault("thinking", {"type": "disabled"})
            params = {**params, "extra_body": extra}
        return await _orig_call_api(self, params)

    OpenAILLM._call_api = _call_api
    setattr(OpenAILLM, _PATCHED_ATTR, True)
