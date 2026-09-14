"""Provider-abstracted LLM client for the optional LLM-based NER / RE methods.

A single ``generate(prompt) -> (text, model_name)`` entry point so the backend and
model can be swapped/upgraded later via ``config.LLM_BACKEND`` / ``config.LLM_MODEL``.
Only the Gemini backend is wired now; other names raise ``NotImplementedError``.

The API key is read from the environment (never from config): ``GEMINI_API_KEY``
(preferred) or ``GOOGLE_API_KEY``. With no key, or on any error, ``generate`` returns
``("", "unavailable")`` so callers degrade to a no-op instead of crashing — the LLM
methods are opt-in and the rest of the pipeline must run without a key.

Transport failures (rate limit, timeout, 5xx, empty/safety-filtered response) are
retried ``config.LLM_MAX_RETRIES`` times with a growing wait. Whatever still fails is
counted: ``stats()`` reports calls / retries / failed since the last ``reset_stats()``,
so a caller can say how much of its input never reached the model.
"""

import os
import time

# importing config loads .env (see config.py)
from config import LLM_BACKEND, LLM_MODEL, LLM_MAX_RETRIES

_UNAVAILABLE = ("", "unavailable")
_warned = False

# Call tally so a caller can tell a complete run from a partially-lost one. A failed
# call means a whole batch of records produced nothing, and that is invisible in the
# output CSV — it just has fewer rows.
_stats = {"calls": 0, "retries": 0, "failed": 0}


def stats() -> dict:
    """Snapshot of the call tally since the last reset_stats()."""
    return dict(_stats)


def reset_stats() -> None:
    """Zero the tally; callers do this at the start of their run() to report per-method."""
    for k in _stats:
        _stats[k] = 0


def _warn_once(msg: str) -> None:
    global _warned
    if not _warned:
        print(msg)
        _warned = True


def _generate_gemini(prompt: str) -> tuple[str, str]:
    """Call Google Gemini. Mirrors the client mechanics used by the legacy decoder."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        _warn_once("[LLM] No GEMINI_API_KEY / GOOGLE_API_KEY set - LLM methods are a no-op.")
        return _UNAVAILABLE

    model_name = os.environ.get("GENAI_MODEL") or LLM_MODEL
    _stats["calls"] += 1

    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            from google import genai

            try:
                client = genai.Client(api_key=api_key)
            except TypeError:  # older SDK reads the key from the environment instead
                os.environ.setdefault("GOOGLE_API_KEY", api_key)
                client = genai.Client()

            # Interactions API (replaces the deprecated models.generate_content):
            # `input` replaces `contents`, `.output_text` joins the returned text blocks.
            # See https://ai.google.dev/gemini-api/docs/migrate-to-interactions
            interaction = client.interactions.create(model=model_name, input=prompt)
            text = getattr(interaction, "output_text", None)
            if not text:
                raise RuntimeError("empty response from Gemini")
            return text.strip(), model_name
        except Exception as exc:
            if attempt < LLM_MAX_RETRIES:
                _stats["retries"] += 1
                time.sleep(2 ** (attempt + 1))  # 2s, 4s, ... — mainly for rate limits
                continue
            _stats["failed"] += 1
            _warn_once(f"[LLM] Gemini call failed ({type(exc).__name__}: {exc}) "
                       f"after {LLM_MAX_RETRIES} retries - that batch is lost.")
            return _UNAVAILABLE

    return _UNAVAILABLE  # unreachable unless LLM_MAX_RETRIES < 0


_BACKENDS = {"gemini": _generate_gemini}


def generate(prompt: str) -> tuple[str, str]:
    """Return ``(text, model_name)``; ``("", "unavailable")`` when the LLM can't be used."""
    backend = _BACKENDS.get(LLM_BACKEND)
    if backend is None:
        raise NotImplementedError(f"LLM_BACKEND={LLM_BACKEND!r} not implemented")
    return backend(prompt)
