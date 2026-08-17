"""Pluggable reasoning backend for the agent's natural-language decision log.

Important scope note, because this is the one part of the system that is
easy to over-claim: the LLM is used ONLY to produce a human-readable
justification string for (a) which asset/source the scheduler decided to
poll next, and (b) how a detected conflict was resolved. It never decides
WHICH VALUE WINS a conflict -- that's resolver.py, which is pure deterministic
comparison logic so the "correct regardless of arrival order" guarantee
doesn't depend on a model's sampling behaviour. The LLM explains decisions
that were already made deterministically; it doesn't make them.

Two backends:
  - OllamaBackend: calls a local Ollama server's HTTP API directly
    (http://localhost:11434/api/generate). No SDK dependency -- just httpx.
  - HeuristicBackend: template-based, zero dependencies, always available.

`build_reasoner()` probes Ollama once at startup with a short timeout and
picks whichever is available, so the whole system runs identically whether
or not Ollama is installed -- only the phrasing of the decision log changes.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:1b"


class ReasoningBackend(Protocol):
    name: str

    async def explain_scheduling_choice(self, *, chosen: str, candidates: list[dict]) -> str: ...

    async def explain_conflict(self, *, asset_id: str, field: str, winner: dict, loser: dict, reason: str) -> str: ...


class HeuristicBackend:
    """Deterministic, dependency-free fallback. Always correct, never fancy."""

    name = "heuristic"

    async def explain_scheduling_choice(self, *, chosen: str, candidates: list[dict]) -> str:
        top = max(candidates, key=lambda c: c["score"])
        others = len(candidates) - 1
        return (
            f"Polling {top['asset_id']} via {top['source_name']}: staleness "
            f"{top['staleness_s']:.1f}s x priority {top['priority']} gave it the "
            f"highest score ({top['score']:.2f}) among {others} other candidate(s)."
        )

    async def explain_conflict(self, *, asset_id: str, field: str, winner: dict, loser: dict, reason: str) -> str:
        return (
            f"{asset_id}.{field}: kept '{winner['value']}' from {winner['source']} "
            f"over '{loser['value']}' from {loser['source']} -- {reason}."
        )


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, timeout: float = 45.0):
        self.model = model
        self.timeout = timeout

    async def _generate(self, prompt: str, fallback: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False,
                          "options": {"num_predict": 80}},
                )
                resp.raise_for_status()
                text = resp.json().get("response", "").strip()
                return text or fallback
        except Exception as exc:  # network error, model not pulled, server down, etc.
            logger.warning("Ollama call failed (%s: %s); using heuristic text instead",
                            type(exc).__name__, exc)
            return fallback

    async def explain_scheduling_choice(self, *, chosen: str, candidates: list[dict]) -> str:
        fallback = await HeuristicBackend().explain_scheduling_choice(chosen=chosen, candidates=candidates)
        prompt = (
            "You are a fleet operations agent deciding which asset to check next. "
            "In one short sentence, explain why you are polling this one now.\n"
            f"Candidates (asset, source, staleness_seconds, priority, score): {candidates}\n"
            f"You chose: {chosen}\n"
            "One sentence, plain language, no preamble:"
        )
        return await self._generate(prompt, fallback)

    async def explain_conflict(self, *, asset_id: str, field: str, winner: dict, loser: dict, reason: str) -> str:
        fallback = await HeuristicBackend().explain_conflict(
            asset_id=asset_id, field=field, winner=winner, loser=loser, reason=reason
        )
        prompt = (
            "You are a fleet operations agent that just resolved a conflicting update. "
            "In one short sentence, explain which value you kept and why.\n"
            f"Asset: {asset_id}, field: {field}\n"
            f"Kept: {winner}\nDiscarded: {loser}\nMechanical reason: {reason}\n"
            "One sentence, plain language, no preamble:"
        )
        return await self._generate(prompt, fallback)


async def build_reasoner(model: str = DEFAULT_MODEL) -> ReasoningBackend:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            resp.raise_for_status()
            tags = [m["name"] for m in resp.json().get("models", [])]
            if not any(model in t or t in model for t in tags):
                logger.warning(
                    "Ollama is running but model '%s' is not pulled (have: %s); "
                    "using heuristic backend. Run: ollama pull %s", model, tags, model,
                )
                return HeuristicBackend()
        logger.info("Ollama detected with model '%s' -- using it for decision-log reasoning", model)
        return OllamaBackend(model=model)
    except Exception:
        logger.info("Ollama not reachable at %s -- using heuristic reasoning backend", OLLAMA_URL)
        return HeuristicBackend()
