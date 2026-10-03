"""Thin LLM adapter. Providers: offline (default, no network) | anthropic | openai (any OpenAI-compatible endpoint).

The LLM only does two things: (1) propose risky spans with a severity (semantic analysis) and
(2) polish replacement text for spans that Python already selected. Scoring, decision and constraints stay in aotgoat.py.
"""
import json
import os
import re

import requests

ANALYZE_SYS = """You are a linguistic risk annotator for workplace messages. Return ONLY JSON, no prose."""
ANALYZE_USER = """Find spans in the MESSAGE that a recipient could misread. Allowed types:
coreference (pronoun without clear antecedent), temporal (relative/vague time), scope (who/what is included),
terminology (jargon/acronym), implicit (depends on unstated shared context).
`span` MUST be copied verbatim from the message. `severity` is 0-1 (intrinsic ambiguity, ignoring who reads it).
Return: {{"risks":[{{"type":"...","span":"...","reason":"<=15 words","severity":0.0}}]}}
MESSAGE: {message}"""
REWRITE_SYS = """You minimally edit messages. Return ONLY JSON, no prose."""
REWRITE_USER = """For each edit, return the text that should REPLACE its span in the message.
Rules: keep the span's own words (except for a pronoun, which is replaced by the evidence), add ONLY information from
`evidence`/`proposal`, change nothing else, keep it short, never add facts, never change meaning or polarity.
Return: {{"replacements":{{"<id>":"<replacement text>"}}}}
MESSAGE: {message}
EDITS: {edits}"""


def _json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


class LLM:
    def __init__(self, provider=None):
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "offline").strip().lower()
        default = "claude-sonnet-5-5" if self.provider == "anthropic" else "gpt-4o-mini"
        self.model = os.getenv("LLM_MODEL") or default
        self.timeout = float(os.getenv("LLM_TIMEOUT") or 30)

    @property
    def remote(self):
        return self.provider in ("anthropic", "openai")

    def _chat(self, system, user):
        if self.provider == "anthropic":
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=self.timeout,
                              headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                              json={"model": self.model, "max_tokens": 1000, "system": system,
                                    "messages": [{"role": "user", "content": user}]})
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        base = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        r = requests.post(f"{base}/chat/completions", timeout=self.timeout, headers={"Authorization": f"Bearer {key}"},
                          json={"model": self.model, "temperature": 0,
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def analyze(self, message):
        data = _json(self._chat(ANALYZE_SYS, ANALYZE_USER.format(message=message)))
        return [r for r in data.get("risks", []) if isinstance(r, dict)]

    def rewrite(self, message, edits):
        data = _json(self._chat(REWRITE_SYS, REWRITE_USER.format(message=message, edits=json.dumps(edits, ensure_ascii=False))))
        out = {}
        for k, v in (data.get("replacements") or {}).items():
            try:
                out[int(k)] = str(v).strip()
            except ValueError:
                pass
        return out
