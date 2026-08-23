"""
Structured answer schema.

The compose node returns raw LLM text (hopefully JSON). This module parses it
into a validated Pydantic model, which the API then serializes cleanly to the
frontend. No more relying on the frontend to parse a potentially-truncated
JSON string out of an SSE stream.
"""
from __future__ import annotations

import json
import re
from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    title: str = ""
    excerpt: str = ""
    authority: int = 0
    status: str = "current"


class StructuredAnswer(BaseModel):
    verdict: str = ""
    reasoning: str = ""
    confidence: str = "medium"
    conflicts: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    suggested_action: Optional[str] = None


def parse_answer(raw: str) -> StructuredAnswer:
    """
    Parse the LLM's raw compose output into a StructuredAnswer.

    Tries multiple strategies:
    1. Direct JSON parse
    2. Extract JSON from markdown fences
    3. Find first { to last } and parse
    4. Regex extraction of individual fields
    5. Return the raw text as the verdict (graceful degradation)
    """
    if not raw or not raw.strip():
        return StructuredAnswer(verdict="No answer generated.", confidence="low")

    text = raw.strip()

    # Strategy 1: direct parse
    try:
        data = json.loads(text)
        return StructuredAnswer(**data)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Strategy 2: strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        return StructuredAnswer(**data)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Strategy 3: find braces
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last > first:
        candidate = cleaned[first:last + 1]
        try:
            data = json.loads(candidate)
            return StructuredAnswer(**data)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Strategy 3b: the JSON might be truncated — try to repair by closing
        # open strings and braces
        repaired = _repair_truncated_json(candidate)
        if repaired:
            try:
                data = json.loads(repaired)
                return StructuredAnswer(**data)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    # Strategy 4: regex extraction of key fields
    verdict = _extract_field(text, "verdict")
    reasoning = _extract_field(text, "reasoning")
    confidence = _extract_field(text, "confidence") or "medium"
    if verdict:
        return StructuredAnswer(
            verdict=verdict,
            reasoning=reasoning or "",
            confidence=confidence,
        )

    # Strategy 5: treat the whole thing as the verdict
    # Truncate to something reasonable
    if len(text) > 2000:
        text = text[:2000] + "..."
    return StructuredAnswer(verdict=text, confidence="medium")


def _extract_field(text: str, field_name: str) -> Optional[str]:
    """Extract a JSON string field value by regex."""
    # Match "field_name": "value" (handling escaped quotes inside)
    pattern = rf'"{field_name}"\s*:\s*"((?:[^"\\]|\\.)*)?"'
    m = re.search(pattern, text, re.DOTALL)
    if m:
        val = m.group(1) or ""
        # Unescape
        val = val.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
        return val
    return None


def _repair_truncated_json(text: str) -> Optional[str]:
    """
    Attempt to close a truncated JSON object so it can be parsed.
    Only handles the common case: a string value that got cut off.
    """
    # Count unmatched braces/brackets
    opens = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")

    if opens <= 0 and open_brackets <= 0:
        return None  # Not truncated in a way we can fix

    # Check if we're inside an unclosed string (odd number of unescaped quotes)
    in_string = False
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            i += 2
            continue
        if text[i] == '"':
            in_string = not in_string
        i += 1

    result = text
    if in_string:
        result += '"'

    # Close arrays then objects
    result += "]" * max(0, open_brackets)
    result += "}" * max(0, opens)

    return result
