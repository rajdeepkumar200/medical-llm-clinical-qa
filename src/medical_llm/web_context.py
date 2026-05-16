"""Lightweight web grounding for short / ambiguous medical queries.

Uses the public Wikipedia REST + OpenSearch APIs (no auth, no API key).
We:
  1. Run an OpenSearch query to pick the best-matching page title.
  2. Fetch that page's summary extract.
  3. Return a trimmed text snippet plus the resolved page title.

The caller can inject the snippet into the LLM prompt to ground the answer,
and surface the title in a "Did you mean ...?" clarifier.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Tuple
from urllib import parse, request

logger = logging.getLogger(__name__)

_UA = "ClinicalAIAssistant/1.0 (educational use; contact: hf-spaces)"
_TIMEOUT_S = 4.0
_MAX_EXTRACT_CHARS = 1200


def _http_get_json(url: str) -> Optional[dict]:
    try:
        req = request.Request(url, headers={"User-Agent": _UA})
        with request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:  # network errors, JSON errors, anything
        logger.info(f"web_context fetch failed for {url}: {e}")
        return None


def _clean_query(q: str) -> str:
    return (q or "").strip().strip("?.,!").strip()


def fetch_wiki_context(query: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(extract, title)`` for the best Wikipedia match of *query*.

    Returns ``(None, None)`` if nothing useful was found or the network call
    failed. Network is best-effort and bounded by a short timeout so a slow
    Wikipedia response never blocks generation for long.
    """
    q = _clean_query(query)
    if not q:
        return None, None

    # Step 1: OpenSearch picks the best matching page title.
    search_url = (
        "https://en.wikipedia.org/w/api.php?"
        + parse.urlencode(
            {
                "action": "opensearch",
                "search": q,
                "limit": 1,
                "namespace": 0,
                "format": "json",
            }
        )
    )
    data = _http_get_json(search_url)
    # OpenSearch returns [query, [titles], [descriptions], [urls]]
    if not data or not isinstance(data, list) or len(data) < 2 or not data[1]:
        return None, None
    title: str = data[1][0]

    # Step 2: REST summary endpoint -> { "extract": "..." }
    summary_url = (
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{parse.quote(title)}"
    )
    summary = _http_get_json(summary_url)
    if not summary:
        return None, title

    extract = (summary.get("extract") or "").strip()
    if not extract:
        return None, title

    # Trim to a reasonable size on a sentence boundary if possible.
    if len(extract) > _MAX_EXTRACT_CHARS:
        snippet = extract[:_MAX_EXTRACT_CHARS]
        if ". " in snippet:
            snippet = snippet.rsplit(". ", 1)[0] + "."
        extract = snippet

    return extract, title


def should_ground(query: str) -> bool:
    """Heuristic: ground via Wikipedia when the query is short / underspecified.

    Short queries (<= 6 tokens) give a small 1B model nothing to anchor on,
    so it tends to hallucinate. We fetch an encyclopaedic snippet to anchor
    its answer.
    """
    q = _clean_query(query)
    if not q:
        return False
    return len(q.split()) <= 6
