from __future__ import annotations

import urllib.request
import urllib.parse
from html.parser import HTMLParser
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True, slots=True)
class WebRetrievalStatus:
    enabled: bool
    max_sources: int
    require_citation: bool
    fallback_allowed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DDGParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self.current_result: dict[str, str] | None = None
        self.in_title = False
        self.in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k: v or "" for k, v in attrs}
        cls = attrs_dict.get("class", "")
        
        if tag == "div" and "web-result" in cls:
            if self.current_result and self.current_result.get("url"):
                self.results.append(self.current_result)
            self.current_result = {"title": "", "url": "", "snippet": ""}
            
        elif self.current_result is not None:
            if tag == "a" and "result__a" in cls:
                self.in_title = True
                self.current_result["url"] = attrs_dict.get("href", "")
            elif tag == "a" and "result__snippet" in cls:
                self.in_snippet = True
            elif tag == "a" and "result__url" in cls:
                self.current_result["url"] = attrs_dict.get("href", "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self.in_title = False
            self.in_snippet = False

    def handle_data(self, data: str) -> None:
        if self.current_result is not None:
            if self.in_title:
                self.current_result["title"] += data
            elif self.in_snippet:
                self.current_result["snippet"] += data

    def close(self) -> None:
        super().close()
        if self.current_result and self.current_result.get("url"):
            self.results.append(self.current_result)
        # Clean URLs
        for r in self.results:
            url = r["url"]
            if "uddg=" in url:
                parsed = urllib.parse.urlparse(url)
                qs = urllib.parse.parse_qs(parsed.query)
                if "uddg" in qs:
                    url = qs["uddg"][0]
            r["url"] = url


class WebRetrievalProvider:
    def __init__(self, enabled: bool, max_sources: int, require_citation: bool, fallback_allowed: bool):
        self.enabled = enabled
        self.max_sources = max_sources
        self.require_citation = require_citation
        self.fallback_allowed = fallback_allowed

    def status(self) -> WebRetrievalStatus:
        return WebRetrievalStatus(
            enabled=self.enabled,
            max_sources=self.max_sources,
            require_citation=self.require_citation,
            fallback_allowed=self.fallback_allowed,
            detail="Internet retrieval is off by default and only runs when explicitly enabled.",
        )

    def retrieve(self, query: str) -> list[dict[str, str]]:
        if not self.enabled:
            return []
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8")
                parser = DDGParser()
                parser.feed(html)
                parser.close()
                return parser.results[:self.max_sources]
        except Exception:
            return []
