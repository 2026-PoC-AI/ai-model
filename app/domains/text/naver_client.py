from __future__ import annotations
from typing import List, Dict, Optional
import os
import html
import re

import httpx


class NaverNewsClient:
    """
    네이버 뉴스 검색 Open API 클라이언트.
    - 환경변수: NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
    """

    BASE_URL = "https://openapi.naver.com/v1/search/news.json"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        display: int = 3,
        timeout: float = 5.0,
    ):
        self.client_id = client_id or os.getenv("NAVER_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("NAVER_CLIENT_SECRET", "")
        self.display = int(display)
        self.timeout = timeout

    def _strip_html(self, s: str) -> str:
        s = re.sub(r"<[^>]+>", "", s or "")
        s = html.unescape(s)
        return s.strip()

    def search_news(self, query: str) -> List[Dict[str, str]]:
        """
        반환 형식(고정 스펙):
        [
          {"title": "...", "url": "...", "snippet": "..."},
          ...
        ]
        """
        query = (query or "").strip()
        if not query:
            return []

        # 키 없으면 조용히 빈 리스트
        if not self.client_id or not self.client_secret:
            return []

        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }
        params = {
            "query": query,
            "display": max(1, min(self.display, 10)),
            "sort": "sim",  # 유사도순
        }

        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(self.BASE_URL, headers=headers, params=params)
            r.raise_for_status()
            data = r.json()

        items = data.get("items", []) or []
        results: List[Dict[str, str]] = []
        for it in items:
            title = self._strip_html(it.get("title", ""))
            snippet = self._strip_html(it.get("description", ""))
            url = it.get("link", "") or ""
            if not url or not title:
                continue
            results.append({"title": title, "url": url, "snippet": snippet})

        return results
