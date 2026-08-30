import hashlib
import json
import logging
import math
from collections.abc import Mapping
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.graph.state import RetrievedDocument

logger = logging.getLogger(__name__)

# 직렬화 형식이 바뀌면 namespace를 올려 이전 형식과 섞이지 않게 한다.
CACHE_NAMESPACE = "rag:v5"
CACHE_KEY_PREFIX = f"{CACHE_NAMESPACE}:"
CACHE_SCHEMA_VERSION = 1

_instance: "QueryCache | None" = None


def _normalize_json_value(value: Any) -> Any:
    """메타데이터를 JSON 기본 타입으로 재귀적으로 정규화한다."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return _normalize_json_value(value.value)
    if isinstance(value, (Path, datetime, date, time)):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_normalize_json_value(item) for item in value]
    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}>"


def _is_json_metadata(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_metadata(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_metadata(item)
            for key, item in value.items()
        )
    return False


def serialize_results(results: list[RetrievedDocument]) -> bytes:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "results": [
            {
                "page_content": item.document.page_content,
                "metadata": _normalize_json_value(item.document.metadata),
                "score": item.score,
            }
            for item in results
        ],
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def deserialize_results(raw: bytes | str) -> list[RetrievedDocument] | None:
    """외부 캐시 값은 엄격한 schema를 통과한 경우에만 문서로 복원한다."""
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "results"}:
            return None
        if payload["schema_version"] != CACHE_SCHEMA_VERSION:
            return None
        raw_results = payload["results"]
        if not isinstance(raw_results, list):
            return None

        results: list[RetrievedDocument] = []
        for item in raw_results:
            if not isinstance(item, dict) or set(item) != {
                "page_content",
                "metadata",
                "score",
            }:
                return None
            page_content = item["page_content"]
            metadata = item["metadata"]
            score = item["score"]
            if not isinstance(page_content, str) or not isinstance(metadata, dict):
                return None
            if not _is_json_metadata(metadata):
                return None
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                return None
            if not math.isfinite(float(score)):
                return None
            results.append(
                RetrievedDocument(
                    document=Document(page_content=page_content, metadata=metadata),
                    score=float(score),
                )
            )
        return results
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


class QueryCache:
    """Redis 기반 쿼리 결과 캐시. Redis 연결 실패 시 자동으로 비활성화."""

    def __init__(self, url: str, ttl: int = 3600) -> None:
        self.ttl = ttl
        self.enabled = False
        try:
            import redis

            self._client = redis.from_url(url, decode_responses=False)
            self._client.ping()
            self.enabled = True
            logger.info("Redis 연결 성공 (TTL=%ds)", ttl)
        except Exception:
            # URL에는 비밀번호가 포함될 수 있으므로 연결 정보를 로그에 남기지 않는다.
            logger.info("Redis 연결 실패 — 쿼리 캐시 비활성화")

    def _key(self, query: str) -> str:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return f"{CACHE_KEY_PREFIX}{digest}"

    def get(self, query: str) -> list[RetrievedDocument] | None:
        if not self.enabled:
            return None
        try:
            raw = self._client.get(self._key(query))
            return deserialize_results(raw) if raw else None
        except Exception:
            return None

    def set(self, query: str, results: list[RetrievedDocument]) -> None:
        if not self.enabled:
            return
        try:
            self._client.setex(self._key(query), self.ttl, serialize_results(results))
        except Exception:
            # 캐시 직렬화나 Redis 쓰기 실패가 기본 RAG 응답을 중단하면 안 된다.
            return

    def flush(self) -> int:
        if not self.enabled:
            return 0
        try:
            keys = list(self._client.scan_iter(match=f"{CACHE_KEY_PREFIX}*"))
            if not keys:
                return 0
            return int(self._client.delete(*keys))
        except Exception:
            # 캐시 장애가 기본 RAG 요청을 중단시키지 않도록 실패를 격리한다.
            return 0


class DisabledQueryCache:
    """테스트 환경에서 외부 Redis 상태가 결과에 개입하지 않게 하는 no-op 캐시."""

    enabled = False

    def get(self, _: str) -> None:
        return None

    def set(self, _: str, __: list[RetrievedDocument]) -> None:
        return None

    def flush(self) -> int:
        return 0


def get_query_cache() -> QueryCache | DisabledQueryCache:
    global _instance
    from app.core.config import get_settings

    settings = get_settings()
    if settings.app_env == "test":
        return DisabledQueryCache()
    if _instance is None:
        _instance = QueryCache(url=settings.redis_url, ttl=settings.redis_cache_ttl)
    return _instance
