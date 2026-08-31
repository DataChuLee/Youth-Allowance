import json
import sys
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from app.core.cache import (
    CACHE_KEY_PREFIX,
    CACHE_NAMESPACE,
    DisabledQueryCache,
    QueryCache,
    deserialize_results,
)
from app.graph.state import RetrievedDocument


class FakeRedisClient:
    def __init__(
        self,
        *,
        ping_error: Exception | None = None,
        get_error: Exception | None = None,
        set_error: Exception | None = None,
        delete_error: Exception | None = None,
    ) -> None:
        self.ping_error = ping_error
        self.get_error = get_error
        self.set_error = set_error
        self.delete_error = delete_error
        self.values: dict[str | bytes, bytes] = {}
        self.deleted_keys: tuple[str | bytes, ...] = ()

    def ping(self) -> None:
        if self.ping_error is not None:
            raise self.ping_error

    def get(self, key: str) -> bytes | None:
        if self.get_error is not None:
            raise self.get_error
        return self.values.get(key)

    def setex(self, key: str, _ttl: int, value: bytes) -> None:
        if self.set_error is not None:
            raise self.set_error
        self.values[key] = value

    def scan_iter(self, *, match: str):
        prefix = match.removesuffix("*")
        return iter(
            key
            for key in self.values
            if (key.decode() if isinstance(key, bytes) else key).startswith(prefix)
        )

    def delete(self, *keys: str | bytes) -> int:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted_keys = keys
        deleted = 0
        for key in keys:
            if key in self.values:
                deleted += 1
                del self.values[key]
        return deleted


def install_fake_redis(monkeypatch, client: FakeRedisClient) -> None:
    redis_module = SimpleNamespace(from_url=lambda *_args, **_kwargs: client)
    monkeypatch.setitem(sys.modules, "redis", redis_module)


def make_result(
    *,
    metadata: dict | None = None,
    score: float = 0.0,
) -> RetrievedDocument:
    return RetrievedDocument(
        document=Document(
            page_content="청년수당 안내",
            metadata=metadata or {"page": 3, "chunk_id": "page-3-chunk-1"},
        ),
        score=score,
    )


def test_query_cache_falls_back_when_redis_is_unavailable(monkeypatch) -> None:
    client = FakeRedisClient(ping_error=ConnectionError("redis unavailable"))
    install_fake_redis(monkeypatch, client)

    cache = QueryCache("redis://local-test:6379")

    assert cache.enabled is False
    assert cache.get("질문") is None
    cache.set("질문", [make_result()])
    assert cache.flush() == 0


def test_query_cache_round_trips_json_results_and_metadata(monkeypatch) -> None:
    client = FakeRedisClient()
    install_fake_redis(monkeypatch, client)
    cache = QueryCache("redis://local-test:6379")
    metadata = {
        "page": 3,
        "chunk_id": "page-3-chunk-1",
        "optional": None,
        "nested": {"enabled": True},
    }

    cache.set("질문", [make_result(metadata=metadata, score=0.0)])
    restored = cache.get("질문")

    assert restored is not None
    assert restored[0].document.page_content == "청년수당 안내"
    assert restored[0].document.metadata == metadata
    assert restored[0].score == 0.0
    stored = next(iter(client.values.values()))
    assert json.loads(stored)["schema_version"] == 1
    assert not stored.startswith(b"\x80")


def test_query_cache_normalizes_non_json_metadata(monkeypatch, tmp_path: Path) -> None:
    client = FakeRedisClient()
    install_fake_redis(monkeypatch, client)
    cache = QueryCache("redis://local-test:6379")
    metadata = {
        "path": tmp_path / "booklet.pdf",
        "labels": {"policy", "pdf"},
        "coordinates": (1, 2),
    }

    cache.set("질문", [make_result(metadata=metadata, score=0.75)])
    restored = cache.get("질문")

    assert restored is not None
    assert restored[0].document.metadata["path"] == str(tmp_path / "booklet.pdf")
    assert set(restored[0].document.metadata["labels"]) == {"policy", "pdf"}
    assert restored[0].document.metadata["coordinates"] == [1, 2]
    assert restored[0].score == 0.75


def test_query_cache_treats_corrupt_json_as_cache_miss(monkeypatch) -> None:
    client = FakeRedisClient()
    install_fake_redis(monkeypatch, client)
    cache = QueryCache("redis://local-test:6379")
    client.values[cache._key("질문")] = b"{not-json"

    assert cache.get("질문") is None


def test_query_cache_treats_invalid_schema_and_types_as_cache_miss(monkeypatch) -> None:
    client = FakeRedisClient()
    install_fake_redis(monkeypatch, client)
    cache = QueryCache("redis://local-test:6379")
    invalid_payloads = [
        {"schema_version": 999, "results": []},
        {"schema_version": 1, "results": {}},
        {
            "schema_version": 1,
            "results": [{"page_content": 7, "metadata": {}, "score": 0.5}],
        },
        {
            "schema_version": 1,
            "results": [{"page_content": "text", "metadata": [], "score": 0.5}],
        },
        {
            "schema_version": 1,
            "results": [{"page_content": "text", "metadata": {}, "score": "high"}],
        },
    ]

    for payload in invalid_payloads:
        client.values[cache._key("질문")] = json.dumps(payload).encode()
        assert cache.get("질문") is None


def test_deserialize_results_does_not_accept_pickle_bytes() -> None:
    # v4 pickle payload를 호환 목적으로 역직렬화하지 않는다.
    assert deserialize_results(b"\x80\x04N.") is None


def test_query_cache_ignores_previous_namespace_and_flushes_only_current(
    monkeypatch,
) -> None:
    client = FakeRedisClient()
    install_fake_redis(monkeypatch, client)
    cache = QueryCache("redis://local-test:6379")
    old_key = "rag:v4:legacy"
    current_keys = (f"{CACHE_KEY_PREFIX}first", f"{CACHE_KEY_PREFIX}second")
    client.values = {
        old_key: b"legacy-pickle",
        current_keys[0]: b"{}",
        current_keys[1]: b"{}",
    }

    assert CACHE_NAMESPACE == "rag:v5"
    assert cache.get("legacy") is None
    assert cache.flush() == 2
    assert client.deleted_keys == current_keys
    assert client.values == {old_key: b"legacy-pickle"}


def test_query_cache_get_set_and_delete_failures_are_isolated(monkeypatch) -> None:
    get_client = FakeRedisClient(get_error=ConnectionError("get failed"))
    install_fake_redis(monkeypatch, get_client)
    assert QueryCache("redis://local-test:6379").get("질문") is None

    set_client = FakeRedisClient(set_error=ConnectionError("set failed"))
    install_fake_redis(monkeypatch, set_client)
    QueryCache("redis://local-test:6379").set("질문", [make_result()])

    delete_client = FakeRedisClient(delete_error=ConnectionError("delete failed"))
    delete_client.values[f"{CACHE_KEY_PREFIX}key"] = b"{}"
    install_fake_redis(monkeypatch, delete_client)
    assert QueryCache("redis://local-test:6379").flush() == 0


def test_disabled_query_cache_is_a_safe_noop() -> None:
    cache = DisabledQueryCache()

    assert cache.enabled is False
    assert cache.get("질문") is None
    cache.set("질문", [make_result()])
    assert cache.flush() == 0
