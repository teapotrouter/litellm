"""Group grants open provider pass-through reads and never pass-through writes."""

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import litellm
from litellm import Router
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app
from litellm.types.vector_stores import (
    IndexCreateLiteLLMParams,
    LiteLLM_ManagedVectorStore,
    LiteLLM_ManagedVectorStoreIndex,
)
from litellm.vector_stores.vector_store_registry import VectorStoreIndexRegistry, VectorStoreRegistry
from tests.unit.proxy.vector_store_access_group_state import (
    OTHER_TEAM_ID,
    TEAM_ID,
    ProxyState,
    make_key,
    make_team,
)

MODULE: Final = "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints"
DISCOVERY: Final = "/vertex_ai/discovery/v1/projects/p/locations/global/collections/default_collection/dataStores"


def _store(store_id: str, provider: str, owner: str = OTHER_TEAM_ID) -> LiteLLM_ManagedVectorStore:
    return LiteLLM_ManagedVectorStore(
        vector_store_id=store_id,
        custom_llm_provider=provider,
        team_id=owner,
        vector_store_name=f"name-{store_id}",
        litellm_params={"api_base": "https://provider.example", "api_key": "k", "vertex_project": "p"},
    )


def _index(index_name: str, store_id: str) -> LiteLLM_ManagedVectorStoreIndex:
    return LiteLLM_ManagedVectorStoreIndex(
        id=index_name,
        index_name=index_name,
        litellm_params=IndexCreateLiteLLMParams(vector_store_index=index_name, vector_store_name=f"name-{store_id}"),
    )


def _with_index_permissions(key: UserAPIKeyAuth, indexes: Sequence[str] = ("idx-a", "idx-b")) -> UserAPIKeyAuth:
    return key.model_copy(
        update={
            "metadata": {
                "allowed_vector_store_indexes": [
                    {"index_name": name, "index_permissions": ["read", "write"]} for name in indexes
                ]
            }
        }
    )


@pytest.fixture
async def call_as(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Callable[[UserAPIKeyAuth], TestClient]]:
    """Team team-t's group lists store-a. Another team owns both stores, and the provider call is stubbed."""
    proxy: Final = ProxyState(monkeypatch)
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_team(make_team(groups=("g1",)))
    stores: Final = [_store("store-a", "milvus"), _store("store-b", "milvus")]
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry(vector_stores=stores))
    monkeypatch.setattr(
        litellm,
        "vector_store_index_registry",
        VectorStoreIndexRegistry(vector_store_indexes=[_index("idx-a", "store-a"), _index("idx-b", "store-b")]),
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.llm_router",
        Router(model_list=[{"model_name": "m", "litellm_params": {"model": "openai/m", "api_key": "k"}}]),
    )
    original_overrides: Final = app.dependency_overrides.copy()

    def client_for(key: UserAPIKeyAuth) -> TestClient:
        app.dependency_overrides[user_api_key_auth] = lambda: key
        return TestClient(app, raise_server_exceptions=False)

    forwarded: Final = AsyncMock(return_value={"forwarded": True})
    with (
        patch(f"{MODULE}.create_pass_through_route", return_value=forwarded),
        patch(f"{MODULE}.BaseOpenAIPassThroughHandler.base_openai_pass_through_handler", forwarded),
        patch(
            f"{MODULE}.user_api_key_auth",
            new=AsyncMock(side_effect=lambda **_: app.dependency_overrides[user_api_key_auth]()),
        ),
        patch(f"{MODULE}._prepare_vertex_auth_headers", new=AsyncMock(return_value=({}, False, "p", "global"))),
    ):
        yield client_for
    app.dependency_overrides = original_overrides


GROUP_MEMBER: Final = _with_index_permissions(make_key(team_id=TEAM_ID))

READS: Final = (
    pytest.param("POST", "/milvus/v2/vectordb/entities/search", {"collectionName": "{index}"}, id="milvus-search"),
    pytest.param("POST", "/milvus/v2/vectordb/entities/query", {"collectionName": "{index}"}, id="milvus-query"),
    pytest.param("POST", "/milvus/v2/vectordb/entities/get", {"collectionName": "{index}"}, id="milvus-get"),
    pytest.param("POST", "/azure_ai/indexes/{index}/docs/search", {"search": "hi"}, id="azure-search"),
    pytest.param("POST", "/azure_ai/indexes/{index}/docs/suggest", {"search": "hi"}, id="azure-suggest"),
    pytest.param("GET", "/azure_ai/indexes/{index}", None, id="azure-get"),
    pytest.param(
        "POST", DISCOVERY + "/{store}/servingConfigs/default_config:search", {"query": "hi"}, id="vertex-search"
    ),
)

WRITES: Final = (
    pytest.param("POST", "/milvus/v2/vectordb/entities/upsert", {"collectionName": "{index}"}, id="milvus-upsert"),
    pytest.param("POST", "/milvus/v2/vectordb/entities/insert", {"collectionName": "{index}"}, id="milvus-insert"),
    pytest.param("POST", "/azure_ai/indexes/{index}/docs/index", {"value": []}, id="azure-index"),
    pytest.param("POST", DISCOVERY + "/{store}/branches/0/documents:import", {}, id="vertex-import"),
    pytest.param("DELETE", DISCOVERY + "/{store}", None, id="vertex-delete"),
    pytest.param("POST", DISCOVERY + "/{store}/x:search/%2e%2e/%2e%2e/documents:purge", {}, id="vertex-traversal"),
    pytest.param("POST", DISCOVERY + "/{store}/documents:purge%3Fx:search", {}, id="vertex-encoded-query"),
    pytest.param("POST", DISCOVERY + "/{store}/documents:purge%23x:search", {}, id="vertex-encoded-fragment"),
    pytest.param("POST", DISCOVERY + "/{store}/documents:purge;x:search", {}, id="vertex-semicolon"),
    pytest.param("POST", DISCOVERY + "/{store}/models/m:search", {}, id="vertex-model-rewritten-after-the-gate"),
    pytest.param("POST", "/milvus/x:/v2/vectordb/entities/search", {"collectionName": "{index}"}, id="milvus-scheme"),
    pytest.param("POST", "/milvus/host:1/v2/vectordb/entities/search", {"collectionName": "{index}"}, id="milvus-port"),
    pytest.param("GET", "/azure_ai///indexes/{index}", None, id="azure-leading-empty-segment"),
    pytest.param("POST", DISCOVERY + "/{store}//servingConfigs/x:search", {}, id="vertex-empty-segment"),
    pytest.param(
        "POST", DISCOVERY + "/{store}/%2e%2e/store-b/servingConfigs/x:search", {}, id="vertex-into-another-store"
    ),
    pytest.param("DELETE", DISCOVERY + "/{store}/servingConfigs/x:search", None, id="vertex-delete-on-a-read-path"),
    pytest.param("POST", DISCOVERY + "/{store}/x:search/documents:purge", {}, id="vertex-read-inside-a-write-path"),
    pytest.param("POST", "/azure_ai/indexes/{index}/docs/index/docs/search", {}, id="azure-write-inside-a-read-path"),
    pytest.param("POST", "/azure_ai/indexes/{index}/docs/index/%2e%2e/search", {}, id="azure-traversal"),
    pytest.param(
        "POST",
        "/milvus/v2/vectordb/entities/delete%3Fx/entities/search",
        {"collectionName": "{index}"},
        id="milvus-encoded-query",
    ),
)


def _send(client: TestClient, method: str, path: str, body: Mapping[str, object] | None, index: str, store: str) -> int:
    def fill(value: object) -> object:
        return value.replace("{index}", index).replace("{store}", store) if isinstance(value, str) else value

    payload: Final = None if body is None else {name: fill(value) for name, value in body.items()}
    return client.request(method, str(fill(path)), json=payload).status_code


@pytest.mark.parametrize(("method", "path", "body"), READS)
def test_group_member_reads_only_the_group_store(
    call_as: Callable[[UserAPIKeyAuth], TestClient], method: str, path: str, body: Mapping[str, object] | None
) -> None:
    client: Final = call_as(GROUP_MEMBER)

    assert _send(client, method, path, body, "idx-a", "store-a") == 200
    assert _send(client, method, path, body, "idx-b", "store-b") == 403


@pytest.mark.parametrize(("method", "path", "body"), WRITES)
def test_group_member_cannot_write(
    call_as: Callable[[UserAPIKeyAuth], TestClient], method: str, path: str, body: Mapping[str, object] | None
) -> None:
    assert _send(call_as(GROUP_MEMBER), method, path, body, "idx-a", "store-a") == 403


@pytest.mark.parametrize(("method", "path", "body"), READS)
def test_key_limited_to_another_store_cannot_read_the_group_store(
    call_as: Callable[[UserAPIKeyAuth], TestClient], method: str, path: str, body: Mapping[str, object] | None
) -> None:
    key: Final = _with_index_permissions(make_key(("store-x",), team_id=TEAM_ID))

    assert _send(call_as(key), method, path, body, "idx-a", "store-a") == 403


def test_group_grant_does_not_replace_the_index_permission(call_as: Callable[[UserAPIKeyAuth], TestClient]) -> None:
    key: Final = _with_index_permissions(make_key(team_id=TEAM_ID), indexes=())
    response: Final = call_as(key).post("/milvus/v2/vectordb/entities/search", json={"collectionName": "idx-a"})

    assert response.status_code == 403


def test_store_owner_keeps_writes(call_as: Callable[[UserAPIKeyAuth], TestClient]) -> None:
    owner: Final = _with_index_permissions(make_key(team_id=OTHER_TEAM_ID))
    response: Final = call_as(owner).post("/milvus/v2/vectordb/entities/upsert", json={"collectionName": "idx-a"})

    assert response.status_code == 200
