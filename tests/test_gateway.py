import json

import httpx
import pytest

from app.config import RuntimeConfig
from app.main import create_app


def build_config() -> RuntimeConfig:
    return RuntimeConfig.model_validate(
        {
            "gateway": {"strategy": "round_robin", "timeout_seconds": 5},
            "providers": [
                {
                    "name": "p1",
                    "base_url": "https://provider-1.example.com",
                    "api_key": "provider-key-1",
                },
                {
                    "name": "p2",
                    "base_url": "https://provider-2.example.com",
                    "api_key": "provider-key-2",
                },
            ],
            "routes": {
                "demo-model": {
                    "api_key": "route-secret",
                    "targets": [
                        {"provider": "p1", "upstream_model": "upstream-a"},
                        {"provider": "p2", "upstream_model": "upstream-a"},
                    ]
                }
            },
        }
    )


@pytest.mark.asyncio
async def test_list_models_returns_alias() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream should not be called")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as upstream_client:
        app = create_app(config=build_config(), http_client=upstream_client)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "demo-model"


@pytest.mark.asyncio
async def test_round_robin_switches_provider() -> None:
    seen_hosts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        assert request.headers["authorization"] == (
            "Bearer provider-key-1" if request.url.host == "provider-1.example.com" else "Bearer provider-key-2"
        )
        payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [],
            },
        )

    upstream_transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=upstream_transport) as upstream_client:
        app = create_app(config=build_config(), http_client=upstream_client)
        downstream_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=downstream_transport,
            base_url="http://testserver",
        ) as client:
            for _ in range(2):
                response = await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer route-secret"},
                    json={
                        "model": "demo-model",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert response.status_code == 200
                assert response.json()["model"] == "demo-model"

    assert seen_hosts == ["provider-1.example.com", "provider-2.example.com"]


@pytest.mark.asyncio
async def test_retry_second_provider_when_first_fails() -> None:
    attempts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.host)
        if request.url.host == "provider-1.example.com":
            return httpx.Response(503, text="temporary error")
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "upstream-a",
                "choices": [],
            },
        )

    upstream_transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=upstream_transport) as upstream_client:
        app = create_app(config=build_config(), http_client=upstream_client)
        downstream_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=downstream_transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer route-secret"},
                json={
                    "model": "demo-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

    assert response.status_code == 200
    assert response.json()["model"] == "demo-model"
    assert attempts == ["provider-1.example.com", "provider-2.example.com"]


@pytest.mark.asyncio
async def test_route_api_key_is_required() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream should not be called")

    upstream_transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=upstream_transport) as upstream_client:
        app = create_app(config=build_config(), http_client=upstream_client)
        downstream_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=downstream_transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "demo-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

    assert response.status_code == 401
