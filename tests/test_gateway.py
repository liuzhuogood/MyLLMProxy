import json
import time
from gzip import compress

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
        assert request.headers["accept-encoding"] == "identity"
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


@pytest.mark.asyncio
async def test_strip_content_encoding_from_downstream_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = {"id": "chatcmpl-test", "object": "chat.completion", "model": "upstream-a", "choices": []}
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
            },
            content=compress(json.dumps(payload).encode("utf-8")),
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
    assert "content-encoding" not in response.headers
    assert response.json()["model"] == "demo-model"


@pytest.mark.asyncio
async def test_anthropic_messages_non_stream() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert request.headers["anthropic-beta"] == "tools-2024-04-04"
        assert payload["model"] == "upstream-a"
        assert payload["max_tokens"] == 128
        assert payload["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert payload["messages"][1] == {"role": "user", "content": "Hello"}
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "upstream-a",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Hi there"},
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
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
                "/v1/messages",
                headers={
                    "x-api-key": "route-secret",
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "tools-2024-04-04",
                },
                json={
                    "model": "demo-model",
                    "system": "You are helpful.",
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert data["model"] == "demo-model"
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "Hi there"
    assert data["usage"] == {"input_tokens": 12, "output_tokens": 5}


@pytest.mark.asyncio
async def test_anthropic_messages_default_max_tokens() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["max_tokens"] == 4096
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "upstream-a",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Hi"},
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
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
                "/anthropic/v1/messages",
                headers={
                    "x-api-key": "route-secret",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "demo-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "Hi"


@pytest.mark.asyncio
async def test_anthropic_messages_stream() -> None:
    stream_body = (
        'data: {"id":"chatcmpl-test","choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-test","choices":[{"delta":{"content":" there"},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=stream_body.encode("utf-8"),
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
                "/v1/messages",
                headers={
                    "x-api-key": "route-secret",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "demo-model",
                    "max_tokens": 64,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "event: message_start" in response.text
    assert "event: content_block_delta" in response.text
    assert "event: message_stop" in response.text


@pytest.mark.asyncio
async def test_anthropic_models_list() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream should not be called")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as upstream_client:
        app = create_app(config=build_config(), http_client=upstream_client)
        downstream_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=downstream_transport, base_url="http://testserver") as client:
            response = await client.get(
                "/v1/models",
                headers={"x-api-key": "route-secret", "anthropic-version": "2023-06-01"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["has_more"] is False
    assert data["data"][0]["type"] == "model"
    assert data["data"][0]["id"] == "demo-model"


@pytest.mark.asyncio
async def test_anthropic_models_list_alias() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream should not be called")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as upstream_client:
        app = create_app(config=build_config(), http_client=upstream_client)
        downstream_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=downstream_transport, base_url="http://testserver") as client:
            response = await client.get(
                "/anthropic/v1/models",
                headers={"x-api-key": "route-secret", "anthropic-version": "2023-06-01"},
            )

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "demo-model"


@pytest.mark.asyncio
async def test_anthropic_model_detail() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream should not be called")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as upstream_client:
        app = create_app(config=build_config(), http_client=upstream_client)
        downstream_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=downstream_transport, base_url="http://testserver") as client:
            response = await client.get(
                "/v1/models/demo-model",
                headers={"x-api-key": "route-secret", "anthropic-version": "2023-06-01"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "demo-model"
    assert data["type"] == "model"


@pytest.mark.asyncio
async def test_anthropic_count_tokens() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream should not be called")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as upstream_client:
        app = create_app(config=build_config(), http_client=upstream_client)
        downstream_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=downstream_transport, base_url="http://testserver") as client:
            response = await client.post(
                "/v1/messages/count_tokens",
                headers={"x-api-key": "route-secret", "anthropic-version": "2023-06-01"},
                json={
                    "model": "demo-model",
                    "system": "You are helpful.",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

    assert response.status_code == 200
    data = response.json()
    assert "input_tokens" in data
    assert data["input_tokens"] > 0


@pytest.mark.asyncio
async def test_reload_config_when_file_changes(tmp_path) -> None:
    config_path = tmp_path / "providers.yaml"
    config_path.write_text(
        json.dumps(
            {
                "gateway": {"strategy": "round_robin", "timeout_seconds": 5},
                "providers": [{"name": "p1", "base_url": "https://provider-1.example.com"}],
                "routes": {
                    "demo-model": {
                        "targets": [{"provider": "p1", "upstream_model": "upstream-a"}]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("upstream should not be called")

    upstream_transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=upstream_transport) as upstream_client:
        app = create_app(config_path=config_path, http_client=upstream_client)
        downstream_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=downstream_transport,
            base_url="http://testserver",
        ) as client:
            response = await client.get("/v1/models")
            assert response.status_code == 200
            assert [item["id"] for item in response.json()["data"]] == ["demo-model"]

            time.sleep(0.01)
            config_path.write_text(
                json.dumps(
                    {
                        "gateway": {"strategy": "round_robin", "timeout_seconds": 5},
                        "providers": [{"name": "p1", "base_url": "https://provider-1.example.com"}],
                        "routes": {
                            "demo-model": {
                                "targets": [{"provider": "p1", "upstream_model": "upstream-a"}]
                            },
                            "demo-model-2": {
                                "targets": [{"provider": "p1", "upstream_model": "upstream-b"}]
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            response = await client.get("/v1/models")

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == ["demo-model", "demo-model-2"]
