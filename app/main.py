from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.anthropic import anthropic_error_payload
from app.config import RuntimeConfig, load_runtime_config, resolve_config_path
from app.proxy import OpenAIProxyService
from app.routing import ModelRouter


def create_app(
        config: RuntimeConfig | None = None,
        config_path: str | Path | None = None,
        http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    resolved_config_path = None if config is not None else resolve_config_path(config_path)
    runtime_config = config or load_runtime_config(resolved_config_path)
    model_router = ModelRouter(runtime_config)
    client = http_client or httpx.AsyncClient()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            if http_client is None:
                await client.aclose()

    app = FastAPI(title="MyLLMProxy", version="0.1.0", lifespan=lifespan)
    # 提前挂到 app.state，测试和运行时都能直接取到。
    app.state.proxy_service = OpenAIProxyService(
        runtime_config,
        model_router,
        client,
        config_path=resolved_config_path,
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request):
        proxy_service: OpenAIProxyService = request.app.state.proxy_service
        proxy_service.reload_config_if_needed()
        if request.headers.get("anthropic-version") or request.headers.get("x-api-key"):
            proxy_service.validate_gateway_api_key(request)
            return JSONResponse(proxy_service.anthropic_models_payload())

        proxy_service.validate_gateway_api_key(request)
        return JSONResponse(proxy_service.list_models_payload())

    @app.get("/anthropic/v1/models")
    async def anthropic_models_alias(request: Request):
        proxy_service: OpenAIProxyService = request.app.state.proxy_service
        proxy_service.reload_config_if_needed()
        proxy_service.validate_gateway_api_key(request)
        return JSONResponse(proxy_service.anthropic_models_payload())

    @app.get("/v1/models/{model_id}")
    async def anthropic_model_detail(request: Request, model_id: str):
        proxy_service: OpenAIProxyService = request.app.state.proxy_service
        proxy_service.reload_config_if_needed()
        proxy_service.validate_gateway_api_key(request)
        try:
            return JSONResponse(proxy_service.anthropic_model_payload(model_id))
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "request failed"
            return JSONResponse(
                status_code=exc.status_code,
                content=anthropic_error_payload(exc.status_code, detail),
            )

    @app.get("/anthropic/v1/models/{model_id}")
    async def anthropic_model_detail_alias(request: Request, model_id: str):
        proxy_service: OpenAIProxyService = request.app.state.proxy_service
        proxy_service.reload_config_if_needed()
        proxy_service.validate_gateway_api_key(request)
        try:
            return JSONResponse(proxy_service.anthropic_model_payload(model_id))
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "request failed"
            return JSONResponse(
                status_code=exc.status_code,
                content=anthropic_error_payload(exc.status_code, detail),
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        proxy_service: OpenAIProxyService = request.app.state.proxy_service
        proxy_service.reload_config_if_needed()
        # 这里只做最小校验，剩下的字段尽量原样透传给上游。
        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "request body must be valid JSON"}},
            )
        if not isinstance(payload, dict):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "request body must be a JSON object"}},
            )
        proxy_service.validate_route_api_key(request, payload)
        return await proxy_service.proxy_chat_completions(payload)

    @app.post("/v1/responses")
    async def anthropic_messages(request: Request):
        return await anthropic_messages(request)

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        proxy_service: OpenAIProxyService = request.app.state.proxy_service
        proxy_service.reload_config_if_needed()

        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse(
                status_code=400,
                content=anthropic_error_payload(400, "request body must be valid JSON"),
            )

        if not isinstance(payload, dict):
            return JSONResponse(
                status_code=400,
                content=anthropic_error_payload(400, "request body must be a JSON object"),
            )

        try:
            proxy_service.validate_route_api_key(request, payload)
            return await proxy_service.proxy_anthropic_messages(payload, dict(request.headers))
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "request failed"
            return JSONResponse(
                status_code=exc.status_code,
                content=anthropic_error_payload(exc.status_code, detail),
            )

    @app.post("/anthropic/v1/messages")
    async def anthropic_messages_alias(request: Request):
        return await anthropic_messages(request)

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request):
        proxy_service: OpenAIProxyService = request.app.state.proxy_service
        proxy_service.reload_config_if_needed()

        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse(
                status_code=400,
                content=anthropic_error_payload(400, "request body must be valid JSON"),
            )

        if not isinstance(payload, dict):
            return JSONResponse(
                status_code=400,
                content=anthropic_error_payload(400, "request body must be a JSON object"),
            )

        try:
            proxy_service.validate_route_api_key(request, payload)
            return JSONResponse(proxy_service.anthropic_count_tokens_payload(payload))
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "request failed"
            return JSONResponse(
                status_code=exc.status_code,
                content=anthropic_error_payload(exc.status_code, detail),
            )

    @app.post("/anthropic/v1/messages/count_tokens")
    async def anthropic_count_tokens_alias(request: Request):
        return await anthropic_count_tokens(request)

    return app


app = create_app()
