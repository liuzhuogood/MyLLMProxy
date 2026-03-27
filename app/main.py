from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import RuntimeConfig, load_runtime_config
from app.proxy import OpenAIProxyService
from app.routing import ModelRouter


def create_app(
    config: RuntimeConfig | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    runtime_config = config or load_runtime_config()
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
    app.state.proxy_service = OpenAIProxyService(runtime_config, model_router, client)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request):
        proxy_service: OpenAIProxyService = request.app.state.proxy_service
        proxy_service.validate_gateway_api_key(request)
        return JSONResponse(proxy_service.list_models_payload())

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        proxy_service: OpenAIProxyService = request.app.state.proxy_service
        proxy_service.validate_gateway_api_key(request)
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
        return await proxy_service.proxy_chat_completions(payload)

    return app


app = create_app()
