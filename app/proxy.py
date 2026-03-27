from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from app.config import RuntimeConfig
from app.routing import ModelRouter, ResolvedTarget

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class UpstreamRetryableError(Exception):
    pass


class OpenAIProxyService:
    def __init__(
        self,
        config: RuntimeConfig,
        model_router: ModelRouter,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._config = config
        self._model_router = model_router
        self._http_client = http_client

    def list_models_payload(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "my-llm-proxy",
                    "permission": [],
                    "root": model,
                }
                for model in self._model_router.list_models()
            ],
        }

    async def proxy_chat_completions(self, payload: dict[str, Any]) -> Response:
        model_alias = payload.get("model")
        if not isinstance(model_alias, str) or not model_alias:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="field `model` is required and must be a string",
            )

        if not isinstance(payload.get("messages"), list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="field `messages` is required and must be a list",
            )

        stream = bool(payload.get("stream", False))
        try:
            candidates = self._model_router.route_candidates(model_alias)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"model {model_alias!r} is not configured",
            ) from exc

        errors: list[str] = []
        for target in candidates:
            # 对外暴露的是别名模型，对上游发送的是实际模型名。
            upstream_payload = dict(payload)
            upstream_payload["model"] = target.upstream_model
            try:
                return await self._dispatch_chat_completion(
                    target=target,
                    payload=upstream_payload,
                    response_model_alias=model_alias,
                    stream=stream,
                )
            except UpstreamRetryableError as exc:
                errors.append(f"{target.provider.name}: {exc}")

        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "error": {
                    "message": "all upstream providers failed",
                    "type": "upstream_error",
                    "details": errors,
                }
            },
        )

    async def _dispatch_chat_completion(
        self,
        target: ResolvedTarget,
        payload: dict[str, Any],
        response_model_alias: str,
        stream: bool,
    ) -> Response:
        headers = self._build_upstream_headers(target)
        url = self._join_url(target.provider.base_url, target.provider.chat_path)
        timeout = self._config.gateway.timeout_seconds

        if stream:
            try:
                upstream_request = self._http_client.build_request(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )
                upstream_response = await self._http_client.send(upstream_request, stream=True)
            except httpx.HTTPError as exc:
                raise UpstreamRetryableError(str(exc)) from exc
            if upstream_response.status_code in RETRYABLE_STATUS_CODES:
                body = await upstream_response.aread()
                await upstream_response.aclose()
                raise UpstreamRetryableError(
                    f"retryable status {upstream_response.status_code}: {body.decode('utf-8', 'ignore')}"
                )
            if upstream_response.status_code >= 400:
                return await self._build_error_response(upstream_response)
            response_headers = self._sanitize_response_headers(upstream_response.headers)
            media_type = upstream_response.headers.get("content-type", "text/event-stream")
            return StreamingResponse(
                # 流式模式不改写内容，直接把 SSE 数据往下游转发。
                self._stream_upstream(upstream_response),
                status_code=upstream_response.status_code,
                headers=response_headers,
                media_type=media_type,
                background=BackgroundTask(upstream_response.aclose),
            )

        try:
            upstream_response = await self._http_client.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise UpstreamRetryableError(str(exc)) from exc

        if upstream_response.status_code in RETRYABLE_STATUS_CODES:
            raise UpstreamRetryableError(
                f"retryable status {upstream_response.status_code}: {upstream_response.text}"
            )
        if upstream_response.status_code >= 400:
            return await self._build_error_response(upstream_response)

        response_headers = self._sanitize_response_headers(upstream_response.headers)
        content_type = upstream_response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return Response(
                content=upstream_response.content,
                status_code=upstream_response.status_code,
                headers=response_headers,
                media_type=content_type or None,
            )

        data = upstream_response.json()
        if isinstance(data, dict) and "model" in data:
            # 对客户端保持模型别名一致，隐藏真实上游模型名。
            data["model"] = response_model_alias
        return JSONResponse(
            content=data,
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    async def _build_error_response(self, upstream_response: httpx.Response) -> Response:
        headers = self._sanitize_response_headers(upstream_response.headers)
        content_type = upstream_response.headers.get("content-type", "")
        body = await upstream_response.aread()
        await upstream_response.aclose()
        if "application/json" in content_type:
            return JSONResponse(
                status_code=upstream_response.status_code,
                content=json.loads(body.decode("utf-8")),
                headers=headers,
            )
        return Response(
            status_code=upstream_response.status_code,
            content=body,
            headers=headers,
            media_type=content_type or None,
        )

    async def _stream_upstream(self, upstream_response: httpx.Response):
        try:
            async for chunk in upstream_response.aiter_raw():
                yield chunk
        except httpx.HTTPError as exc:
            error_payload = {
                "error": {
                    "message": f"stream terminated: {exc}",
                    "type": "upstream_stream_error",
                }
            }
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

    def validate_gateway_api_key(self, request: Request) -> None:
        expected_key = self._config.gateway.resolved_api_key()
        if not expected_key:
            return

        # 网关自己的 key 和上游 provider 的 key 分开，便于对外统一鉴权。
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or token != expected_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid gateway api key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def _build_upstream_headers(self, target: ResolvedTarget) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            **target.provider.headers,
        }
        api_key = target.provider.resolved_api_key()
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        return headers

    @staticmethod
    def _join_url(base_url: str, path: str) -> str:
        return f"{base_url.rstrip('/')}/{path.lstrip('/')}"

    @staticmethod
    def _sanitize_response_headers(headers: httpx.Headers) -> dict[str, str]:
        sanitized: dict[str, str] = {}
        for key, value in headers.items():
            if key.lower() in HOP_BY_HOP_HEADERS or key.lower() == "content-length":
                continue
            sanitized[key] = value
        return sanitized
