from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from app.anthropic import (
    anthropic_count_tokens_payload,
    anthropic_error_payload,
    anthropic_model_payload,
    anthropic_models_payload,
    anthropic_to_openai_request,
    anthropic_stream_from_openai,
    openai_to_anthropic_response,
)
from app.config import RuntimeConfig, load_runtime_config
from app.routing import ModelRouter, ResolvedTarget

HOP_BY_HOP_HEADERS = {
    "connection",
    "content-encoding",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
LOGGER = logging.getLogger("my_llm_proxy")


def setup_logger() -> None:
    # reload 模式下真正处理请求的是 uvicorn 子进程，这里要在应用侧保证日志能落到控制台。
    if LOGGER.handlers:
        return

    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))

    LOGGER.addHandler(handler)
    LOGGER.setLevel(log_level)
    LOGGER.propagate = False


setup_logger()


class UpstreamRetryableError(Exception):
    pass


class OpenAIProxyService:
    def __init__(
        self,
        config: RuntimeConfig,
        model_router: ModelRouter,
        http_client: httpx.AsyncClient,
        config_path: str | Path | None = None,
    ) -> None:
        self._config = config
        self._model_router = model_router
        self._http_client = http_client
        self._config_path = Path(config_path) if config_path else None
        self._config_mtime_ns = self._read_config_mtime_ns()
        self._reload_lock = threading.Lock()

    def reload_config_if_needed(self) -> None:
        if self._config_path is None:
            return

        current_mtime_ns = self._read_config_mtime_ns()
        if current_mtime_ns is None or current_mtime_ns == self._config_mtime_ns:
            return

        with self._reload_lock:
            current_mtime_ns = self._read_config_mtime_ns()
            if current_mtime_ns is None or current_mtime_ns == self._config_mtime_ns:
                return

            try:
                new_config = load_runtime_config(self._config_path)
            except Exception as exc:
                LOGGER.warning("配置文件重载失败 path=%s error=%s", self._config_path, exc)
                return

            self._config = new_config
            self._model_router = ModelRouter(new_config)
            self._config_mtime_ns = current_mtime_ns
            LOGGER.info("检测到配置文件变更，已自动重载 path=%s", self._config_path)

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

    def anthropic_models_payload(self) -> dict[str, Any]:
        return anthropic_models_payload(self._model_router.list_models())

    def anthropic_model_payload(self, model_id: str) -> dict[str, Any]:
        if model_id not in self._model_router.list_models():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"model {model_id!r} is not configured",
            )
        return anthropic_model_payload(model_id)

    def anthropic_count_tokens_payload(self, payload: dict[str, Any]) -> dict[str, int]:
        try:
            return anthropic_count_tokens_payload(payload)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

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

        self._get_route(model_alias)
        stream = bool(payload.get("stream", False))
        LOGGER.info(
            "收到客户端请求 model=%s stream=%s body=%s",
            model_alias,
            stream,
            self._dump_json(payload),
        )
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

    async def proxy_anthropic_messages(
        self,
        payload: dict[str, Any],
        request_headers: dict[str, str] | None = None,
    ) -> Response:
        try:
            openai_payload = anthropic_to_openai_request(payload)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        model_alias = openai_payload["model"]
        stream = bool(openai_payload.get("stream", False))
        LOGGER.info(
            "收到 Anthropic 请求 model=%s stream=%s body=%s",
            model_alias,
            stream,
            self._dump_json(payload),
        )

        try:
            candidates = self._model_router.route_candidates(model_alias)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"model {model_alias!r} is not configured",
            ) from exc

        errors: list[str] = []
        for target in candidates:
            upstream_payload = dict(openai_payload)
            upstream_payload["model"] = target.upstream_model
            try:
                return await self._dispatch_anthropic_messages(
                    target=target,
                    payload=upstream_payload,
                    response_model_alias=model_alias,
                    stream=stream,
                    request_headers=request_headers,
                )
            except UpstreamRetryableError as exc:
                errors.append(f"{target.provider.name}: {exc}")

        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=anthropic_error_payload(status.HTTP_502_BAD_GATEWAY, "all upstream providers failed"),
        )

    def validate_route_api_key(self, request: Request, payload: dict[str, Any]) -> None:
        model_alias = payload.get("model")
        if not isinstance(model_alias, str) or not model_alias:
            return

        route = self._get_route(model_alias)
        expected_key = route.api_key or self._config.gateway.api_key
        if not expected_key:
            return

        received_key = self._extract_client_api_key(request)
        if received_key != expected_key:
            LOGGER.warning("路由鉴权失败 model=%s", model_alias)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid route api key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def _get_route(self, model_alias: str):
        try:
            return self._model_router.get_route(model_alias)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"model {model_alias!r} is not configured",
            ) from exc

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
        started_at = time.perf_counter()
        LOGGER.info(
            "开始转发 model=%s provider=%s upstream_model=%s url=%s headers=%s body=%s",
            response_model_alias,
            target.provider.name,
            target.upstream_model,
            url,
            self._sanitize_log_headers(headers),
            self._dump_json(payload),
        )

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
                LOGGER.warning("上游请求异常 provider=%s error=%s", target.provider.name, exc)
                raise UpstreamRetryableError(str(exc)) from exc
            if upstream_response.status_code in RETRYABLE_STATUS_CODES:
                body = await upstream_response.aread()
                await upstream_response.aclose()
                LOGGER.warning(
                    "上游可重试失败 provider=%s status=%s elapsed_ms=%s body=%s",
                    target.provider.name,
                    upstream_response.status_code,
                    self._elapsed_ms(started_at),
                    self._truncate_text(body.decode("utf-8", "ignore")),
                )
                raise UpstreamRetryableError(
                    f"retryable status {upstream_response.status_code}: {body.decode('utf-8', 'ignore')}"
                )
            if upstream_response.status_code >= 400:
                LOGGER.warning(
                    "上游返回错误 provider=%s status=%s elapsed_ms=%s",
                    target.provider.name,
                    upstream_response.status_code,
                    self._elapsed_ms(started_at),
                )
                return await self._build_error_response(upstream_response)
            response_headers = self._sanitize_response_headers(upstream_response.headers)
            media_type = upstream_response.headers.get("content-type", "text/event-stream")
            LOGGER.info(
                "上游流式响应 provider=%s status=%s elapsed_ms=%s",
                target.provider.name,
                upstream_response.status_code,
                self._elapsed_ms(started_at),
            )
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
            LOGGER.warning("上游请求异常 provider=%s error=%s", target.provider.name, exc)
            raise UpstreamRetryableError(str(exc)) from exc

        if upstream_response.status_code in RETRYABLE_STATUS_CODES:
            LOGGER.warning(
                "上游可重试失败 provider=%s status=%s elapsed_ms=%s body=%s",
                target.provider.name,
                upstream_response.status_code,
                self._elapsed_ms(started_at),
                self._truncate_text(upstream_response.text),
            )
            raise UpstreamRetryableError(
                f"retryable status {upstream_response.status_code}: {upstream_response.text}"
            )
        if upstream_response.status_code >= 400:
            LOGGER.warning(
                "上游返回错误 provider=%s status=%s elapsed_ms=%s body=%s",
                target.provider.name,
                upstream_response.status_code,
                self._elapsed_ms(started_at),
                self._truncate_text(upstream_response.text),
            )
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
        LOGGER.info(
            "转发完成 provider=%s status=%s elapsed_ms=%s body=%s",
            target.provider.name,
            upstream_response.status_code,
            self._elapsed_ms(started_at),
            self._dump_json(data),
        )
        return JSONResponse(
            content=data,
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    async def _dispatch_anthropic_messages(
        self,
        target: ResolvedTarget,
        payload: dict[str, Any],
        response_model_alias: str,
        stream: bool,
        request_headers: dict[str, str] | None = None,
    ) -> Response:
        headers = self._build_upstream_headers(target, self._extract_anthropic_passthrough_headers(request_headers))
        url = self._join_url(target.provider.base_url, target.provider.chat_path)
        timeout = self._config.gateway.timeout_seconds
        started_at = time.perf_counter()
        LOGGER.info(
            "开始 Anthropic 转发 model=%s provider=%s upstream_model=%s url=%s headers=%s body=%s",
            response_model_alias,
            target.provider.name,
            target.upstream_model,
            url,
            self._sanitize_log_headers(headers),
            self._dump_json(payload),
        )

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
                LOGGER.warning("上游请求异常 provider=%s error=%s", target.provider.name, exc)
                raise UpstreamRetryableError(str(exc)) from exc

            if upstream_response.status_code in RETRYABLE_STATUS_CODES:
                body = await upstream_response.aread()
                await upstream_response.aclose()
                raise UpstreamRetryableError(
                    f"retryable status {upstream_response.status_code}: {body.decode('utf-8', 'ignore')}"
                )
            if upstream_response.status_code >= 400:
                return await self._build_anthropic_error_response(upstream_response)

            LOGGER.info(
                "Anthropic 流式响应 provider=%s status=%s elapsed_ms=%s",
                target.provider.name,
                upstream_response.status_code,
                self._elapsed_ms(started_at),
            )
            return StreamingResponse(
                anthropic_stream_from_openai(upstream_response, response_model_alias),
                status_code=upstream_response.status_code,
                headers=self._sanitize_response_headers(upstream_response.headers),
                media_type="text/event-stream",
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
            LOGGER.warning("上游请求异常 provider=%s error=%s", target.provider.name, exc)
            raise UpstreamRetryableError(str(exc)) from exc

        if upstream_response.status_code in RETRYABLE_STATUS_CODES:
            raise UpstreamRetryableError(
                f"retryable status {upstream_response.status_code}: {upstream_response.text}"
            )
        if upstream_response.status_code >= 400:
            return await self._build_anthropic_error_response(upstream_response)

        data = openai_to_anthropic_response(upstream_response.json(), response_model_alias)
        LOGGER.info(
            "Anthropic 转发完成 provider=%s status=%s elapsed_ms=%s body=%s",
            target.provider.name,
            upstream_response.status_code,
            self._elapsed_ms(started_at),
            self._dump_json(data),
        )
        return JSONResponse(
            content=data,
            status_code=upstream_response.status_code,
            headers=self._sanitize_response_headers(upstream_response.headers),
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

    async def _build_anthropic_error_response(self, upstream_response: httpx.Response) -> Response:
        body = await upstream_response.aread()
        await upstream_response.aclose()

        message = body.decode("utf-8", "ignore")
        content_type = upstream_response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                data = json.loads(message)
                message = (
                    data.get("error", {}).get("message")
                    or data.get("message")
                    or data.get("detail")
                    or message
                )
            except json.JSONDecodeError:
                pass

        return JSONResponse(
            status_code=upstream_response.status_code,
            content=anthropic_error_payload(upstream_response.status_code, message),
            headers=self._sanitize_response_headers(upstream_response.headers),
        )

    async def _stream_upstream(self, upstream_response: httpx.Response):
        try:
            # 用 aiter_bytes 让 httpx 处理上游压缩，避免把压缩头和解压后的内容混着转发。
            async for chunk in upstream_response.aiter_bytes():
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
        expected_key = self._config.gateway.api_key
        if not expected_key:
            return

        if self._extract_client_api_key(request) != expected_key:
            LOGGER.warning("网关鉴权失败 path=%s", request.url.path)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid gateway api key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def _build_upstream_headers_with_extra(
        self,
        target: ResolvedTarget,
        extra_headers: dict[str, str] | None,
    ) -> dict[str, str]:
        headers = {
            # 主动要求上游返回未压缩内容，减少代理层压缩/解压不一致的问题。
            "accept-encoding": "identity",
            "content-type": "application/json",
            **target.provider.headers,
        }
        if extra_headers:
            headers.update(extra_headers)
        api_key = target.provider.api_key
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        return headers

    def _build_upstream_headers(
        self,
        target: ResolvedTarget,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        return self._build_upstream_headers_with_extra(target, extra_headers)

    @staticmethod
    def _extract_client_api_key(request: Request) -> str:
        if request.headers.get("x-api-key"):
            return request.headers["x-api-key"].strip()

        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token.strip()
        return header.strip()

    @staticmethod
    def _extract_anthropic_passthrough_headers(request_headers: dict[str, str] | None) -> dict[str, str]:
        if not request_headers:
            return {}

        passthrough: dict[str, str] = {}
        for header_name in ("anthropic-version", "anthropic-beta"):
            value = request_headers.get(header_name)
            if value:
                passthrough[header_name] = value
        return passthrough

    def _read_config_mtime_ns(self) -> int | None:
        if self._config_path is None or not self._config_path.exists():
            return None
        return self._config_path.stat().st_mtime_ns

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

    @staticmethod
    def _sanitize_log_headers(headers: dict[str, str]) -> dict[str, str]:
        sanitized = dict(headers)
        if "authorization" in sanitized:
            sanitized["authorization"] = "***"
        if "Authorization" in sanitized:
            sanitized["Authorization"] = "***"
        return sanitized

    @staticmethod
    def _truncate_text(text: str, limit: int = 1500) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit]}...(truncated)"

    @classmethod
    def _dump_json(cls, data: Any) -> str:
        try:
            return cls._truncate_text(json.dumps(data, ensure_ascii=False))
        except TypeError:
            return cls._truncate_text(str(data))

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return int((time.perf_counter() - started_at) * 1000)
