from __future__ import annotations

import json
import math
import uuid
from typing import Any

import httpx

DEFAULT_ANTHROPIC_MAX_TOKENS = 4096


def anthropic_error_type(status_code: int) -> str:
    if status_code == 400:
        return "invalid_request_error"
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 429:
        return "rate_limit_error"
    return "api_error"


def anthropic_error_payload(status_code: int, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "type": anthropic_error_type(status_code),
            "message": message,
        },
    }


def anthropic_models_payload(model_ids: list[str]) -> dict[str, Any]:
    data = [anthropic_model_payload(model_id) for model_id in model_ids]
    return {
        "data": data,
        "first_id": data[0]["id"] if data else None,
        "has_more": False,
        "last_id": data[-1]["id"] if data else None,
    }


def anthropic_model_payload(model_id: str) -> dict[str, Any]:
    return {
        "created_at": "2025-01-01T00:00:00Z",
        "display_name": model_id,
        "id": model_id,
        "type": "model",
    }


def anthropic_count_tokens_payload(payload: dict[str, Any]) -> dict[str, int]:
    # 这里做本地估算，给 Anthropic SDK 一个兼容的 count_tokens 接口。
    anthropic_to_openai_request({**payload, "max_tokens": payload.get("max_tokens", 1), "stream": False})
    estimated = _estimate_tokens(payload)
    return {"input_tokens": estimated}


def normalize_anthropic_request(payload: dict[str, Any]) -> dict[str, Any]:
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("field `model` is required and must be a string")

    max_tokens = payload.get("max_tokens", DEFAULT_ANTHROPIC_MAX_TOKENS)
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("field `max_tokens` must be a positive integer")

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("field `messages` is required and must be a non-empty list")

    request = dict(payload)
    request["model"] = model
    request["max_tokens"] = max_tokens
    request["stream"] = bool(payload.get("stream", False))
    return request


def anthropic_to_openai_request(payload: dict[str, Any]) -> dict[str, Any]:
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("field `model` is required and must be a string")

    # Anthropic 官方接口要求 max_tokens，但为了兼容更多客户端，这里允许省略并给默认值。
    max_tokens = payload.get("max_tokens", DEFAULT_ANTHROPIC_MAX_TOKENS)
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("field `max_tokens` must be a positive integer")

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("field `messages` is required and must be a non-empty list")

    openai_messages: list[dict[str, Any]] = []
    system = payload.get("system")
    if system is not None:
        system_text = _blocks_to_text(_normalize_blocks(system))
        if system_text:
            openai_messages.append({"role": "system", "content": system_text})

    for message in messages:
        openai_messages.extend(_anthropic_message_to_openai_messages(message))

    request: dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
        "max_tokens": max_tokens,
        "stream": bool(payload.get("stream", False)),
    }

    # 只透传最常见、兼容性较好的 OpenAI 字段，避免上游 OpenAI 兼容接口因扩展参数报 400。
    for field in ("temperature", "top_p"):
        if field in payload:
            request[field] = payload[field]

    if "stop_sequences" in payload:
        request["stop"] = payload["stop_sequences"]

    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        request["tools"] = [_anthropic_tool_to_openai(tool) for tool in tools]

    if "tool_choice" in payload:
        request["tool_choice"] = _anthropic_tool_choice_to_openai(payload["tool_choice"])

    output_config = payload.get("output_config")
    response_format = _anthropic_output_config_to_openai(output_config)
    if response_format is not None:
        request["response_format"] = response_format

    return request


def openai_to_anthropic_response(data: dict[str, Any], model_alias: str) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}

    content = _openai_message_to_anthropic_content(message)

    return {
        "id": _anthropic_message_id(data.get("id")),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model_alias,
        "stop_reason": _map_finish_reason(choice.get("finish_reason"), content),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


async def anthropic_stream_from_openai(upstream_response: httpx.Response, model_alias: str):
    message_started = False
    text_block_open = False
    text_block_index = 0
    next_block_index = 0
    tool_blocks: dict[int, dict[str, Any]] = {}
    stop_reason = "end_turn"
    output_tokens = 0
    message_id = _anthropic_message_id(None)

    try:
        async for line in upstream_response.aiter_lines():
            if not line or not line.startswith("data:"):
                continue

            raw_data = line[5:].strip()
            if raw_data == "[DONE]":
                break

            chunk = json.loads(raw_data)
            message_id = _anthropic_message_id(chunk.get("id") or message_id)

            if not message_started:
                message_started = True
                yield _sse_event(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model_alias,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    },
                )

            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}

            if delta.get("content"):
                if not text_block_open:
                    text_block_index = next_block_index
                    next_block_index += 1
                    text_block_open = True
                    yield _sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": text_block_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                yield _sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": text_block_index,
                        "delta": {"type": "text_delta", "text": delta["content"]},
                    },
                )

            for tool_call in delta.get("tool_calls") or []:
                openai_index = tool_call.get("index", 0)
                block = tool_blocks.get(openai_index)
                if block is None:
                    block = {
                        "index": next_block_index,
                        "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": "tool",
                        "started": False,
                    }
                    tool_blocks[openai_index] = block
                    next_block_index += 1

                function = tool_call.get("function") or {}
                if tool_call.get("id"):
                    block["id"] = tool_call["id"]
                if function.get("name"):
                    block["name"] = function["name"]

                if not block["started"]:
                    if text_block_open:
                        yield _sse_event(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": text_block_index},
                        )
                        text_block_open = False

                    block["started"] = True
                    yield _sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block["index"],
                            "content_block": {
                                "type": "tool_use",
                                "id": block["id"],
                                "name": block["name"],
                                "input": {},
                            },
                        },
                    )

                arguments_delta = function.get("arguments") or ""
                if arguments_delta:
                    yield _sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block["index"],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": arguments_delta,
                            },
                        },
                    )

            finish_reason = choice.get("finish_reason")
            if finish_reason:
                stop_reason = _map_finish_reason(finish_reason, [])

            usage = chunk.get("usage") or {}
            if "completion_tokens" in usage:
                output_tokens = usage["completion_tokens"]

        if not message_started:
            yield _sse_event("message_start", {"type": "message_start", "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model_alias,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }})

        if text_block_open:
            yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": text_block_index})

        for block in tool_blocks.values():
            if block["started"]:
                yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": block["index"]})

        yield _sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )
        yield _sse_event("message_stop", {"type": "message_stop"})
    except httpx.HTTPError as exc:
        yield _sse_event(
            "error",
            anthropic_error_payload(500, f"stream terminated: {exc}"),
        )


def _anthropic_message_to_openai_messages(message: dict[str, Any]) -> list[dict[str, Any]]:
    role = message.get("role")
    if role not in {"user", "assistant"}:
        raise ValueError("Anthropic messages only support `user` and `assistant` roles")

    blocks = _normalize_blocks(message.get("content", ""))
    openai_messages: list[dict[str, Any]] = []

    if role == "assistant":
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in blocks:
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": block.get("name", "tool"),
                            "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                        },
                    }
                )
            else:
                raise ValueError(f"unsupported Anthropic content block type: {block_type}")

        assistant_message: dict[str, Any] = {"role": "assistant"}
        if text_parts:
            assistant_message["content"] = "".join(text_parts)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        if "content" not in assistant_message:
            assistant_message["content"] = ""
        openai_messages.append(assistant_message)
        return openai_messages

    user_text_parts: list[str] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            user_text_parts.append(block.get("text", ""))
            continue
        if block_type == "tool_result":
            content_text = _blocks_to_text(_normalize_blocks(block.get("content", "")))
            openai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": content_text,
                }
            )
            continue
        raise ValueError(f"unsupported Anthropic content block type: {block_type}")

    if user_text_parts:
        openai_messages.insert(0, {"role": "user", "content": "".join(user_text_parts)})

    return openai_messages


def _anthropic_tool_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        },
    }


def _anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return "auto"
        if choice_type == "any":
            return "required"
        if choice_type == "tool":
            return {
                "type": "function",
                "function": {"name": tool_choice.get("name", "")},
            }
    return tool_choice


def _openai_message_to_anthropic_content(message: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []

    message_content = message.get("content")
    if isinstance(message_content, str) and message_content:
        content.append({"type": "text", "text": message_content})
    elif isinstance(message_content, list):
        text_parts = [part.get("text", "") for part in message_content if isinstance(part, dict) and part.get("type") == "text"]
        if text_parts:
            content.append({"type": "text", "text": "".join(text_parts)})

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            parsed_arguments = {"raw": arguments}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": function.get("name", "tool"),
                "input": parsed_arguments,
            }
        )

    if not content:
        content.append({"type": "text", "text": ""})
    return content


def _normalize_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        normalized: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("message content blocks must be objects")
            normalized.append(block)
        return normalized
    raise ValueError("message content must be a string or a list of content blocks")


def _blocks_to_text(blocks: list[dict[str, Any]]) -> str:
    text_parts: list[str] = []
    for block in blocks:
        if block.get("type") != "text":
            raise ValueError("only text blocks are supported in this field")
        text_parts.append(block.get("text", ""))
    return "\n\n".join(part for part in text_parts if part)


def _anthropic_message_id(raw_id: Any) -> str:
    if isinstance(raw_id, str) and raw_id.startswith("msg_"):
        return raw_id
    if isinstance(raw_id, str) and raw_id:
        return f"msg_{raw_id}"
    return f"msg_{uuid.uuid4().hex}"


def _map_finish_reason(finish_reason: Any, content: list[dict[str, Any]]) -> str | None:
    if any(block.get("type") == "tool_use" for block in content):
        return "tool_use"
    if finish_reason == "stop":
        return "end_turn"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "content_filter":
        return "refusal"
    if finish_reason == "tool_calls":
        return "tool_use"
    return finish_reason or "end_turn"


def _sse_event(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def _anthropic_output_config_to_openai(output_config: Any) -> dict[str, Any] | None:
    if not isinstance(output_config, dict):
        return None

    format_config = output_config.get("format")
    if not isinstance(format_config, dict):
        return None

    if format_config.get("type") == "json_schema" and isinstance(format_config.get("schema"), dict):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "anthropic_output",
                "schema": format_config["schema"],
            },
        }

    if format_config.get("type") == "json_object":
        return {"type": "json_object"}

    return None


def _estimate_tokens(data: Any) -> int:
    if data is None:
        return 0
    if isinstance(data, str):
        # 粗略估算：按 UTF-8 字节近似，英文更接近 4 chars/token，中文会稍大一些。
        return max(1, math.ceil(len(data.encode("utf-8")) / 4))
    if isinstance(data, bool):
        return 1
    if isinstance(data, (int, float)):
        return 1
    if isinstance(data, list):
        return sum(_estimate_tokens(item) for item in data) + len(data) * 2
    if isinstance(data, dict):
        return sum(_estimate_tokens(key) + _estimate_tokens(value) for key, value in data.items()) + len(data) * 2
    return _estimate_tokens(str(data))
