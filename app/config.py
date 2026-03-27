from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class GatewayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str = "round_robin"
    timeout_seconds: float = 90.0
    api_key: str | None = None


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    base_url: str
    api_key: str | None = None
    chat_path: str = "/v1/chat/completions"
    models_path: str = "/v1/models"
    headers: dict[str, str] = Field(default_factory=dict)


class TargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    upstream_model: str
    weight: int = 1

    @model_validator(mode="after")
    def validate_weight(self) -> "TargetConfig":
        if self.weight < 1:
            raise ValueError("weight must be >= 1")
        return self


class RouteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: str | None = None
    strategy: str | None = None
    targets: list[TargetConfig]

    @model_validator(mode="after")
    def validate_targets(self) -> "RouteConfig":
        if not self.targets:
            raise ValueError("at least one target is required")
        return self


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    providers: list[ProviderConfig]
    routes: dict[str, RouteConfig]

    @model_validator(mode="after")
    def validate_references(self) -> "RuntimeConfig":
        provider_names = {provider.name for provider in self.providers}
        missing = {
            target.provider
            for route in self.routes.values()
            for target in route.targets
            if target.provider not in provider_names
        }
        if missing:
            raise ValueError(f"unknown providers referenced in routes: {sorted(missing)}")
        return self


def _read_yaml(path: Path) -> dict[str, Any]:
    raw_text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text) or {}
    if not isinstance(data, dict):
        raise ValueError("config file must contain a top-level mapping")
    return data


def resolve_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    resolved_path = Path(path or os.getenv("MY_LLM_PROXY_CONFIG", "config/providers.yaml"))
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"config file not found: {resolved_path}. "
            "Set MY_LLM_PROXY_CONFIG or create config/providers.yaml."
        )
    return resolved_path


def load_runtime_config(path: str | os.PathLike[str] | None = None) -> RuntimeConfig:
    resolved_path = resolve_config_path(path)

    try:
        return RuntimeConfig.model_validate(_read_yaml(resolved_path))
    except ValidationError as exc:
        raise ValueError(f"invalid config file {resolved_path}: {exc}") from exc
