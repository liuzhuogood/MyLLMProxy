from __future__ import annotations

import random
import threading
from dataclasses import dataclass

from app.config import ProviderConfig, RuntimeConfig, RouteConfig, TargetConfig


@dataclass(frozen=True)
class ResolvedTarget:
    alias: str
    provider: ProviderConfig
    upstream_model: str


class ModelRouter:
    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._providers = {provider.name: provider for provider in config.providers}
        self._counter_lock = threading.Lock()
        self._round_robin_counters: dict[str, int] = {}

    def list_models(self) -> list[str]:
        return sorted(self._config.routes.keys())

    def route_candidates(self, model_alias: str) -> list[ResolvedTarget]:
        route = self._config.routes.get(model_alias)
        if route is None:
            raise KeyError(model_alias)

        # 这里返回的是“尝试顺序”，前面的优先请求，失败后再回退到后面的节点。
        ordered_targets = self._order_targets(model_alias, route)
        return [
            ResolvedTarget(
                alias=model_alias,
                provider=self._providers[target.provider],
                upstream_model=target.upstream_model,
            )
            for target in ordered_targets
        ]

    def _order_targets(self, model_alias: str, route: RouteConfig) -> list[TargetConfig]:
        strategy = (route.strategy or self._config.gateway.strategy).lower()
        targets = list(route.targets)

        if strategy == "round_robin":
            # 最简单的轮询：每个模型别名单独维护一个计数器。
            with self._counter_lock:
                offset = self._round_robin_counters.get(model_alias, 0)
                self._round_robin_counters[model_alias] = offset + 1
            index = offset % len(targets)
            return targets[index:] + targets[:index]

        if strategy == "random":
            random.shuffle(targets)
            return targets

        if strategy == "weighted_random":
            chosen = random.choices(targets, weights=[target.weight for target in targets], k=1)[0]
            remaining = [target for target in targets if target is not chosen]
            random.shuffle(remaining)
            return [chosen, *remaining]

        raise ValueError(
            f"unsupported routing strategy {strategy!r}; "
            "expected round_robin, random, or weighted_random"
        )
