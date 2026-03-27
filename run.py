from __future__ import annotations

import logging
import os

import uvicorn


def main() -> None:
    # 保持启动方式尽量简单，默认监听本机 8000 端口。
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "4000"))
    reload = os.getenv("RELOAD", "true").lower() in {"1", "true", "yes", "on"}
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower(),
    )


if __name__ == "__main__":
    main()
