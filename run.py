from __future__ import annotations

import os

import uvicorn


def main() -> None:
    # 保持启动方式尽量简单，默认监听本机 8000 端口。
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "4000"))
    reload = os.getenv("RELOAD", "true").lower() in {"1", "true", "yes", "on"}

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
