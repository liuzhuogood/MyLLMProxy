# MyLLMProxy

一个尽量薄的 LLM 网关，目标只有三个：

1. 兼容常见的 OpenAI 接口路径
2. 通过配置接多个 provider
3. 在多个上游之间做简单负载和失败回退

当前只实现了最核心的两个接口：

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/messages`（Anthropic 兼容）
- `POST /v1/messages/count_tokens`（Anthropic 兼容）
- `GET /anthropic/v1/models` / `POST /anthropic/v1/messages`（Anthropic 命名空间别名）

## 为什么这样拆

- `app/config.py`：读取 YAML 配置
- `app/routing.py`：决定某个模型别名该打到哪个上游
- `app/proxy.py`：负责鉴权、转发请求、处理失败回退
- `app/anthropic.py`：负责 Anthropic 和 OpenAI 请求/响应格式互转
- `app/main.py`：FastAPI 入口

代码故意保持简单，没有做太多抽象，方便你后面继续改。

## 配置说明

默认读取 `config/providers.yaml`，也可以通过环境变量覆盖：

```bash
export MY_LLM_PROXY_CONFIG=/your/path/providers.yaml
```

示例配置：

```yaml
gateway:
  strategy: round_robin
  timeout_seconds: 90
  # api_key: sk-gateway

providers:
  - name: openai-primary
    base_url: https://api.openai.com
    api_key: sk-provider-1
    protocol: openai

  - name: openai-backup
    base_url: https://api.openai.com
    api_key: sk-provider-2
    protocol: openai

  - name: anthropic-primary
    base_url: https://api.anthropic.com
    api_key: sk-ant-provider-1
    protocol: anthropic
    chat_path: /v1/messages

routes:
  gpt-4o-mini:
    api_key: sk-demo
    strategy: round_robin
    targets:
      - provider: openai-primary
        upstream_model: gpt-4o-mini
      - provider: openai-backup
        upstream_model: gpt-4o-mini
```

这里的意思是：

- 对外暴露模型名 `gpt-4o-mini`
- 客户端调用这个模型时，要带 `Authorization: Bearer sk-demo`
- 实际会轮询打到 `openai-primary` 和 `openai-backup`
- 请求体里的 `model` 会被改成对应上游的 `upstream_model`
- 转发到上游时，会自动带上 provider 自己的 `api_key`

`providers[].protocol` 用来声明上游说的是哪套接口协议：

- `openai`：默认值。`/v1/messages` 会先转成 OpenAI `chat/completions` 再发给上游
- `anthropic`：`/v1/messages` 会保留 Anthropic 请求体直接转发到上游，同时自动带 `x-api-key` 和 `anthropic-version`

## 运行

先安装依赖：

```bash
pip install -e ".[dev]"
```

启动服务：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 4000 --reload
```

## 调用示例

```bash
curl http://127.0.0.1:4000/v1/models
```

```bash
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'
```

Anthropic 兼容调用：

```bash
curl http://127.0.0.1:4000/v1/messages \
  -H "x-api-key: sk-demo" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "max_tokens": 512,
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'
```

这个接口会把 Anthropic 风格的 `messages` 请求转换成 OpenAI 风格的 chat completion，再把响应转换回 Anthropic 风格。
如果没有传 `max_tokens`，网关会默认补成 `4096`。

如果路由目标 provider 配的是 `protocol: anthropic`，则不会做这层格式转换，而是直接把 Anthropic 请求发到上游的 `chat_path`，通常应配置为 `/v1/messages`。

Anthropic `count_tokens`：

```bash
curl http://127.0.0.1:4000/v1/messages/count_tokens \
  -H "x-api-key: sk-demo" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'
```

如果你是给 Claude Code 或其他 Anthropic SDK 用，推荐两种接法：

- 直接把基地址指到网关根路径，用 `/v1/messages`
- 或者把基地址指到带命名空间的 `/anthropic`，走 `/anthropic/v1/messages`

当前 `count_tokens` 是本地估算实现，主要目的是兼容 Anthropic 客户端调用流程，不依赖上游 provider 原生支持。

## 后续你可以继续加什么

- `/v1/embeddings`
- 更细的重试策略
- 每个 provider 的并发限制
- 调用日志、统计、熔断
- 管理后台或者热更新配置


curl https://llm.s5.5slive.com/anthropic/v1/messages \
  -H "x-api-key: sk-liuzhuo" \
  -H "max-tokens: 1000" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'


export ANTHROPIC_BASE_URL="https://llm.s5.5slive.com/anthropic" 
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000
export ANTHROPIC_AUTH_TOKEN="sk-liuzhuo" 
export ANTHROPIC_DEFAULT_OPUS_MODEL="a-glm-4.7" 
export ANTHROPIC_DEFAULT_SONNET_MODEL="a-glm-4.7" 
export ANTHROPIC_DEFAULT_HAIKU_MODEL="a-glm-4.7" 
export ANTHROPIC_MODEL="a-glm-4.7"
/Users/liuzhuo/.claude/local/claude


2026-03-27 17:45:35,663 [INFO] my_llm_proxy - 开始 Anthropic 转发 model=a-glm-4.7 provider=gs_a upstream_model=glm-4.7 
url=https://open.bigmodel.cn/api/anthropic 
headers={'accept-encoding': 'identity', 'content-type': 'application/json', 
'anthropic-version': '2023-06-01', 'anthropic-beta': 'claude-code-20250219,interleaved-thinking-2025-05-14,
context-management-2025-06-27,prompt-caching-scope-2026-01-05,effort-2025-11-24', 'authorization': '***'} 
body={"model": "glm-4.7", "messages": [{"role": "system", "content": "x-anthropic-billing-header: cc_version=2.1.85.351; 
cc_entrypoint=cli; cch=00000;You are Claude Code, Anthropic'
s official CLI for Claude.\nYou are an interactive agent that help
s users with software engineering tasks. Use the instructions below and the 
tools available to you to assist the user.\n\nIMPORTANT: Assist with authorized se
curity testing, defensive security, CTF challenges, and educational contexts. Refuse req
uests for destructive techniques, DoS attacks, mass targeting, supply chain compromise, or 
detection evasion for malicious purposes. Dual-use security tools (C2 frameworks, credential
testing, exploit development) require clear authorization context: pentesting engagements, CTF co
mpetitions, security research, or defensive use cases.\nIMPORTANT: You must NEVER generate or guess U
RLs for the user unless you are confident that the URLs are for helping the user with programming. You ma
y use URLs provided by the user in their messages or local files.\n\n# System\n - All text you output outside of tool use is displayed to the user. Output text to communicate with the user. You can use Github-flavored markdown for formatting, and will be rendered in a monospace font using the CommonMark specification.\n - Tools are executed in a user-selected permission mode. When you attempt to call a tool that is not automatically allowed by the user's permission mode or permission settings, the user will be prompted ...(truncated)
