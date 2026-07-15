# AI Assistant Loop

循环式 AI 助手：本地 MinerU 论文库门控 RAG + Plan–Act–Reflect 工具环，可扩展 ToolSpec / TaskSpec 与 MCP（本地 stdio / 远程 Streamable HTTP）。

本地开发文档与过程日志（`Development/`）、语料（`raw/`）、密钥（`setting/`）、运行时数据（`data/`）与二进制 MCP 不在本仓库备份范围内。

## 当前已有功能

### 知识库（Phase 0–1）

- 读取 `raw/pdf2md/` 下 MinerU 产出（`full.md` + `*_content_list.json` + `images/`）
- 父子层级多模态切块（文本 / 表 / 公式 / 图）
- Qwen `text-embedding-v4`（API）写入本地 Qdrant（`data/qdrant_local`，无需 Docker）
- Retrieval Gate：`none` | `pinned`（钉论文注入 broker）| `retrieve`（非每轮盲检索）
- 语料就绪：`config/corpus.yaml` + `/health.corpus`；搜索主备 AnySearch → Tavily（TaskSpec.`search_stack`）

### 智能体环（Phase 2 + 协作基础）

- 环路：Classify → **Orchestrate** → Gate → Broker → Plan → Act → Observe → Reflect → Compress?
- **编排器**分解 `work_items[]`（一对多；后续环节可打回前序补充）；**不是**每次跑遍所有任务包
- MATLAB MCP 仅服务编码类 `matlab_assist` work_item（懒连接）；普通问答不会拉起 MATLAB
- 内置工具：`kb_retrieve`、`tavily_search`、`fetch_page`、`run_local_script`
- 会话持久化、中断、修正（amend 可带 `target_work_item_id`）、检查点、FileGuard
- **异步 HITL**：`POST /agent/run` 立即返回（`sync` 可选）；轮询 trajectory / SSE `events/stream`
- `/ui` 人读：当前 work_item、分项/终答、证据卡片；JSON 收进 Debug
- **Bootstrap**：`POST /agent/bootstrap` / `scripts/bootstrap_first_run.py`（语料就绪 → 固定 Bufort 问答）
- 环路：Classify → Orchestrate → Gate → Broker → Plan → Act → Observe → Reflect
### 插件与 MCP（Phase 2.5 + P0）

- **ToolSpec / TaskSpec**：YAML 发现注册；任务包 `research_qa` / `web_research` / `general_qa` / `matlab_assist`
- **MCP**
  - `stdio` 持久会话（已验证 MATLAB：`mcp/matlab-mcp-core-server-win64.exe`）
  - `streamable_http`（如 AnySearch `https://api.anysearch.com/mcp`）
  - `fake` 单测；不支持旧版 HTTP+SSE
- **证据链**：`registry.call → harvest → evidence → reflect`（含 `source_type=mcp`）
- AnySearch 官方 skill：`setting/skills/AnySearch/`；探针 `scripts/probe_anysearch.py`

### 模型与密钥

- 推理：ComiRouter（`src/llm/client.py`，httpx + Bearer）
- 密钥：`setting/API-key/`（**文件内容**为 key，勿提交 git）
- 多模态对话 / 换重模型前需人工确认模型 id

## 仓库结构

```text
AI-Assistant_loop/
├── README.md                 # 本文件
├── pyproject.toml            # 依赖（含 mcp）
├── config/                   # paths / models / agent / tools / mcp_servers …
├── src/                      # 核心代码（ingest / agent / tools / api …）
├── scripts/                  # CLI / smoke / Tools 实现
├── web/hitl/                 # 最小 HITL 控制台
└── docker-compose.yml        # 可选辅助
# 本地不备份：setting/ Development/ raw/ mcp/ data/ tests/
```

## 环境

```powershell
$py = "E:\application\miniforge3\envs\copilot-agent\python.exe"
& $py -m pip install -e .
```

## 系统启动（首次成功路径）

1. **安装**：见上 `pip install -e .`（环境 `copilot-agent`）。  
2. **密钥**：`setting/API-key/` 放置 ComiRouter / embedding / AnySearch 等 key 文件。  
3. **默认语料**（`config/corpus.yaml` → Bufort）：  
   `& $py scripts\ingest_one.py --doc Bufort --embed`  
4. **就绪检查**：`GET /health` 看 `corpus.ok` / `ready_for_bootstrap`，或  
   `& $py scripts\bootstrap_first_run.py`（异步跑固定 KB 问答并轮询至结束）。  
5. **HITL UI**：  
   `& $py -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000`  
   → http://127.0.0.1:8000/ui （Run 默认异步；可 Bootstrap / Interrupt / Amend）  
6. **钉论文**：查询写 `钉住 Bufort …`，或 API `pinned_docs: ["Bufort"]`（跳过向量检索，注入 full.md）。

夹带验收：`scripts\smoke_async_hitl.py`、`scripts\smoke_pinned_broker.py`。

## 常用命令

```powershell
$py = "E:\application\miniforge3\envs\copilot-agent\python.exe"

# 语料：切块 / 嵌入 / 检索冒烟
& $py scripts\ingest_one.py --doc Bufort
& $py scripts\ingest_one.py --doc Bufort --embed --recreate-collection
& $py scripts\smoke_retrieve.py "radio propagation GNN"
& $py scripts\smoke_llm.py

# 智能体
& $py scripts\run_agent.py run "用知识库解释 Bufort 的 GNN 传播建模思路"
& $py scripts\run_agent.py status --session <id>
& $py scripts\run_agent.py stop --session <id>
& $py scripts\run_agent.py amend --session <id> --text "目标改成只要三句话"
& $py scripts\run_agent.py resume --session <id>

# 协作编排旅程（多 work_item / supplement / amend target；禁口算）
& $py scripts\smoke_work_items_journey.py
# 深度研究：地缘/贸易/美指 ↔ 金铜期货（任务链 + 实检索准确度/时效）
& $py scripts\smoke_macro_commodities_journey.py

# MCP / 检索探针（不经过完整 agent 环）
& $py scripts\test_matlab_mcp_launch.py --hold 30
& $py scripts\probe_anysearch.py
& $py scripts\smoke_n1_mcp_io.py      # AnySearch → evidence
& $py scripts\smoke_n1_agent_mcp.py   # AnySearch + MATLAB via act/observe

# HITL WebUI + API（异步 run）
& $py -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
# 打开 http://127.0.0.1:8000/ui
# POST /agent/run | /agent/bootstrap ； GET .../trajectory|events|events/stream
& $py scripts\bootstrap_first_run.py
& $py scripts\smoke_async_hitl.py
& $py scripts\smoke_pinned_broker.py

# 检索评测 / 批量嵌入
& $py scripts\eval_retrieve.py
# & $py scripts\ingest_all.py --embed

# 测试
& $py Development\tests\run_p0_checks.py
& $py -m pytest tests -q
```

## 扩展入口（摘要）

| 要加什么 | 怎么做 |
|----------|--------|
| 内置工具 | `scripts/Tools/x.py` + `get_tool_spec()` → 单测 → `config/tools_enabled.yaml` |
| MCP 服务 | `config/mcp_servers.yaml`（`stdio` / `streamable_http`）→ `enabled: true` |
| 任务类型 | `src/tasks/packs/` + `get_task_spec()` → `tools_enabled.yaml` 的 `tasks:` |
| 允许 MCP 工具 | TaskSpec `allowed_tools` 写 `mcp:<server>` 或 `mcp_<server>_*` |

更完整的约定与状态表以 [Development/Design.md](Development/Design.md) 为准。
