# AI Router

智能网络拓扑路由优化平台。用户在前端拖拽绘制网络拓扑，后端自动调用 network-rl 强化学习模型推理最优路由路径，结果在拓扑图上高亮展示并附带与 OSPF 基线的对比数据。

同时集成 LLM Agent（Claude API），支持自然语言驱动全流程。

## 架构

```
浏览器 (topology-editor.html)  ←→  FastAPI 后端  ←→  Mininet Docker / network-rl 模型
        │                              │
        └── 拖拽画拓扑                    ├── /api/infer  推理（无需 Docker）
        └── 点部署看结果                  ├── /api/deploy 完整流水线
        └── Chat 自然语言交互             └── /api/chat   Agent 对话
```

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 启动后端
cd backend && uv run python run.py

# 3. 打开浏览器
# http://localhost:8000
```

打开后即可拖拽设备、连线、设置带宽/时延，点击「部署」查看推理结果。

**无需 Docker、无需 GPU、无需模型权重** — 没有网络模型时会自动使用 OSPF 最短路径基线，前端流程完全可用。

## 使用方式

### 拓扑编辑器

| 操作 | 说明 |
|------|------|
| 左侧拖拽 | 拖拽设备到画布 |
| 选择模式 (S) | 点击选中设备/连线，右侧面板编辑属性 |
| 连线模式 (C) | 点击设备端口连线 |
| 连线属性 | 选中连线后可编辑带宽 (Mbps) 和时延 (ms) |
| 删除 (Del) | 删除选中设备或连线 |
| 导出 | 导出拓扑为 JSON |

### 部署与推理

点击工具栏 **「▶ 部署」** 按钮，后端自动：
1. 将拓扑转换为网络图
2. 自动生成流量需求
3. 运行 network-rl 模型推理（如有 checkpoint）或 OSPF 基线
4. 在下方面板展示结果：路径高亮 + 对比表格

### Chat 面板

点击右下角 **✈ 按钮** → 打开 AI 助手对话面板。

**API 设置**：点击 ⚙ 齿轮，填入 API Key 和 Base URL（可选），自动保存到浏览器 localStorage。

示例：
> "帮我搭建一个三层电商架构，两台LB，三台App服务器，两台数据库主从"

Agent 会生成拓扑并加载到编辑器中，后续可手动调整后部署。

## API 端点

| 端点 | 需要 | 说明 |
|------|------|------|
| `GET /` | - | 拓扑编辑器页面 |
| `POST /api/infer` | - | 推理最优路由（无模型时用 OSPF 基线） |
| `POST /api/deploy` | Docker | 完整流水线：Mininet 部署 + 推理 |
| `POST /api/chat` | API Key | Agent 对话 |
| `GET /api/docs` | - | Swagger API 文档 |

## 可选：启用完整流水线

### Docker + Mininet

```bash
# 安装 Docker SDK
uv pip install docker

# 拉取 Mininet 镜像
docker pull iwaseyusuke/mininet
```

### network-rl 模型推理

```bash
# 安装 ML 依赖
uv sync --extra ml

# 训练模型权重
cd 模型项目/network-rl
pip install -e .
XCHIRL_TOPO=BQD python train/ppo/train.py
```

训练完成后将 checkpoint 放到 `模型项目/network-rl/runs/FILM_PPO/best.pt`，后端自动加载。

### Agent 对话

在 Chat 面板 ⚙ 设置中填入 Anthropic API Key，或设置环境变量 `ANTHROPIC_API_KEY`。

## 项目结构

```
├── topology-editor.html    # 前端（单页应用）
├── pyproject.toml          # 项目依赖（uv）
├── start.sh                # 启动脚本
├── backend/
│   ├── main.py             # FastAPI 入口
│   ├── run.py              # 开发服务器
│   ├── schemas/models.py   # Pydantic 数据模型
│   ├── mininet/            # Mininet Docker 管理
│   │   ├── manager.py      # 容器生命周期
│   │   └── templates.py    # 拓扑 → Mininet 脚本
│   ├── traffic/generator.py # 自动流量生成
│   ├── model/inference.py   # network-rl 推理封装
│   └── agent/orchestrator.py # LLM Agent 编排
└── 模型项目/network-rl/     # RL 路由模型（子项目）
```
