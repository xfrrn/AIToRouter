# AI Router — 智能网络拓扑路由优化平台

## 概述

一个本地全栈 Web 应用，用户通过前端拖拽绘制网络拓扑，后端自动在 Mininet Docker 容器中创建网络、生成流量、调用 network-rl 强化学习模型推理最优路由路径，结果在拓扑图上高亮展示并附带对比数据。

同时集成 LLM Agent，支持自然语言驱动全流程：描述需求 → 生成拓扑 → 部署 → 推理 → 解读结果。

## 架构总览

```
┌──────────────────────┐     ┌──────────────────────────────────────────┐
│   前端 (单页 HTML)    │     │             后端 (FastAPI)                │
│                      │     │                                          │
│  ┌────────────────┐  │     │  ┌────────────────────────────────────┐  │
│  │  拓扑编辑器     │  │ POST │  │  POST /api/deploy                  │  │
│  │  + 连线属性    │──┼────►│  │  1. 拓扑 JSON → Mininet 脚本        │  │
│  │  + 部署按钮    │  │     │  │  2. Docker 创建 Mininet 容器         │  │
│  └────────────────┘  │     │  │  3. 运行网络 + 自动生成流量          │  │
│                      │     │  │  4. 拓扑 → GraphML                  │  │
│  ┌────────────────┐  │     │  │  5. network-rl 推理                 │  │
│  │  结果展示       │◄─┼─────│  │  6. 返回路径 + 指标                 │  │
│  │  路径高亮+表格  │  │     │  └────────────────────────────────────┘  │
│  └────────────────┘  │     │                                          │
│                      │     │  ┌────────────────────────────────────┐  │
│  ┌────────────────┐  │     │  │  Agent (LLM Function Calling)       │  │
│  │  Chat 面板      │◄─┼────►│  │  POST /api/chat                    │  │
│  └────────────────┘  │     │  │  ├─ generate_topology()             │  │
└──────────────────────┘     │  │  ├─ generate_traffic()              │  │
                              │  │  ├─ deploy_and_analyze()           │  │
                              │  │  └─ explain_results()              │  │
                              │  └────────────────────────────────────┘  │
                              │                                          │
                              │  ┌──────────┐  ┌──────────────────────┐  │
                              │  │ Mininet   │  │  network-rl (Python) │  │
                              │  │ Docker    │  │  Policy 推理          │  │
                              │  └──────────┘  └──────────────────────┘  │
                              └──────────────────────────────────────────┘
```

## 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| 前端 | 现有 `topology-editor.html` (原生 HTML/CSS/JS) | 增量改造，不动核心逻辑 |
| 后端 | Python 3.12+ / FastAPI | 与 network-rl 同语言，直接调用 |
| 容器 | Docker SDK for Python (`docker-py`) | 管理 Mininet 容器生命周期 |
| 模型 | network-rl (PPO Policy) | 本地 Python 导入，直接调用 forward |
| Agent | Claude API (Anthropic SDK) | Function calling 编排工具 |
| 异步 | FastAPI BackgroundTasks / asyncio | 部署任务异步执行 |

---

## 一、前端改造

### 1.1 连线属性扩展

现有连线只有起止点，需在属性面板中增加可编辑字段：

- `bandwidth` (Mbps)：默认 100，范围 1-10000
- `delay` (ms)：默认 5，范围 0-500

选中连线后，属性面板显示这些字段（类似设备属性编辑）。

### 1.2 新增工具栏按钮

- 「部署」按钮 (`btn-deploy`)：收集拓扑数据 POST 到 `/api/deploy`，显示进度
- 部署期间按钮禁用，显示 spinner 状态

### 1.3 结果展示

部署完成后在画布上展示结果：

- **路径高亮**：每条流的推荐路径用不同颜色加粗叠加在原有连线上方
- **流选择器**：下拉框切换查看不同流的高亮路径
- **结果表格**：右侧面板展示

| 流 ID | src → dst | bw_req | 模型路径 | 跳数 | 最大链路利用率 |
|--------|-----------|--------|----------|------|---------------|

### 1.4 Chat 面板

右下角浮动 chat 按钮，展开对话面板，用户用自然语言描述需求，Agent 逐步执行。

---

## 二、后端 (FastAPI)

### 2.1 项目结构

```
backend/
├── main.py              # FastAPI 入口、路由、CORS
├── agent/
│   └── orchestrator.py  # LLM Agent：工具定义 + 对话编排
├── mininet/
│   ├── manager.py       # Docker 容器生命周期
│   └── templates.py     # 拓扑 JSON → Mininet Python 脚本
├── traffic/
│   └── generator.py     # 自动流量生成
├── model/
│   └── inference.py     # network-rl Policy 推理封装
├── schemas/
│   └── models.py        # Pydantic 数据模型
└── requirements.txt
```

### 2.2 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/deploy` | 一次性流水线：拓扑 → Mininet → 流量 → 推理 → 结果 |
| `POST` | `/api/chat` | Agent 对话入口 |
| `GET` | `/api/status/{job_id}` | 查询异步任务状态 |
| `DELETE` | `/api/containers/{id}` | 清理 Mininet 容器 |

### 2.3 POST /api/deploy 流程

```
1. 接收 TopologyJSON (devices + connections + 连线属性)
2. 拓扑 → Mininet Python 脚本 (Jinja2 模板生成)
3. 拉取 Mininet Docker 镜像 (mnknowles/mininet 或自建)
4. 创建容器，挂载脚本，执行 mn -c && python topo.py
5. 自动生成流量矩阵 (基于拓扑节点对，bw ~ U[0,40])
6. 在容器内运行流量 (iperf / custom script)，采集链路利用率
7. 拓扑 → GraphML → network-rl Policy.forward()
8. 每条流返回：action, logits, selected_path, metrics
9. 销毁容器
10. 返回 DeploymentResult
```

### 2.4 数据模型 (schemas/models.py)

```python
class Device(BaseModel):
    id: str
    type: str          # router, switch, server, etc.
    x: float; y: float
    label: str
    ip: str = ""

class Connection(BaseModel):
    id: str
    from_: dict        # {devId, port}
    to: dict           # {devId, port}
    bandwidth: float = 100.0   # Mbps
    delay: float = 5.0         # ms

class TopologyJSON(BaseModel):
    devices: list[Device]
    connections: list[Connection]

class FlowResult(BaseModel):
    flow_id: int
    src: int; dst: int
    bw_req: float
    phi: float
    selected_path: list[int]
    hops: int
    max_link_utilization: float
    ospf_path: list[int] | None = None

class DeploymentResult(BaseModel):
    job_id: str
    flows: list[FlowResult]
    topology_nx: dict   # 序列化后的 networkx 图数据
```

---

## 三、Mininet 集成

### 3.1 Docker 镜像

使用社区维护的 Mininet Docker 镜像（如 `mnknowles/mininet`），或自建包含 Mininet + iperf + 自定义工具的镜像。

### 3.2 拓扑 → Mininet 脚本

Jinja2 模板，将前端的 devices + connections 转换为 Mininet Python 脚本：

```python
# 模板生成示例
from mininet.topo import Topo
class CustomTopo(Topo):
    def build(self):
        # 每个 device → addHost (带 IP)
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        # 每个 connection → addLink (带 bandwidth, delay)
        self.addLink(h1, h2, bw=100, delay='5ms')
```

### 3.3 流量注入

自动生成的流量矩阵通过 Mininet 主机的 `iperf` 命令注入。多流并发执行，采集完成后汇总链路利用率。

---

## 四、network-rl 推理集成

### 4.1 推理封装

```python
class InferenceEngine:
    def __init__(self, ckpt_path: str):
        self.policy = Policy(ckpt_path)

    def infer(self, G: nx.Graph, flows: list[Flow]) -> list[FlowResult]:
        # 1. 构建有向边索引 [2, E]
        # 2. 构建边特征 [E, 3] (delay_norm, util, bw_norm)
        # 3. 对每条流: x, metrics, K最短路径 → policy.forward()
        # 4. 更新边利用率 (流序列推理)
        # 5. 返回所有流的结果
```

### 4.2 拓扑映射

前端拓扑 editor 的设备/连线 → networkx Graph（节点 ID 映射为整数），连线属性 `bandwidth`/`delay` 映射为边属性。

---

## 五、Agent 设计

### 5.1 工具定义

| 工具 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `generate_topology` | 自然语言描述 | 拓扑 JSON | LLM 根据描述生成设备+连线+属性，前端加载到编辑器 |
| `generate_traffic` | 拓扑 + 场景描述 | 流量矩阵 | 生成结构化的 (src, dst, bw, phi) 列表 |
| `deploy_and_analyze` | 拓扑 + 流量 | DeploymentResult | 触发 Mininet 部署 + 模型推理 |
| `explain_results` | DeploymentResult | 自然语言 | 解读模型决策逻辑（为什么选这条路径） |

### 5.2 POST /api/chat 工作流

```
用户: "模拟电商三层架构，双11峰值流量"
  ↓
Agent 第1轮: generate_topology("电商三层架构...")
  → 返回拓扑 JSON → 前端加载到编辑器 → 用户确认/微调
  ↓
Agent 第2轮: generate_traffic(topology, "双11峰值流量...")
  → 返回流量矩阵 → 用户确认
  ↓
Agent 第3轮: deploy_and_analyze(topology, traffic)
  → 部署 Mininet → 跑流量 → network-rl 推理
  ↓
Agent 第4轮: explain_results(results)
  → 自然语言解读 → 前端展示
```

### 5.3 LLM 选择

Claude API (Anthropic SDK)，使用 `claude-sonnet-4-6`，启用 prompt caching 优化成本。

---

## 六、数据流完整路径

```
用户拖拽设备/连线 → 设置连线属性(bw, delay)
  → 点击"部署" (或通过 Chat 说需求)
  → 拓扑 JSON POST 到后端
  → Jinja2 生成 Mininet .py 脚本
  → Docker 创建 Mininet 容器
  → 容器内: mn + python topo.py 创建网络
  → 自动生成流量 → iperf 注入
  → 采集链路利用率
  → 拓扑转为 networkx Graph (GraphML)
  → network-rl Policy.forward() 逐流推理
  → 汇总 FlowResult[]
  → 返回前端
  → 前端高亮路径 + 结果表格
  → Agent 解读 (如果是 chat 模式)
  → 清理 Docker 容器
```

## 七、实现顺序

| 阶段 | 内容 | 预估 |
|------|------|------|
| Phase 1 | 后端骨架：FastAPI + schemas + Mininet 容器管理 | 核心 |
| Phase 2 | Mininet 脚本模板 + 流量生成 + 推理集成 | 核心 |
| Phase 3 | 前端改造：连线属性 + 部署按钮 + 结果展示 | 核心 |
| Phase 4 | Agent 集成：function calling + Chat 面板 | 增强 |
| Phase 5 | 多流并发优化 + 错误处理 + 状态管理 | 完善 |

## 八、关键决策记录

- **前端不重写**：现有 topology-editor.html 功能完善，只做增量
- **流量自动生成**：用户不需手动配流量，降低使用门槛
- **Agent 作为增强层**：直接 `/api/deploy` 可用，Agent 是可选的便利入口
- **network-rl 本地调用**：作为 Python 模块导入，不通过 HTTP，零序列化开销
- **容器即用即毁**：每次部署新建容器，结束后销毁，不留状态
