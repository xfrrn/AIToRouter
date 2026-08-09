# AI Router 项目工作流详解

## 1. 项目概览

AI Router 是一个**智能网络拓扑路由优化平台**。用户在前端拖拽绘制网络拓扑，后端自动调用强化学习模型推理最优路由路径，结果在拓扑图上高亮展示并与 OSPF 基线对比。

核心价值：**用训练好的 RL 策略替代传统 OSPF 最短路径，在多条候选路径中选择使全局链路利用率更均衡的路由方案**。

### 降级链

系统有三层降级保障，即使没有 Docker 和训练模型也能运行：

```
Docker 可用？── 是 → Mininet 实测网络性能 → RL 模型推理 → 返回结果
                └ 否 → 跳过实测 ──────────→ RL 模型推理 → 返回结果
                                            └ 无模型？→ OSPF 最短路径 → 返回结果
```

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│  浏览器 (topology-editor.html)                                │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ SVG 画布  │  │ 结果面板      │  │ AI 助手聊天面板         │ │
│  │ 拖拽/连线 │  │ 流量表+路径高亮│  │ 自然语言→拓扑/流量/分析 │ │
│  └─────┬────┘  └──────┬───────┘  └───────────┬────────────┘ │
└────────┼──────────────┼───────────────────────┼──────────────┘
         │ POST          │                      │ POST
         │ /api/deploy   │                      │ /api/chat
         ▼               ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│  FastAPI 后端 (backend/main.py)                               │
│                                                              │
│  /api/deploy ──┬──► MininetManager (Docker 容器管理)          │
│                │     ├─ 生成 Mininet 脚本 (templates.py)      │
│                │     ├─ 容器内运行 iperf 测吞吐 + ping 测延迟  │
│                │     └─ 返回实测数据 {measured_bw, link_rtts}  │
│                │                                             │
│                ├──► 实测 RTT 回填边权重（70%实测 + 30%配置）    │
│                │                                             │
│                └──► InferenceEngine (model/inference.py)      │
│                      ├─ 有模型 → GNN 策略推理选路              │
│                      └─ 无模型 → OSPF 跳数最短路径             │
│                                                              │
│  /api/infer ─────► 同上但跳过 Docker 步骤                     │
│                                                              │
│  /api/chat ──────► AgentOrchestrator (agent/orchestrator.py)  │
│                      ├─ generate_topology (LLM 生成拓扑)      │
│                      ├─ generate_traffic  (LLM 生成流量)      │
│                      ├─ deploy_and_analyze (调用推理引擎)      │
│                      └─ explain_results   (LLM 解释结果)      │
│                                                              │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐ │
│  │ Traffic      │  │ InferenceEngine │  │ Agent            │ │
│  │ Generator    │  │ (XCHiRL GNN)    │  │ Orchestrator     │ │
│  └──────────────┘  └─────────────────┘  └──────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Docker 容器: iwaseyusuke/mininet                            │
│  ┌──────────────────────────────────────────────────────────┐│
│  │  CustomTopo → 启动虚拟网络                               ││
│  │  → iperf 逐流测实际吞吐 (measured_bw)                     ││
│  │  → ping 并行测链路 RTT (link_rtts)                       ││
│  │  → 输出 MININET_RESULTS:{JSON}                          ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 前端：拓扑编辑器

`topology-editor.html` 是一个纯 HTML/CSS/JS 单页应用，由 FastAPI 在 `GET /` 时提供。

### 3.1 设备与画布

| 功能 | 操作 |
|------|------|
| 添加设备 | 从左侧设备面板拖拽到 SVG 画布 |
| 连线 | 切换到连线模式 (C)，点击源端口 → 目标端口 |
| 选择编辑 | 选择模式 (S) 下点击设备/连线，右侧面板编辑属性 |
| 连线属性 | 选中连线后可设 **带宽 (Mbps)** 和 **时延 (ms)** |
| 缩放/平移 | 鼠标滚轮缩放，空白处拖拽平移 |
| 撤销/重做 | Ctrl+Z / Ctrl+Y |
| 导入/导出 | 导出为 JSON / 导入 JSON 恢复拓扑 |

10 种设备类型：路由器、交换机、防火墙、服务器、终端、数据库、负载均衡、云服务、无线AP、打印机。

### 3.2 部署按钮

点击「▶ 部署」按钮后：

1. 前端收集当前画布状态 `{devices, connections}`
2. 发送 `POST /api/deploy`，按钮变为「部署中...」
3. 后端完成推理后返回 `DeploymentResult`
4. 前端调用 `showResults()` 展示结果：
   - 底部滑出结果面板，显示每条流的详细数据表格
   - 画布上用彩色贝塞尔曲线高亮选中流的路径
   - 流选择器下拉框可切换查看单条流或全部
   - Mininet 徽章显示「Mininet 实测」或「仅模型推理」

### 3.3 AI 助手面板

右侧可折叠聊天面板，通过 `POST /api/chat` 与 LLM Agent 交互：

- **生成拓扑**：用自然语言描述网络需求，Agent 调用 `generate_topology` 生成 TopologyJSON 并加载到画布
- **生成流量**：描述流量场景，Agent 生成匹配的流量需求
- **部署分析**：Agent 触发推理引擎，结果直接展示在结果面板
- **解释结果**：Agent 对比模型路径 vs OSPF 路径，用自然语言解释路由选择原因

需在齿轮设置中配置 API Key 和可选的 Base URL（兼容 OpenAI API 格式）。

---

## 4. 后端 API 详解

### 4.1 数据模型 (`schemas/models.py`)

```
TopologyJSON
  ├── devices: list[Device]          # id, type, x, y, label, ip
  └── connections: list[Connection]  # id, from, to, bandwidth, delay

DeploymentResult
  ├── job_id, status
  ├── flows: list[FlowResult]        # 每条流的推理结果
  │     ├── flow_id, src, dst, bw_req, phi
  │     ├── selected_path             # RL 模型选中的路径（节点索引列表）
  │     ├── hops, max_link_utilization
  │     ├── ospf_path                 # OSPF 最短路径（对比基线）
  │     └── measured_bw               # iperf 实测吞吐（仅 Mininet 模式）
  ├── topology_edges                  # 每条边的带宽、延迟、利用率
  ├── mininet_used: bool              # 是否走了 Mininet 实测
  ├── mininet_flow_results            # iperf 实测数据
  └── mininet_link_rtts              # ping 实测链路延迟
```

### 4.2 API 端点

| 端点 | 方法 | 需求 | 说明 |
|------|------|------|------|
| `/` | GET | — | 返回拓扑编辑器页面 |
| `/api/health` | GET | — | 返回 Docker 可用性和模型加载状态 |
| `/api/infer` | POST | — | 纯推理，不启动 Docker |
| `/api/deploy` | POST | Docker（可选） | 完整流水线：Mininet + 推理 |
| `/api/chat` | POST | API Key | LLM Agent 对话 |
| `/api/status/{job_id}` | GET | — | 查询部署任务状态 |
| `/api/containers/{id}` | DELETE | Docker | 清理指定容器 |

---

## 5. 核心流程详解

### 5.1 部署流程 (`POST /api/deploy`)

这是系统最核心的端到端流程：

```
前端拓扑 JSON
     │
     ▼
① build_nx_graph(topology)
   将 TopologyJSON → NetworkX Graph
   每条边带 {bandwidth, delay} 属性
   返回 (G, id_to_idx) 映射
     │
     ▼
② generate_flows(N, seed=42)
   根据节点数自动生成流量需求
   每条流: {flow_id, src, dst, bw_req, phi, duration}
   流量数 = min(N×3, 50)
   bw_req ~ U[0.5, 40] Mbps, phi ~ U[0, 1]
     │
     ▼
③ [如果 Docker 可用] Mininet 实测
   ├─ generate_mininet_script(topology)
   │   生成 Mininet Python 脚本：
   │   - CustomTopo 类定义拓扑
   │   - run_iperf_flows() 逐流测吞吐
   │   - measure_link_rtt() 并行 ping 测延迟
   │
   ├─ Docker 容器创建与运行
   │   docker run iwaseyusuke/mininet (privileged)
   │   挂载脚本 + flows.json
   │   python3 /tmp/topo/topo.py
   │
   ├─ 解析 MININET_RESULTS JSON
   │   {flow_results: [{measured_bw}], link_rtts: {"h1-h2": 0.05}}
   │
   └─ 实测 RTT 回填图边权重
      new_delay = 0.7 × measured_rtt + 0.3 × configured_delay
      (70%实测 + 30%配置，避免异常值)
     │
     ▼
④ InferenceEngine.infer(G, flows)
   RL 模型推理（详见 5.2）
   无模型时退化为 OSPF 基线
     │
     ▼
⑤ _build_result()
   组装 DeploymentResult 返回
```

### 5.2 RL 模型推理 (`InferenceEngine.infer`)

推理引擎封装了 XCHiRL 框架训练的 GNN 策略模型，采用 P4 模式（metrics_dim=2）：

**模型架构：**

```
输入图 (N 节点, E 条边)
     │
     ▼
FILMGNNEncoder
  ├─ 节点特征 [N, 2]: [is_src, is_dst]
  ├─ 边特征 [2E, 3]: [delay_norm, utilization, bw_norm]
  ├─ 流指标 [2]: [phi, bw_req_norm]
  └─ 多层 TransformerConv + FiLM 调制
     │
     ▼
PathPooler
  └─ LSTM 编码每条候选路径的节点嵌入序列
     │
     ▼
KPathScorer
  └─ MLP 打分 → Categorical logits → argmax 选路径
```

**推理步骤：**

1. **图预处理**：将节点重新编号为连续整数 [0, N-1]，每条无向边拆为两条有向边
2. **特征归一化**：
   - delay: `(d - 10.5) / 5.5`（训练分布 U[1,20]）
   - bandwidth: `(bw - 65.0) / 20.2`（训练分布 U[30,100]）
3. **候选路径生成**：对每条流 (src, dst)，用 `nx.shortest_simple_paths()` 生成 K=16 条候选路径
4. **逐流决策**：
   - 构建当前图的张量（节点特征、边特征、候选路径、流指标）
   - 模型前向传播 → 选出最优路径
   - 更新选中路径上每条边的利用率：`util += bw_req / capacity`
   - 同时计算 OSPF 最短路径作为基线对比
5. **输出**：每条流的选中路径、跳数、最大链路利用率、OSPF 对比路径

**关键设计**：模型逐流做决策，每次选路后更新边利用率，后续流能看到前面流造成的拥塞——这使模型能做出全局更优的负载均衡选择。

### 5.3 流量生成 (`generate_flows`)

自动生成匹配 RL 训练分布的流量需求：

| 参数 | 分布 | 说明 |
|------|------|------|
| src, dst | 随机（src ≠ dst） | 节点索引 |
| bw_req | U[0.5, 40.0] Mbps | 带宽需求 |
| phi | U[0, 1] | QoS 敏感度：低=延迟敏感，高=带宽敏感 |
| duration | U[2, 4] 秒 | iperf 测试持续时间 |
| 流量数 | min(N×3, 50) | 节点数 × 3，上限 50 |

### 5.4 Mininet 实测 (`MininetManager.deploy`)

当 Docker 可用时，系统会在 Mininet 容器中实测网络性能：

**容器生命周期：**

```
1. _ensure_image()        → 检查/拉取 iwaseyusuke/mininet 镜像
2. generate_mininet_script() → 生成包含拓扑定义+iperf+ping 的 Python 脚本
3. 写入 tmpdir/            → topo.py + flows.json
4. docker run --privileged → 创建容器，挂载 tmpdir
5. container.exec_run()    → 运行脚本，收集 stdout
6. _parse_results()        → 从 stdout 提取 MININET_RESULTS:{JSON}
7. stop_and_remove()       → 清理容器
8. cleanup_tmpdir()        → 清理临时目录
```

**实测指标：**

| 指标 | 方法 | 说明 |
|------|------|------|
| measured_bw | iperf v2 TCP 测试 | 每条流从 src 到 dst 的实际吞吐量（Mbps） |
| link_rtts | ping -c 3 并行 | 每条链路的平均 RTT（ms），所有链路同时 ping |

**实测数据的使用：**
- `measured_bw` 写入 `FlowResult.measured_bw`，在前端表格中展示
- `link_rtts` 以 70%/30% 比例与配置延迟混合，更新图的边权重后再做推理，让模型基于更真实的网络状态做决策

### 5.5 OSPF 基线 (`_run_ospf_baseline`)

当模型不可用时的降级方案：

- 用 `nx.shortest_path(G, src, dst)` 按跳数选最短路径
- 逐流更新边利用率
- 计算每条流的最大链路利用率

这是传统网络的标准路由方式，也是 RL 模型需要超越的基线。

---

## 6. LLM Agent 编排 (`AgentOrchestrator`)

基于 OpenAI 兼容 API 的 function calling Agent，提供 4 个工具：

| 工具 | 触发场景 | 作用 |
|------|----------|------|
| `generate_topology` | 用户描述网络需求 | LLM 生成 TopologyJSON，加载到画布 |
| `generate_traffic` | 用户描述流量场景 | LLM 生成匹配场景的流量流数组 |
| `deploy_and_analyze` | 用户请求部署分析 | 调用推理引擎，结果展示在结果面板 |
| `explain_results` | 用户询问结果含义 | LLM 对比模型路径 vs OSPF，解释路由选择 |

**交互流程：**

```
用户消息 + 当前拓扑状态
     │
     ▼
LLM 判断意图 → 调用对应工具
     │
     ├── generate_topology → LLM 子调用 → TopologyJSON → action: "load_topology"
     ├── generate_traffic  → LLM 子调用 → flows[]
     ├── deploy_and_analyze → on_deploy() → DeploymentResult → action: "show_results"
     └── explain_results  → LLM 子调用 → 自然语言解释
     │
     ▼
ChatResponse {reply, action, topology?, results?}
```

前端根据 `action` 字段执行对应操作：加载拓扑到画布或展示推理结果。

---

## 7. RL 模型训练（`模型项目/network-rl/`）

### 7.1 训练框架

基于 XCHiRL（自定义 RL 框架），使用 PPO 算法：

- **状态空间**：网络拓扑图（节点特征 + 边特征 + 流指标）
- **动作空间**：离散 K-path 选择（K=16 条候选路径中选一条）
- **奖励**：稀疏奖励，所有流放置完成后根据全局链路利用率计算

### 7.2 训练配置

```bash
cd 模型项目/network-rl
pip install -e .
XCHIRL_TOPO=BQD python train/ppo/train.py
```

训练拓扑：BQD（小型）、GEANT（中型）、ABILENE（大型）。

### 7.3 模型组件

| 组件 | 说明 |
|------|------|
| `FILMGNNEncoder` | FiLM 调制的 GNN 编码器，4 层 TransformerConv，条件化为 (phi, bw_req) |
| `PathPooler` | LSTM 编码候选路径的节点嵌入序列 → 路径表示向量 |
| `KPathScorer` | MLP 评分器，输出 K 条路径的 logits → argmax 选择 |

### 7.4 关键实验结论

- GNN 编码器对路由性能至关重要
- 稀疏奖励优于奖励塑形
- FiLM 调制提供边际改进
- K=16 是候选路径数的良好默认值
- PPO 在 3 种拓扑上以 ≥8σ 显著性优于 OSPF/ECMP/CSPF/RANDOM 基线

---

## 8. 部署与运行

### 8.1 基础运行（无需 Docker、GPU、模型权重）

```bash
uv sync
cd backend && uv run python run.py
# 浏览器打开 http://localhost:8000
```

系统自动降级为 OSPF 最短路径基线，前端功能完全可用。

### 8.2 启用 Mininet 实测

```bash
uv pip install docker
docker pull iwaseyusuke/mininet
```

Docker 可用时，`/api/deploy` 会先在 Mininet 容器中实测网络性能，再基于实测数据推理。

### 8.3 启用 RL 模型推理

```bash
uv sync --extra ml    # 安装 torch, torchrl 等依赖
# 训练或获取 checkpoint
# 放到 runs/FILM_PPO/best.pt 或 模型项目/network-rl/runs/FILM_PPO/best.pt
```

### 8.4 启用 AI 助手

在 Chat 面板 ⚙ 设置中填入 OpenAI 兼容 API Key 和 Base URL（可选）。

---

## 9. 项目文件结构

```
AI_ROUTER/
├── topology-editor.html          # 前端单页应用（拖拽编辑器 + 结果展示 + Chat）
├── README.md                     # 项目说明
├── pyproject.toml                # 依赖管理（uv）
├── start.sh                      # 开发启动脚本
├── complex-topology.json         # 示例拓扑导出
│
├── backend/
│   ├── main.py                   # FastAPI 入口：路由、部署、推理、Chat
│   ├── run.py                    # 开发服务器 (uvicorn --reload)
│   ├── schemas/models.py         # Pydantic 数据模型
│   ├── mininet/
│   │   ├── manager.py            # Docker Mininet 容器生命周期管理
│   │   └── templates.py          # 拓扑 → Mininet Python 脚本生成
│   ├── traffic/
│   │   └── generator.py          # 自动流量需求生成
│   ├── model/
│   │   └── inference.py          # XCHiRL 策略模型推理封装
│   ├── agent/
│   │   └── orchestrator.py       # LLM Agent 编排（function calling）
│   └── test_integration.py       # 集成测试（无 Docker/torch 依赖）
│
└── 模型项目/network-rl/           # RL 路由模型子项目
    ├── xchirl/                   # 核心包
    │   ├── modules/              # GNN编码器、路径池化器、评分器
    │   ├── envs/                 # 路由环境 (PPO + SAC)
    │   ├── baselines/            # OSPF, ECMP, ILP 基线
    │   └── utils/                # 组件工厂
    ├── train/ppo/train.py        # PPO 训练循环
    └── experiments/              # 实验报告 (P0-P7)
```
