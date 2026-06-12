# AI Router Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local full-stack platform where users draw network topologies in a browser, deploy them to Mininet Docker containers, run traffic, and get RL-optimized routing paths with visual results.

**Architecture:** Single-page HTML frontend (extending existing topology-editor.html) talks to a FastAPI backend. The backend manages Docker containers running Mininet, auto-generates traffic, and calls the network-rl Python model for inference. An optional LLM Agent (Claude API) provides natural-language-driven orchestration.

**Tech Stack:** Python 3.12+ / FastAPI / docker-py / networkx / Jinja2 / Anthropic SDK / existing topology-editor.html (vanilla JS)

**Environment:** Python 3.13.5, Docker 28.5.1, Windows 11. network-rl package at `模型项目/network-rl/`.

---

### Task 1: Backend project scaffold

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/schemas/__init__.py`
- Create: `backend/schemas/models.py`
- Create: `backend/__init__.py`

- [ ] **Step 1: Create backend directory and requirements.txt**

```bash
mkdir -p backend/schemas backend/mininet backend/traffic backend/model backend/agent
```

- [ ] **Step 2: Write requirements.txt**

```txt
# backend/requirements.txt
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
docker>=7.1.0
networkx>=3.4
jinja2>=3.1
anthropic>=0.49.0
```

- [ ] **Step 3: Write schemas/models.py**

```python
# backend/schemas/__init__.py
# backend/schemas/models.py
from __future__ import annotations

from pydantic import BaseModel, Field


class Device(BaseModel):
    id: str
    type: str  # router, switch, firewall, server, laptop, database, lb, cloud, wifi, printer
    x: float
    y: float
    label: str
    ip: str = ""


class ConnectionEndpoint(BaseModel):
    devId: str
    port: str  # top, right, bottom, left


class Connection(BaseModel):
    id: str
    from_: ConnectionEndpoint = Field(alias="from")
    to: ConnectionEndpoint
    bandwidth: float = 100.0  # Mbps
    delay: float = 5.0  # ms

    class Config:
        populate_by_name = True


class TopologyJSON(BaseModel):
    devices: list[Device]
    connections: list[Connection]


class FlowResult(BaseModel):
    flow_id: int
    src: int
    dst: int
    bw_req: float
    phi: float
    selected_path: list[int]
    hops: int
    max_link_utilization: float
    ospf_path: list[int] | None = None


class DeploymentResult(BaseModel):
    job_id: str
    status: str  # "running" | "completed" | "failed"
    flows: list[FlowResult] = []
    error: str | None = None
    topology_nodes: list[int] = []
    topology_edges: list[dict] = []  # [{src, dst, bandwidth, delay, utilization}]


class ChatRequest(BaseModel):
    message: str
    topology: TopologyJSON | None = None  # current editor state, if any


class ChatResponse(BaseModel):
    reply: str
    action: str | None = None  # "load_topology" | "show_results" | None
    topology: TopologyJSON | None = None
    results: DeploymentResult | None = None
```

- [ ] **Step 4: Commit**

```bash
git add backend/ && git commit -m "feat: add backend project scaffold with Pydantic schemas"
```

---

### Task 2: Mininet script template

**Files:**
- Create: `backend/mininet/__init__.py`
- Create: `backend/mininet/templates.py`

- [ ] **Step 1: Write the Mininet topology template generator**

```python
# backend/mininet/templates.py
"""Convert frontend TopologyJSON to a Mininet Python script string."""

from __future__ import annotations

from schemas.models import TopologyJSON

MININET_SCRIPT_TEMPLATE = '''\
#!/usr/bin/env python3
"""Auto-generated Mininet topology."""
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.cli import CLI
from mininet.log import setLogLevel
import json, sys, time, os

class CustomTopo(Topo):
    def build(self):
        {% for dev in devices %}
        {{ dev.hostname }} = self.addHost('{{ dev.hostname }}', ip='{{ dev.ip }}')
        {% endfor %}
        {% for conn in connections %}
        self.addLink({{ conn.src }}, {{ conn.dst }}, bw={{ conn.bw }}, delay='{{ conn.delay }}ms')
        {% endfor %}

def run_iperf_flows(flows_file):
    """Run iperf flows from a JSON file and collect results."""
    with open(flows_file) as f:
        flows = json.load(f)

    results = []
    for flow in flows:
        src = flow["src"]
        dst = flow["dst"]
        bw = flow["bw_req"]
        duration = flow.get("duration", 5)

        # Start iperf server on dst
        dst_host = net.get(dst)
        dst_host.cmd(f"iperf -s -p 5001 &")

        # Run iperf client on src
        src_host = net.get(src)
        out = src_host.cmd(f"iperf -c {dst_host.IP()} -p 5001 -b {bw}M -t {duration} 2>&1")

        # Parse bandwidth from iperf output
        results.append({"flow_id": flow["flow_id"], "src": src, "dst": dst, "output": out})

        # Kill iperf server
        dst_host.cmd("pkill iperf")

    return results

def collect_link_util(net, G_data):
    """Collect link utilization from the running network."""
    edges = []
    for link_data in G_data["edges"]:
        u, v = link_data["src"], link_data["dst"]
        # Get interface stats from both sides
        try:
            src_host = net.get(u)
            # Use tc to read current utilization (approximate from iperf runs)
            edges.append({"src": u, "dst": v, "utilization": 0.0})
        except:
            edges.append({"src": u, "dst": v, "utilization": 0.0})
    return edges

if __name__ == "__main__":
    setLogLevel("info")
    topo = CustomTopo()
    net = Mininet(topo=topo)
    net.start()

    # Wait for network to stabilize
    time.sleep(2)

    # Ping all hosts to verify connectivity
    net.pingAll()

    # Run flows if provided
    flows_file = os.environ.get("FLOWS_FILE", "")
    if flows_file and os.path.exists(flows_file):
        results = run_iperf_flows(flows_file)
        print("FLOW_RESULTS:", json.dumps(results))

    # Output topology info for the host
    print("TOPO_READY")
    sys.stdout.flush()

    # Keep running for CLI or script control
    CLI(net)
    net.stop()
'''


def generate_mininet_script(topology: TopologyJSON) -> str:
    """Generate a Mininet Python script from a TopologyJSON."""

    # Map device IDs to Mininet-safe hostnames
    hostname_map: dict[str, str] = {}
    for i, dev in enumerate(topology.devices):
        hostname_map[dev.id] = f"h{i + 1}"

    # Build device entries for the template
    devices = []
    for i, dev in enumerate(topology.devices):
        hostname = hostname_map[dev.id]
        ip = dev.ip if dev.ip else f"10.0.{i // 254}.{i % 254 + 1}/24"
        devices.append({"hostname": hostname, "ip": ip})

    # Build connection entries
    connections = []
    for conn in topology.connections:
        src_hostname = hostname_map[conn.from_.devId]
        dst_hostname = hostname_map[conn.to.devId]
        connections.append({
            "src": src_hostname,
            "dst": dst_hostname,
            "bw": conn.bandwidth,
            "delay": conn.delay,
        })

    # Simple Jinja2 rendering (avoid dependency for a single template)
    import re

    template = MININET_SCRIPT_TEMPLATE
    # Render devices
    device_lines = []
    for d in devices:
        device_lines.append(
            f'        {d["hostname"]} = self.addHost(\'{d["hostname"]}\', ip=\'{d["ip"]}\')'
        )
    template = template.replace(
        "        {% for dev in devices %}\n        {{ dev.hostname }} = self.addHost('{{ dev.hostname }}', ip='{{ dev.ip }}')\n        {% endfor %}",
        "\n".join(device_lines),
    )

    # Render connections
    conn_lines = []
    for c in connections:
        conn_lines.append(
            f'        self.addLink({c["src"]}, {c["dst"]}, bw={c["bw"]}, delay=\'{c["delay"]}ms\')'
        )
    template = template.replace(
        "        {% for conn in connections %}\n        self.addLink({{ conn.src }}, {{ conn.dst }}, bw={{ conn.bw }}, delay='{{ conn.delay }}ms')\n        {% endfor %}",
        "\n".join(conn_lines),
    )

    return template


def build_nx_graph(topology: TopologyJSON) -> "nx.Graph":
    """Convert TopologyJSON to a networkx Graph for the model."""
    import networkx as nx

    G = nx.Graph()

    # Map device IDs to integer node indices
    id_to_idx: dict[str, int] = {}
    for i, dev in enumerate(topology.devices):
        id_to_idx[dev.id] = i
        G.add_node(i, label=dev.label, type=dev.type, ip=dev.ip)

    for conn in topology.connections:
        u = id_to_idx[conn.from_.devId]
        v = id_to_idx[conn.to.devId]
        G.add_edge(u, v, bandwidth=conn.bandwidth, delay=conn.delay)

    return G, id_to_idx
```

- [ ] **Step 2: Commit**

```bash
git add backend/mininet/ && git commit -m "feat: add Mininet script template generator and topology-to-nx converter"
```

---

### Task 3: Docker container manager

**Files:**
- Create: `backend/mininet/manager.py`

- [ ] **Step 1: Write Docker manager for Mininet containers**

```python
# backend/mininet/manager.py
"""Manage Mininet Docker container lifecycle."""

from __future__ import annotations

import uuid
import time
import json
import os
import tempfile
from pathlib import Path

import docker
from docker.errors import ImageNotFound, NotFound

from schemas.models import TopologyJSON
from mininet.templates import generate_mininet_script, build_nx_graph

MININET_IMAGE = "mnknowles/mininet:latest"


class MininetManager:
    def __init__(self):
        self.client = docker.from_env()
        self._ensure_image()

    def _ensure_image(self):
        """Pull Mininet image if not present."""
        try:
            self.client.images.get(MININET_IMAGE)
        except ImageNotFound:
            print(f"Pulling Mininet image {MININET_IMAGE}...")
            self.client.images.pull(MININET_IMAGE)

    def deploy(self, topology: TopologyJSON, flows: list[dict]) -> str:
        """Deploy topology to a Mininet container. Returns container ID."""
        container_name = f"mininet-{uuid.uuid4().hex[:8]}"
        script = generate_mininet_script(topology)

        # Write script and flows to temp directory
        tmpdir = tempfile.mkdtemp(prefix="mininet-")
        script_path = os.path.join(tmpdir, "topo.py")
        flows_path = os.path.join(tmpdir, "flows.json")

        with open(script_path, "w") as f:
            f.write(script)
        with open(flows_path, "w") as f:
            json.dump(flows, f)

        container = self.client.containers.run(
            MININET_IMAGE,
            name=container_name,
            command="tail -f /dev/null",  # keep alive
            volumes={
                tmpdir: {"bind": "/tmp/topo", "mode": "rw"},
            },
            environment={"FLOWS_FILE": "/tmp/topo/flows.json"},
            privileged=True,  # Mininet needs network privileges
            detach=True,
            remove=False,
        )

        # Execute the topology script in background
        exec_id = self.client.api.exec_create(
            container.id,
            "python /tmp/topo/topo.py",
            stdout=True,
            stderr=True,
        )

        return container.id, exec_id["Id"], tmpdir

    def get_container(self, container_id: str):
        """Get a container by ID."""
        try:
            return self.client.containers.get(container_id)
        except NotFound:
            return None

    def exec_command(self, container_id: str, cmd: str) -> tuple[str, str]:
        """Execute a command in the container, return (stdout, stderr)."""
        container = self.get_container(container_id)
        if not container:
            return "", "Container not found"
        exit_code, output = container.exec_run(cmd, stdout=True, stderr=True)
        return output.decode("utf-8", errors="replace"), ""

    def stop_and_remove(self, container_id: str):
        """Stop and remove a container."""
        container = self.get_container(container_id)
        if container:
            container.stop(timeout=5)
            container.remove(force=True)

    def cleanup_tmpdir(self, tmpdir: str):
        """Remove temporary directory."""
        import shutil

        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
```

- [ ] **Step 2: Commit**

```bash
git add backend/mininet/manager.py && git commit -m "feat: add Docker container manager for Mininet"
```

---

### Task 4: Traffic generator

**Files:**
- Create: `backend/traffic/__init__.py`
- Create: `backend/traffic/generator.py`

- [ ] **Step 1: Write automatic traffic generator**

```python
# backend/traffic/generator.py
"""Auto-generate traffic flows for a given topology."""

from __future__ import annotations

import random
import math


def generate_flows(
    num_nodes: int,
    num_flows: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Generate random flow demands for a topology.

    Matches the distribution used in network-rl training:
    - bw ~ U[0, 40] (matches FLOW_BW max ~40 used in training)
    - phi ~ U[0, 1]
    - Each flow between a random (src, dst) pair, src != dst

    Returns list of {flow_id, src, dst, bw_req, phi, duration}.
    """
    if seed is not None:
        random.seed(seed)

    if num_flows is None:
        num_flows = min(num_nodes * 3, 50)  # reasonable default

    flows = []
    for flow_id in range(num_flows):
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        while dst == src:
            dst = random.randint(0, num_nodes - 1)

        bw_req = round(random.uniform(0.5, 40.0), 2)
        phi = round(random.uniform(0.0, 1.0), 2)
        duration = random.randint(3, 10)

        flows.append({
            "flow_id": flow_id,
            "src": src,
            "dst": dst,
            "bw_req": bw_req,
            "phi": phi,
            "duration": duration,
        })

    return flows
```

- [ ] **Step 2: Commit**

```bash
git add backend/traffic/ && git commit -m "feat: add automatic traffic flow generator"
```

---

### Task 5: network-rl inference engine

**Files:**
- Create: `backend/model/__init__.py`
- Create: `backend/model/inference.py`

- [ ] **Step 1: Write the inference engine**

```python
# backend/model/inference.py
"""network-rl Policy inference wrapper.

Follows the inference interface from 模型项目/network-rl/api introduction.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import numpy as np
import networkx as nx

# Add network-rl to path so we can import xchirl
_NETWORK_RL_ROOT = Path(__file__).resolve().parents[2] / "模型项目" / "network-rl"
if str(_NETWORK_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(_NETWORK_RL_ROOT))

from xchirl.utils.make_component_ppo import make_encoder
from xchirl.modules.encoders import PathPooler
from xchirl.modules.scorers import KPathScorer

# Normalization constants from api introduction.md
DELAY_MU, DELAY_SIG = 10.5, 5.5
BW_MU, BW_SIG = 65.0, 20.2


class Policy:
    """XCHiRL routing policy model (P4 mode, metrics dim=2)."""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        data = torch.load(ckpt_path, weights_only=False, map_location=device)
        hp = data.get("hparams", {})

        hidden_dim = hp.get("hidden_dim", 256)
        kind = hp.get("encoder_kind", "film_gnn")

        self.encoder = make_encoder(
            hidden_dim, hp.get("layer_num", 4), kind=kind, heads=hp.get("heads", 1)
        )
        self.pooler = PathPooler(hidden_dim=hidden_dim)
        self.scorer = KPathScorer(hidden_dim=hidden_dim)

        sd = data["actor_state_dict"]
        self.encoder.load_state_dict(sd, strict=False)
        self.pooler.load_state_dict(sd, strict=False)
        self.scorer.load_state_dict(sd, strict=False)

        self.encoder.eval()
        self.pooler.eval()
        self.scorer.eval()
        self.to(device)

    def to(self, device):
        self.encoder.to(device)
        self.pooler.to(device)
        self.scorer.to(device)
        self._device = device
        return self

    @torch.no_grad()
    def forward(self, x, index, features, metrics, paths, paths_mask):
        """Select best path. Returns (action: int, logits: Tensor[K])."""
        h = self.encoder(x, index, features, metrics)
        h = self.pooler(h, paths, paths_mask)
        logits = self.scorer(h, metrics)
        return int(logits.argmax(dim=-1).item()), logits


class InferenceEngine:
    """Runs network-rl inference on a topology + flow list."""

    def __init__(self, ckpt_path: str | None = None, device: str = "cpu"):
        if ckpt_path is None:
            ckpt_path = str(
                _NETWORK_RL_ROOT / "runs" / "FILM_PPO" / "best.pt"
            )
        self.device = device
        self.policy: Policy | None = None
        self.ckpt_path = ckpt_path
        self.K = 16
        self.L_max = 22  # will be adjusted per topology

    def _ensure_loaded(self):
        if self.policy is None:
            if not Path(self.ckpt_path).exists():
                raise FileNotFoundError(
                    f"Checkpoint not found at {self.ckpt_path}. "
                    "Train a model first or provide a valid checkpoint path."
                )
            self.policy = Policy(self.ckpt_path, device=self.device)

    def infer(
        self, G: nx.Graph, flows: list[dict]
    ) -> list[dict]:
        """Run inference on a list of flows over a topology.

        Args:
            G: networkx undirected graph with edge attrs 'bandwidth' and 'delay'
            flows: list of {flow_id, src, dst, bw_req, phi}

        Returns:
            list of {flow_id, src, dst, bw_req, phi, selected_path, hops, ospf_path}
        """
        self._ensure_loaded()

        N = G.number_of_nodes()
        self.L_max = N  # path buffer size = num nodes

        # Relabel nodes to consecutive integers
        G = nx.convert_node_labels_to_integers(G)

        # Build directed edge index [2, E]
        edges = []
        edge_bandwidths = {}
        edge_delays = {}
        for u, v in G.edges():
            edges += [(u, v), (v, u)]
            bw = G[u][v].get("bandwidth", 100.0)
            delay = G[u][v].get("delay", 5.0)
            edge_bandwidths[(u, v)] = bw
            edge_bandwidths[(v, u)] = bw
            edge_delays[(u, v)] = delay
            edge_delays[(v, u)] = delay

        index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        E = index.shape[1]

        # Build edge features [E, 3]: delay_norm, util, bw_norm
        delay_norm_list, util_list, bw_norm_list = [], [], []
        for u, v in edges:
            d = edge_delays[(u, v)]
            c = edge_bandwidths[(u, v)]
            delay_norm_list.append((d - DELAY_MU) / DELAY_SIG)
            util_list.append(0.0)  # initial utilization
            bw_norm_list.append((c - BW_MU) / BW_SIG)

        features = torch.tensor(
            np.stack([delay_norm_list, util_list, bw_norm_list], axis=-1),
            dtype=torch.float32,
            device=self.device,
        )

        # Precompute K-shortest paths for each flow
        import itertools

        all_paths = {}
        for flow in flows:
            src, dst = flow["src"], flow["dst"]
            key = (src, dst)
            if key not in all_paths:
                paths_list = list(
                    itertools.islice(
                        nx.shortest_simple_paths(G, src, dst), self.K
                    )
                )
                all_paths[key] = paths_list

        results = []
        for flow in flows:
            src, dst = flow["src"], flow["dst"]
            bw_req = flow["bw_req"]
            phi = flow["phi"]

            paths_list = all_paths.get((src, dst), [])
            if not paths_list:
                results.append({
                    **flow,
                    "selected_path": [],
                    "hops": 0,
                    "ospf_path": [],
                    "max_link_utilization": 0.0,
                })
                continue

            # Build paths tensor [K, L]
            K_actual = min(len(paths_list), self.K)
            paths_tensor = torch.full(
                (self.K, self.L_max), -1, dtype=torch.long, device=self.device
            )
            paths_mask = torch.zeros(
                self.K, self.L_max, dtype=torch.bool, device=self.device
            )
            for k in range(K_actual):
                p = paths_list[k]
                paths_tensor[k, : len(p)] = torch.tensor(p, device=self.device)
                paths_mask[k, : len(p)] = True

            # Node features [N, 2]: [is_src, is_dst]
            x = torch.zeros(N, 2, device=self.device)
            x[src, 0] = 1.0
            x[dst, 1] = 1.0

            # Flow metrics [2]: phi, bw_req_norm
            metrics = torch.tensor(
                [phi, (bw_req - BW_MU) / BW_SIG],
                device=self.device,
            )

            # Inference
            action, logits = self.policy.forward(
                x, index, features, metrics, paths_tensor, paths_mask
            )

            if action < len(paths_list):
                selected = paths_list[action]
            else:
                selected = paths_list[0]

            # Compute OSPF (hop-shortest) path for comparison
            try:
                ospf_path = nx.shortest_path(G, src, dst, weight=None)
            except nx.NetworkXNoPath:
                ospf_path = []

            # Update edge utilization for this flow
            path_nodes = selected
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                for eid_dir in [(u, v), (v, u)]:
                    try:
                        eid = edges.index(eid_dir)
                        c = edge_bandwidths.get(eid_dir, 100.0)
                        current_util = features[eid, 1].item()
                        new_util = current_util + bw_req / c
                        features[eid, 1] = min(new_util, 1.0)
                    except ValueError:
                        pass

            # Compute max link utilization along selected path
            max_util = 0.0
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                for eid_dir in [(u, v), (v, u)]:
                    try:
                        eid = edges.index(eid_dir)
                        max_util = max(max_util, features[eid, 1].item())
                    except ValueError:
                        pass

            results.append({
                "flow_id": flow["flow_id"],
                "src": src,
                "dst": dst,
                "bw_req": bw_req,
                "phi": phi,
                "selected_path": selected,
                "hops": len(selected) - 1,
                "ospf_path": ospf_path,
                "max_link_utilization": round(max_util, 4),
            })

        # Also return final edge utilization for display
        edge_utils = {}
        for i, (u, v) in enumerate(edges):
            if u < v:  # undirected, take one direction
                edge_utils[(u, v)] = features[i, 1].item()

        return results, edge_utils
```

- [ ] **Step 2: Commit**

```bash
git add backend/model/ && git commit -m "feat: add network-rl inference engine with Policy wrapper"
```

---

### Task 6: FastAPI application with /api/deploy endpoint

**Files:**
- Create: `backend/main.py`

- [ ] **Step 1: Write the FastAPI application**

```python
# backend/main.py
"""AI Router — FastAPI backend."""

from __future__ import annotations

import uuid
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas.models import (
    TopologyJSON,
    DeploymentResult,
    FlowResult,
    ChatRequest,
    ChatResponse,
)
from mininet.manager import MininetManager
from mininet.templates import build_nx_graph
from traffic.generator import generate_flows
from model.inference import InferenceEngine

# Global state
mn_manager: MininetManager | None = None
inference_engine: InferenceEngine | None = None
jobs: dict[str, DeploymentResult] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mn_manager, inference_engine
    mn_manager = MininetManager()
    inference_engine = InferenceEngine()
    yield
    # Cleanup on shutdown
    for job_id in list(jobs.keys()):
        if jobs[job_id].status == "running":
            jobs[job_id].status = "failed"
            jobs[job_id].error = "Server shutdown"


app = FastAPI(title="AI Router", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/deploy")
async def deploy(topology: TopologyJSON) -> DeploymentResult:
    """Deploy topology to Mininet, run traffic, infer optimal routes."""
    job_id = uuid.uuid4().hex[:12]

    # Create initial job entry
    jobs[job_id] = DeploymentResult(
        job_id=job_id,
        status="running",
        flows=[],
        topology_nodes=[],
        topology_edges=[],
    )

    try:
        # 1. Convert topology to networkx
        G, id_to_idx = build_nx_graph(topology)
        N = G.number_of_nodes()

        # 2. Generate traffic flows
        flows = generate_flows(N, seed=42)

        # 3. Map flow src/dst from integer indices to Mininet hostnames
        idx_to_hostname = {v: k for k, v in id_to_idx.items()}
        idx_to_label = {
            id_to_idx[dev.id]: dev.label for dev in topology.devices
        }

        mininet_flows = []
        for f in flows:
            mininet_flows.append({
                "flow_id": f["flow_id"],
                "src": f"h{f['src'] + 1}",
                "dst": f"h{f['dst'] + 1}",
                "bw_req": f["bw_req"],
                "duration": f["duration"],
            })

        # 4. Deploy to Mininet
        container_id, exec_id, tmpdir = mn_manager.deploy(topology, mininet_flows)

        # 5. Wait for topology to be ready
        time.sleep(5)  # give Mininet time to start

        # 6. Run network-rl inference
        flow_results, edge_utils = inference_engine.infer(G, flows)

        # 7. Build result
        result_flows = []
        for fr in flow_results:
            result_flows.append(
                FlowResult(
                    flow_id=fr["flow_id"],
                    src=fr["src"],
                    dst=fr["dst"],
                    bw_req=fr["bw_req"],
                    phi=fr["phi"],
                    selected_path=fr["selected_path"],
                    hops=fr["hops"],
                    max_link_utilization=fr["max_link_utilization"],
                    ospf_path=fr.get("ospf_path"),
                )
            )

        # 8. Build topology edges for frontend display
        topo_edges = []
        for u, v, data in G.edges(data=True):
            util = edge_utils.get((u, v), edge_utils.get((v, u), 0.0))
            topo_edges.append({
                "src": u,
                "dst": v,
                "bandwidth": data.get("bandwidth", 100.0),
                "delay": data.get("delay", 5.0),
                "utilization": round(util, 4),
            })

        result = DeploymentResult(
            job_id=job_id,
            status="completed",
            flows=result_flows,
            topology_nodes=list(range(N)),
            topology_edges=topo_edges,
        )

    except Exception as e:
        result = DeploymentResult(
            job_id=job_id,
            status="failed",
            error=str(e),
        )

    finally:
        # Cleanup
        if 'container_id' in dir():
            mn_manager.stop_and_remove(container_id)
        if 'tmpdir' in dir():
            mn_manager.cleanup_tmpdir(tmpdir)

    jobs[job_id] = result
    return result


@app.get("/api/status/{job_id}")
async def get_status(job_id: str) -> DeploymentResult:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.delete("/api/containers/{container_id}")
async def remove_container(container_id: str):
    mn_manager.stop_and_remove(container_id)
    return {"status": "removed", "container_id": container_id}
```

- [ ] **Step 2: Create backend run script**

Create `backend/run.py`:

```python
# backend/run.py
"""Development server runner."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

- [ ] **Step 3: Commit**

```bash
git add backend/main.py backend/run.py && git commit -m "feat: add FastAPI app with /api/deploy endpoint"
```

---

### Task 7: Frontend — connection properties panel

**Files:**
- Modify: `topology-editor.html`

- [ ] **Step 1: Add bandwidth and delay fields to connection properties**

In `showProps()`, when a connection (line) is selected, extend the properties panel to include editable bandwidth and delay fields.

Replace the connection properties section in `showProps()` (around line 788-799):

```javascript
// In showProps(), replace the connection section:
if (selectedLineId) {
  var conn = connections.find(function(c) { return c.id === selectedLineId; });
  if (!conn) { body.innerHTML = '<p class="props-empty">选择一个设备查看属性</p>'; typeLabel.textContent = ''; return; }
  var fromDev = devices.find(function(d) { return d.id === conn.from.devId; });
  var toDev = devices.find(function(d) { return d.id === conn.to.devId; });

  // Ensure bandwidth/delay defaults
  if (conn.bandwidth === undefined) conn.bandwidth = 100;
  if (conn.delay === undefined) conn.delay = 5;

  typeLabel.textContent = '连线';
  body.innerHTML =
    '<div class="prop-group"><label>连线 ID</label><input readonly value="' + conn.id + '" /></div>' +
    '<div class="prop-group"><label>起点</label><input readonly value="' + (fromDev ? fromDev.label : '?') + ' · ' + conn.from.port + '" /></div>' +
    '<div class="prop-group"><label>终点</label><input readonly value="' + (toDev ? toDev.label : '?') + ' · ' + conn.to.port + '" /></div>' +
    '<div class="prop-group"><label>带宽 (Mbps)</label><input type="number" value="' + conn.bandwidth + '" min="1" max="10000" step="1" onchange="updateConnProp(\'bandwidth\', parseFloat(this.value))" /></div>' +
    '<div class="prop-group"><label>时延 (ms)</label><input type="number" value="' + conn.delay + '" min="0" max="500" step="0.1" onchange="updateConnProp(\'delay\', parseFloat(this.value))" /></div>' +
    '<button class="tb-btn danger" style="width:100%;margin-top:8px;" onclick="deleteSelectedLine()">删除连线</button>';
  return;
}
```

- [ ] **Step 2: Add `updateConnProp` function**

Add after `updateDeviceProp` (around line 828):

```javascript
function updateConnProp(prop, value) {
  if (!selectedLineId) return;
  var conn = connections.find(function(c) { return c.id === selectedLineId; });
  if (!conn) return;
  saveUndo();
  conn[prop] = value;
  renderAll();
  showProps();
}
```

- [ ] **Step 3: Add `deleteSelectedLine` to global scope**

Ensure `deleteSelectedLine` is on `window` (it's already defined at line 830, but add explicit window ref for onclick):

No changes needed — it's already a global function.

- [ ] **Step 4: Commit**

```bash
git add topology-editor.html && git commit -m "feat: add bandwidth/delay fields to connection properties panel"
```

---

### Task 8: Frontend — deploy button and API call

**Files:**
- Modify: `topology-editor.html`

- [ ] **Step 1: Add deploy button to toolbar**

In the toolbar HTML (around line 286), add before the export button:

```html
<button class="tb-btn accent" id="btn-deploy" title="部署到 Mininet 并运行推理">&#9654; 部署</button>
<span class="sep"></span>
```

- [ ] **Step 2: Add deploy logic and status display**

Add after the export button event listener (around line 861):

```javascript
/* ─── deploy ───────────────────────────────────────────────── */
var API_BASE = 'http://localhost:8000';
var deployResult = null;

document.getElementById('btn-deploy').addEventListener('click', function() {
  if (devices.length === 0) { alert('请先添加设备'); return; }

  var btn = document.getElementById('btn-deploy');
  btn.disabled = true;
  btn.textContent = '部署中...';

  // Build topology JSON with connection properties
  var topo = {
    devices: devices.map(function(d) { return { id: d.id, type: d.type, x: d.x, y: d.y, label: d.label, ip: d.ip }; }),
    connections: connections.map(function(c) { return { id: c.id, from: c.from, to: c.to, bandwidth: c.bandwidth || 100, delay: c.delay || 5 }; })
  };

  fetch(API_BASE + '/api/deploy', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(topo)
  })
  .then(function(r) { return r.json(); })
  .then(function(result) {
    deployResult = result;
    btn.disabled = false;
    btn.textContent = '部署';
    if (result.status === 'completed') {
      showResults(result);
    } else {
      alert('部署失败: ' + (result.error || '未知错误'));
    }
  })
  .catch(function(err) {
    btn.disabled = false;
    btn.textContent = '部署';
    alert('请求失败: ' + err.message + '\n请确认后端已启动 (python backend/run.py)');
  });
});
```

- [ ] **Step 3: Commit**

```bash
git add topology-editor.html && git commit -m "feat: add deploy button with API call to backend"
```

---

### Task 9: Frontend — result display (path highlight + table)

**Files:**
- Modify: `topology-editor.html`

- [ ] **Step 1: Add result panel CSS**

Add to the `<style>` block (before `</style>`):

```css
/* ─── result panel ──────────────────────────────────────────── */
.result-panel {
  position: fixed; bottom: 0; left: 0; right: 0;
  height: 200px;
  background: var(--surface);
  border-top: 2px solid var(--accent);
  display: none; flex-direction: column;
  z-index: 100;
  box-shadow: 0 -4px 20px rgba(0,0,0,0.1);
}
.result-panel.open { display: flex; }
.result-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px var(--gap-md);
  border-bottom: 1px solid var(--border);
  font-family: var(--font-mono);
  font-size: var(--fs-xs);
  letter-spacing: 0.05em;
}
.result-header .flow-selector {
  display: flex; align-items: center; gap: 8px;
}
.result-header select {
  padding: 3px 6px; border: 1px solid var(--border);
  border-radius: var(--radius); font: inherit; font-size: var(--fs-sm);
  background: var(--bg); color: var(--fg);
}
.result-table-wrap {
  flex: 1; overflow: auto; padding: 4px var(--gap-md);
}
.result-table {
  width: 100%; border-collapse: collapse;
  font-size: var(--fs-sm);
}
.result-table th {
  text-align: left; padding: 6px 8px;
  font-family: var(--font-mono); font-size: var(--fs-xs);
  color: var(--muted); letter-spacing: 0.04em;
  border-bottom: 1px solid var(--border);
  position: sticky; top: 0; background: var(--surface);
}
.result-table td {
  padding: 5px 8px; border-bottom: 1px solid var(--border);
  font-family: var(--font-mono); font-size: var(--fs-xs);
}
.result-table tr:hover td { background: var(--accent-soft); }
.result-table tr.active-row td { background: var(--accent-soft); }
.path-highlight {
  stroke: var(--accent); stroke-width: 4;
  fill: none; pointer-events: none;
  stroke-linecap: round;
}
.path-highlight-bg {
  stroke: var(--accent); stroke-width: 8;
  fill: none; pointer-events: none;
  opacity: 0.15; stroke-linecap: round;
}
```

- [ ] **Step 2: Add result panel HTML**

Add after the main layout div (after the `</div>` closing `.main`, before `</body>`):

```html
<!-- result panel -->
<div class="result-panel" id="result-panel">
  <div class="result-header">
    <span>推理结果 · <span id="result-flow-count">0</span> 条流</span>
    <div class="flow-selector">
      <label for="flow-select" style="color:var(--muted);">查看流:</label>
      <select id="flow-select" onchange="highlightFlow(this.value)">
        <option value="">全部</option>
      </select>
      <button class="tb-btn" onclick="closeResults()">&#10005;</button>
    </div>
  </div>
  <div class="result-table-wrap">
    <table class="result-table">
      <thead>
        <tr>
          <th>流 ID</th><th>src → dst</th><th>bw (Mbps)</th><th>φ</th>
          <th>模型路径</th><th>跳数</th><th>利用率</th><th>OSPF路径</th>
        </tr>
      </thead>
      <tbody id="result-tbody"></tbody>
    </table>
  </div>
</div>
```

- [ ] **Step 3: Add result display JavaScript**

Add after the deploy logic:

```javascript
/* ─── results display ──────────────────────────────────────── */
var PATH_COLORS = ['#4C9AFF','#36B37E','#FF5630','#6554C0','#FF8B00','#00B8D9','#8777D9','#FFC400'];

function showResults(result) {
  deployResult = result;
  var panel = document.getElementById('result-panel');
  var tbody = document.getElementById('result-tbody');
  var select = document.getElementById('flow-select');
  var countEl = document.getElementById('result-flow-count');

  countEl.textContent = result.flows.length;

  // Populate flow selector
  select.innerHTML = '<option value="">全部</option>';
  result.flows.forEach(function(f, i) {
    select.innerHTML += '<option value="' + i + '">流 ' + f.flow_id + ' (' + f.src + '→' + f.dst + ')</option>';
  });

  // Populate table
  tbody.innerHTML = '';
  result.flows.forEach(function(f, i) {
    var pathStr = f.selected_path.join(' → ');
    var ospfStr = f.ospf_path ? f.ospf_path.join(' → ') : '-';
    var tr = document.createElement('tr');
    tr.id = 'flow-row-' + i;
    tr.innerHTML =
      '<td>' + f.flow_id + '</td>' +
      '<td>' + f.src + ' → ' + f.dst + '</td>' +
      '<td>' + f.bw_req + '</td>' +
      '<td>' + f.phi + '</td>' +
      '<td>' + pathStr + '</td>' +
      '<td>' + f.hops + '</td>' +
      '<td>' + (f.max_link_utilization * 100).toFixed(1) + '%</td>' +
      '<td>' + ospfStr + '</td>';
    tr.addEventListener('click', function() { highlightFlow(i); });
    tbody.appendChild(tr);
  });

  panel.classList.add('open');
  highlightFlow(''); // show all paths initially
}

function highlightFlow(flowIdx) {
  // Update table row highlighting
  var rows = document.querySelectorAll('#result-tbody tr');
  rows.forEach(function(r) { r.classList.remove('active-row'); });

  if (flowIdx === '' || flowIdx === undefined) {
    // Show all
    rows.forEach(function(r) { r.classList.add('active-row'); });
  } else {
    var row = document.getElementById('flow-row-' + flowIdx);
    if (row) row.classList.add('active-row');
  }

  // Redraw canvas with path highlights
  renderAll(); // base render
  drawPathHighlights(flowIdx);
}

function drawPathHighlights(flowIdx) {
  if (!deployResult || !deployResult.flows) return;

  var NS = 'http://www.w3.org/2000/svg';
  var svgEl = document.getElementById('topo-svg');
  var g = svgEl.querySelector('g'); // the transform group

  // Build device ID → integer node index mapping
  var devIdToIdx = {};
  devices.forEach(function(d, i) { devIdToIdx[d.id] = i; });

  // Build node index → position (center of device)
  var nodePos = {};
  devices.forEach(function(d, i) {
    var idx = devIdToIdx[d.id];
    if (idx !== undefined) {
      nodePos[idx] = { x: d.x + DEV_W/2, y: d.y + DEV_H/2 };
    }
  });

  var flowsToShow = [];
  if (flowIdx === '' || flowIdx === undefined) {
    flowsToShow = deployResult.flows;
  } else {
    flowsToShow = [deployResult.flows[parseInt(flowIdx)]];
  }

  flowsToShow.forEach(function(f, fi) {
    var color = PATH_COLORS[fi % PATH_COLORS.length];
    var path = f.selected_path;
    if (!path || path.length < 2) return;

    // Draw background glow
    var bgPath = document.createElementNS(NS, 'path');
    bgPath.classList.add('path-highlight-bg');
    bgPath.setAttribute('stroke', color);
    var dStr = '';
    for (var i = 0; i < path.length; i++) {
      var pos = nodePos[path[i]];
      if (!pos) continue;
      if (i === 0) dStr += 'M' + pos.x + ',' + pos.y;
      else dStr += ' L' + pos.x + ',' + pos.y;
    }
    bgPath.setAttribute('d', dStr);
    g.appendChild(bgPath);

    // Draw foreground line
    var fgPath = document.createElementNS(NS, 'path');
    fgPath.classList.add('path-highlight');
    fgPath.setAttribute('stroke', color);
    fgPath.setAttribute('d', dStr);
    g.appendChild(fgPath);
  });
}

function closeResults() {
  deployResult = null;
  document.getElementById('result-panel').classList.remove('open');
  renderAll();
}
```

- [ ] **Step 4: Commit**

```bash
git add topology-editor.html && git commit -m "feat: add result display with path highlighting and comparison table"
```

---

### Task 10: Agent — LLM orchestrator

**Files:**
- Create: `backend/agent/__init__.py`
- Create: `backend/agent/orchestrator.py`

- [ ] **Step 1: Write the Agent orchestrator with Claude API**

```python
# backend/agent/orchestrator.py
"""LLM Agent orchestrator using Claude API with function calling."""

from __future__ import annotations

import json

from anthropic import Anthropic

from schemas.models import TopologyJSON, DeploymentResult, ChatResponse

SYSTEM_PROMPT = """\
You are a network topology assistant for the AI Router platform. You help users design network topologies, generate traffic scenarios, deploy to Mininet, and interpret routing optimization results.

You have access to tools. Use them to fulfill the user's request step by step.

When generating a topology, the available device types are: router, switch, firewall, server, laptop, database, lb, cloud, wifi, printer.
Connections between devices must have bandwidth (Mbps, default 100) and delay (ms, default 5).

When generating traffic, produce a realistic scenario matching the user's description. Each flow needs: flow_id, src (node index), dst (node index), bw_req (Mbps), phi (0-1, lower = delay-sensitive, higher = bandwidth-sensitive), duration (seconds).

When explaining results, compare the model's chosen path vs the OSPF path. Explain WHY the model made its choice in terms of link utilization and QoS tradeoffs. Be concise but insightful.
"""

TOOLS = [
    {
        "name": "generate_topology",
        "description": "Generate a network topology from a natural language description. Returns a TopologyJSON that will be loaded into the editor.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Natural language description of the desired network topology. Include device types, counts, and how they connect."
                }
            },
            "required": ["description"]
        }
    },
    {
        "name": "generate_traffic",
        "description": "Generate traffic flows for a topology based on a scenario description.",
        "input_schema": {
            "type": "object",
            "properties": {
                "num_nodes": {
                    "type": "integer",
                    "description": "Number of nodes in the topology."
                },
                "scenario": {
                    "type": "string",
                    "description": "Description of the traffic scenario (e.g. 'video conferencing', 'database replication', 'web browsing peak')."
                }
            },
            "required": ["num_nodes", "scenario"]
        }
    },
    {
        "name": "deploy_and_analyze",
        "description": "Deploy the topology to Mininet, run traffic, and get model-optimized routes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topology_json": {
                    "type": "string",
                    "description": "JSON string of the topology."
                },
                "traffic_json": {
                    "type": "string",
                    "description": "JSON string of the traffic flows array."
                }
            },
            "required": ["topology_json", "traffic_json"]
        }
    },
    {
        "name": "explain_results",
        "description": "Explain the routing optimization results in natural language.",
        "input_schema": {
            "type": "object",
            "properties": {
                "results_json": {
                    "type": "string",
                    "description": "JSON string of the deployment results."
                }
            },
            "required": ["results_json"]
        }
    },
]


class AgentOrchestrator:
    def __init__(self, api_key: str | None = None):
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-6-20250514"

    def chat(
        self,
        message: str,
        topology: TopologyJSON | None = None,
        on_topology: callable = None,
        on_traffic: callable = None,
        on_deploy: callable = None,
    ) -> ChatResponse:
        """Process a user message through the Agent.

        Args:
            message: User's natural language message
            topology: Current editor topology state (if any)
            on_topology: Callback receiving TopologyJSON when agent generates one
            on_traffic: Callback receiving traffic list when agent generates one
            on_deploy: Callback receiving (topology, traffic) to trigger deploy

        Returns:
            ChatResponse with agent's reply and optional action data
        """
        messages = []

        # Build context about current state
        context = "The user is interacting with the AI Router topology editor."
        if topology:
            context += f"\nCurrent topology: {len(topology.devices)} devices, {len(topology.connections)} connections."
        context += "\n\nUser message: " + message

        messages.append({"role": "user", "content": context})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Process tool calls
        result_topology = None
        result_data = None
        reply = ""

        for block in response.content:
            if block.type == "text":
                reply += block.text
            elif block.type == "tool_use":
                tool_result = self._execute_tool(
                    block,
                    topology,
                    on_topology,
                    on_traffic,
                    on_deploy,
                )
                if tool_result:
                    if isinstance(tool_result, TopologyJSON):
                        result_topology = tool_result
                    elif isinstance(tool_result, DeploymentResult):
                        result_data = tool_result

        return ChatResponse(
            reply=reply.strip(),
            action="load_topology" if result_topology else ("show_results" if result_data else None),
            topology=result_topology,
            results=result_data,
        )

    def _execute_tool(
        self,
        tool_use,
        current_topology: TopologyJSON | None,
        on_topology: callable | None,
        on_traffic: callable | None,
        on_deploy: callable | None,
    ):
        name = tool_use.name
        inp = tool_use.input

        if name == "generate_topology":
            return self._tool_generate_topology(inp.get("description", ""), on_topology)

        elif name == "generate_traffic":
            return self._tool_generate_traffic(
                inp.get("num_nodes", 5), inp.get("scenario", ""), on_traffic
            )

        elif name == "deploy_and_analyze":
            return self._tool_deploy_and_analyze(
                inp.get("topology_json", "{}"),
                inp.get("traffic_json", "[]"),
                on_deploy,
            )

        elif name == "explain_results":
            return self._tool_explain_results(inp.get("results_json", "{}"))

        return None

    def _tool_generate_topology(self, description: str, on_topology) -> TopologyJSON:
        """Use LLM to generate a topology JSON from description."""
        prompt = f"""\
Generate a network topology as a JSON object based on this description: "{description}"

Available device types (use the exact type strings): router, switch, firewall, server, laptop, database, lb, cloud, wifi, printer

Return ONLY valid JSON in this exact format:
{{
  "devices": [
    {{"id": "dev-1", "type": "router", "x": 200, "y": 100, "label": "Core Router", "ip": "10.0.0.1"}}
  ],
  "connections": [
    {{"id": "conn-1", "from": {{"devId": "dev-1", "port": "bottom"}}, "to": {{"devId": "dev-2", "port": "top"}}, "bandwidth": 100, "delay": 5}}
  ]
}}

Rules:
- Position devices in a readable layout (x from 100-800, y from 50-500)
- Use ports "top", "right", "bottom", "left" appropriately for the layout
- Set reasonable bandwidth (10-10000 Mbps) and delay (0-100 ms) for each link
- Assign reasonable IPs (10.0.x.y)
- Each device needs a unique id starting from "dev-1"
- Each connection needs a unique id starting from "conn-1"
- Include ALL devices from the description"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text
        # Extract JSON from response
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(text[json_start:json_end])
            topo = TopologyJSON(**data)
            if on_topology:
                on_topology(topo)
            return topo
        return None

    def _tool_generate_traffic(
        self, num_nodes: int, scenario: str, on_traffic
    ) -> list[dict]:
        """Generate traffic flows matching a scenario."""
        prompt = f"""\
Generate traffic flows for a network with {num_nodes} nodes based on this scenario: "{scenario}"

Return ONLY a JSON array of flow objects:
[
  {{"flow_id": 0, "src": 0, "dst": 3, "bw_req": 25.0, "phi": 0.3, "duration": 5}}
]

Rules:
- src and dst are integer node indices (0 to {num_nodes - 1}), src != dst
- bw_req: bandwidth requirement in Mbps (0.5 to 40)
- phi: QoS sensitivity (0 to 1). Lower = delay-sensitive (video, voice). Higher = bandwidth-sensitive (file transfer, backup)
- duration: flow duration in seconds (3 to 15)
- Generate 5-15 realistic flows matching the scenario
- Include a mix of phi values appropriate for the scenario"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text
        json_start = text.find("[")
        json_end = text.rfind("]") + 1
        if json_start >= 0 and json_end > json_start:
            flows = json.loads(text[json_start:json_end])
            if on_traffic:
                on_traffic(flows)
            return flows
        return []

    def _tool_deploy_and_analyze(
        self,
        topology_json_str: str,
        traffic_json_str: str,
        on_deploy: callable | None,
    ) -> DeploymentResult | None:
        """Trigger deployment. Returns None (deploy is async in practice)."""
        if on_deploy:
            topology = TopologyJSON(**json.loads(topology_json_str))
            traffic = json.loads(traffic_json_str)
            return on_deploy(topology, traffic)
        return None

    def _tool_explain_results(self, results_json_str: str) -> str:
        """Use LLM to explain deployment results."""
        prompt = f"""\
Explain the following network routing optimization results in natural language.

For each flow, compare the model's chosen path vs the OSPF path. Explain WHY the model made its choice in terms of link utilization, delay, and QoS tradeoffs.

Results:
{results_json_str}

Be concise but insightful. Focus on the most interesting routing decisions."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text
```

- [ ] **Step 2: Commit**

```bash
git add backend/agent/ && git commit -m "feat: add LLM Agent orchestrator with Claude function calling"
```

---

### Task 11: Agent chat endpoint

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Add /api/chat endpoint**

Add to `backend/main.py` after the `/api/containers` endpoint:

```python
from agent.orchestrator import AgentOrchestrator

agent: AgentOrchestrator | None = None

# In lifespan startup, add:
#   global agent
#   agent = AgentOrchestrator()

@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Agent chat endpoint — natural language driven orchestration."""
    global agent, mn_manager, inference_engine

    if agent is None:
        agent = AgentOrchestrator()

    def on_topology(topo):
        pass  # topology is returned in ChatResponse, frontend loads it

    def on_traffic(flows):
        pass  # returned in response

    def on_deploy(topo, traffic):
        # Run the full pipeline synchronously for the agent
        G, _ = build_nx_graph(topo)
        flow_results, edge_utils = inference_engine.infer(G, traffic)

        result_flows = []
        for fr in flow_results:
            result_flows.append(FlowResult(
                flow_id=fr["flow_id"], src=fr["src"], dst=fr["dst"],
                bw_req=fr["bw_req"], phi=fr["phi"],
                selected_path=fr["selected_path"], hops=fr["hops"],
                max_link_utilization=fr["max_link_utilization"],
                ospf_path=fr.get("ospf_path"),
            ))

        topo_edges = []
        for u, v, data in G.edges(data=True):
            util = edge_utils.get((u, v), edge_utils.get((v, u), 0.0))
            topo_edges.append({
                "src": u, "dst": v,
                "bandwidth": data.get("bandwidth", 100.0),
                "delay": data.get("delay", 5.0),
                "utilization": round(util, 4),
            })

        return DeploymentResult(
            job_id="agent",
            status="completed",
            flows=result_flows,
            topology_nodes=list(range(G.number_of_nodes())),
            topology_edges=topo_edges,
        )

    return agent.chat(
        request.message,
        topology=request.topology,
        on_topology=on_topology,
        on_traffic=on_traffic,
        on_deploy=on_deploy,
    )
```

- [ ] **Step 2: Update lifespan to initialize agent**

In `lifespan()`, add `global agent; agent = AgentOrchestrator()` after the other initializations.

- [ ] **Step 3: Commit**

```bash
git add backend/main.py && git commit -m "feat: add /api/chat endpoint for Agent orchestration"
```

---

### Task 12: Frontend — Chat panel

**Files:**
- Modify: `topology-editor.html`

- [ ] **Step 1: Add chat panel CSS**

Add to `<style>`:

```css
/* ─── chat panel ────────────────────────────────────────────── */
.chat-toggle {
  position: fixed; bottom: 20px; right: 20px;
  width: 48px; height: 48px;
  border-radius: 50%;
  background: var(--accent); color: #fff;
  border: none; font-size: 20px;
  cursor: pointer; z-index: 200;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  display: grid; place-items: center;
}
.chat-panel {
  position: fixed; bottom: 80px; right: 20px;
  width: 380px; height: 480px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: 0 8px 30px rgba(0,0,0,0.12);
  display: none; flex-direction: column;
  z-index: 200; overflow: hidden;
}
.chat-panel.open { display: flex; }
.chat-header {
  padding: 12px var(--gap-md);
  border-bottom: 1px solid var(--border);
  font-family: var(--font-mono); font-size: var(--fs-sm);
  font-weight: 600;
  display: flex; justify-content: space-between; align-items: center;
}
.chat-messages {
  flex: 1; overflow-y: auto; padding: var(--gap-md);
  display: flex; flex-direction: column; gap: 10px;
}
.chat-msg {
  max-width: 85%; padding: 8px 12px;
  border-radius: var(--radius);
  font-size: var(--fs-sm); line-height: 1.5;
  white-space: pre-wrap;
}
.chat-msg.user {
  align-self: flex-end;
  background: var(--accent); color: #fff;
}
.chat-msg.agent {
  align-self: flex-start;
  background: var(--bg); border: 1px solid var(--border);
}
.chat-input-wrap {
  display: flex; padding: var(--gap);
  border-top: 1px solid var(--border); gap: var(--gap);
}
.chat-input-wrap input {
  flex: 1; padding: 8px 10px;
  border: 1px solid var(--border); border-radius: var(--radius);
  font: inherit; font-size: var(--fs-sm);
  background: var(--bg); color: var(--fg);
}
.chat-input-wrap button {
  padding: 8px 14px; border-radius: var(--radius);
  background: var(--accent); color: #fff; border: none;
  font-size: var(--fs-sm);
}
```

- [ ] **Step 2: Add chat panel HTML**

Add before `</body>`:

```html
<!-- chat panel -->
<button class="chat-toggle" id="chat-toggle" onclick="toggleChat()" title="AI 助手">&#9991;</button>
<div class="chat-panel" id="chat-panel">
  <div class="chat-header">
    <span>AI 网络助手</span>
    <button class="tb-btn" onclick="toggleChat()" style="border:none;padding:2px 6px;">&#10005;</button>
  </div>
  <div class="chat-messages" id="chat-messages">
    <div class="chat-msg agent">你好！我是网络拓扑助手。你可以用自然语言描述你想要的网络，我会帮你生成拓扑、配置流量并分析路由优化结果。\n\n试试说："帮我搭建一个三层电商架构，两台LB，三台App服务器，两台数据库主从"</div>
  </div>
  <div class="chat-input-wrap">
    <input id="chat-input" placeholder="描述你的网络需求..." onkeydown="if(event.key==='Enter')sendChat()" />
    <button onclick="sendChat()">发送</button>
  </div>
</div>
```

- [ ] **Step 3: Add chat JavaScript**

Add before `</script>`:

```javascript
/* ─── chat ──────────────────────────────────────────────────── */
var chatOpen = false;

function toggleChat() {
  chatOpen = !chatOpen;
  document.getElementById('chat-panel').classList.toggle('open', chatOpen);
}

function addChatMessage(role, text) {
  var msgs = document.getElementById('chat-messages');
  var div = document.createElement('div');
  div.className = 'chat-msg ' + role;
  div.textContent = text;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function sendChat() {
  var input = document.getElementById('chat-input');
  var msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  addChatMessage('user', msg);

  // Build current topology state
  var topo = {
    devices: devices.map(function(d) { return { id: d.id, type: d.type, x: d.x, y: d.y, label: d.label, ip: d.ip }; }),
    connections: connections.map(function(c) { return { id: c.id, from: c.from, to: c.to, bandwidth: c.bandwidth || 100, delay: c.delay || 5 }; })
  };

  addChatMessage('agent', '处理中...');
  var loadingMsg = document.getElementById('chat-messages').lastChild;

  fetch(API_BASE + '/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: msg, topology: topo })
  })
  .then(function(r) { return r.json(); })
  .then(function(resp) {
    // Remove loading message
    if (loadingMsg) loadingMsg.remove();

    addChatMessage('agent', resp.reply || '已完成');

    // Handle actions
    if (resp.action === 'load_topology' && resp.topology) {
      saveUndo();
      devices = resp.topology.devices;
      connections = resp.topology.connections;
      nextId = Math.max.apply(null, devices.map(function(d) { return parseInt(d.id.split('-')[1]) || 0; })) + 1;
      renderAll();
      updateStatus();
      showProps();
      document.getElementById('btn-fit').click();
      addChatMessage('agent', '拓扑已加载到编辑器中，你可以修改后点击"部署"。');
    }

    if (resp.action === 'show_results' && resp.results) {
      showResults(resp.results);
      addChatMessage('agent', '推理结果已展示在下方面板。');
    }
  })
  .catch(function(err) {
    if (loadingMsg) loadingMsg.remove();
    addChatMessage('agent', '请求失败: ' + err.message);
  });
}
```

- [ ] **Step 4: Commit**

```bash
git add topology-editor.html && git commit -m "feat: add chat panel with Agent integration"
```

---

### Task 13: Integration test and final wiring

**Files:**
- Create: `backend/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# backend/test_integration.py
"""Integration test for the AI Router pipeline (no Docker required for inference)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from schemas.models import TopologyJSON, Device, Connection, ConnectionEndpoint
from mininet.templates import build_nx_graph
from traffic.generator import generate_flows
from model.inference import InferenceEngine


def test_topology_to_nx():
    """Test that TopologyJSON converts correctly to networkx Graph."""
    topo = TopologyJSON(
        devices=[
            Device(id="dev-1", type="router", x=100, y=100, label="R1", ip="10.0.0.1"),
            Device(id="dev-2", type="router", x=300, y=100, label="R2", ip="10.0.0.2"),
            Device(id="dev-3", type="server", x=200, y=300, label="S1", ip="10.0.1.1"),
        ],
        connections=[
            Connection(
                id="conn-1",
                from_=ConnectionEndpoint(devId="dev-1", port="bottom"),
                to=ConnectionEndpoint(devId="dev-3", port="top"),
                bandwidth=100,
                delay=5,
            ),
            Connection(
                id="conn-2",
                from_=ConnectionEndpoint(devId="dev-2", port="bottom"),
                to=ConnectionEndpoint(devId="dev-3", port="top"),
                bandwidth=50,
                delay=10,
            ),
        ],
    )

    G, id_to_idx = build_nx_graph(topo)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G[0][2]["bandwidth"] == 100
    assert G[1][2]["delay"] == 10
    print("PASS: topology_to_nx")


def test_flow_generation():
    """Test traffic flow generation."""
    flows = generate_flows(5, num_flows=10, seed=42)
    assert len(flows) == 10
    for f in flows:
        assert 0 <= f["src"] < 5
        assert 0 <= f["dst"] < 5
        assert f["src"] != f["dst"]
        assert 0.5 <= f["bw_req"] <= 40.0
        assert 0.0 <= f["phi"] <= 1.0
    print("PASS: flow_generation")


def test_mininet_script():
    """Test Mininet script generation."""
    from mininet.templates import generate_mininet_script

    topo = TopologyJSON(
        devices=[
            Device(id="dev-1", type="router", x=100, y=100, label="R1", ip="10.0.0.1"),
            Device(id="dev-2", type="server", x=200, y=200, label="S1", ip="10.0.0.2"),
        ],
        connections=[
            Connection(
                id="conn-1",
                from_=ConnectionEndpoint(devId="dev-1", port="bottom"),
                to=ConnectionEndpoint(devId="dev-2", port="top"),
                bandwidth=100,
                delay=5,
            ),
        ],
    )

    script = generate_mininet_script(topo)
    assert "class CustomTopo(Topo)" in script
    assert "h1 = self.addHost('h1'" in script
    assert "h2 = self.addHost('h2'" in script
    assert "bw=100" in script
    assert "delay='5.0ms'" in script
    print("PASS: mininet_script")


def test_inference_without_checkpoint():
    """Test that inference engine reports missing checkpoint gracefully."""
    engine = InferenceEngine(ckpt_path="/nonexistent/path.pt")
    import networkx as nx
    G = nx.Graph()
    G.add_edge(0, 1, bandwidth=100, delay=5)
    try:
        engine.infer(G, [{"flow_id": 0, "src": 0, "dst": 1, "bw_req": 10, "phi": 0.5}])
        print("FAIL: should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"PASS: inference_without_checkpoint — {e}")


if __name__ == "__main__":
    test_topology_to_nx()
    test_flow_generation()
    test_mininet_script()
    test_inference_without_checkpoint()
    print("\nAll integration tests passed!")
```

- [ ] **Step 2: Run tests**

```bash
cd backend && python test_integration.py
```

Expected: all 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add backend/test_integration.py && git commit -m "test: add integration tests for topology, flows, mininet script, and inference"
```

---

### Task 14: Startup script and README

**Files:**
- Create: `start.sh`

- [ ] **Step 1: Write startup script**

```bash
#!/bin/bash
# start.sh — Launch the AI Router platform

echo "=== AI Router Platform ==="
echo ""

# Check prerequisites
if ! command -v python &>/dev/null; then
    echo "ERROR: Python 3.12+ required"
    exit 1
fi

if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker required"
    exit 1
fi

# Install backend dependencies
echo "[1/3] Installing backend dependencies..."
cd backend
pip install -r requirements.txt -q
cd ..

# Install network-rl
echo "[2/3] Installing network-rl package..."
cd 模型项目/network-rl
pip install -e . -q 2>/dev/null || echo "  (network-rl editable install skipped — deps may need manual install)"
cd ../..

# Start backend
echo "[3/3] Starting backend server..."
echo ""
echo "  Backend:  http://localhost:8000"
echo "  API docs: http://localhost:8000/docs"
echo "  Frontend: open topology-editor.html in browser"
echo ""
cd backend && python run.py
```

- [ ] **Step 2: Commit**

```bash
git add start.sh && git commit -m "chore: add startup script"
```
