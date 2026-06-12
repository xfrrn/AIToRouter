# backend/main.py
"""AI Router — FastAPI backend."""

from __future__ import annotations

import uuid
import time
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from schemas.models import (
    TopologyJSON,
    DeploymentResult,
    FlowResult,
    ChatRequest,
    ChatResponse,
)
from mininet.templates import build_nx_graph
from traffic.generator import generate_flows
from model.inference import InferenceEngine

# Global state
mn_manager = None  # MininetManager — initialized only if Docker is available
inference_engine: InferenceEngine | None = None
docker_available = False
jobs: dict[str, DeploymentResult] = {}


def _build_result(job_id: str, G, flow_results: list[dict], edge_utils: dict) -> DeploymentResult:
    """Build DeploymentResult from inference output."""
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

    return DeploymentResult(
        job_id=job_id,
        status="completed",
        flows=result_flows,
        topology_nodes=list(range(G.number_of_nodes())),
        topology_edges=topo_edges,
    )


def _run_ospf_baseline(G, flows: list[dict]) -> tuple[list[dict], dict]:
    """Run OSPF (hop-shortest path) baseline when no trained model is available."""
    import networkx as nx

    results = []
    edge_utils = {}
    # Initialize utilization
    for u, v in G.edges():
        edge_utils[(u, v)] = 0.0
        edge_utils[(v, u)] = 0.0

    for flow in flows:
        src, dst = flow["src"], flow["dst"]
        bw_req = flow["bw_req"]

        try:
            path = nx.shortest_path(G, src, dst, weight=None)
        except nx.NetworkXNoPath:
            path = []

        # Update utilization along path
        max_util = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            cap = G[u][v].get("bandwidth", 100.0)
            edge_utils[(u, v)] = edge_utils.get((u, v), 0.0) + bw_req / cap
            edge_utils[(v, u)] = edge_utils.get((v, u), 0.0) + bw_req / cap
            max_util = max(max_util, edge_utils[(u, v)])

        results.append({
            "flow_id": flow["flow_id"],
            "src": src,
            "dst": dst,
            "bw_req": bw_req,
            "phi": flow["phi"],
            "selected_path": path,
            "hops": max(len(path) - 1, 0),
            "ospf_path": path,
            "max_link_utilization": round(min(max_util, 1.0), 4),
        })

    return results, edge_utils


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mn_manager, inference_engine, docker_available

    # Initialize inference engine (always available)
    inference_engine = InferenceEngine()

    # Check Docker availability (lightweight — no image pull)
    from mininet.manager import check_docker_available
    docker_available = check_docker_available()
    if docker_available:
        print("[OK] Docker is available — /api/deploy enabled.")
    else:
        print("[WARN] Docker not available — /api/deploy disabled. Use /api/infer.")

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


@app.get("/")
async def root():
    topo_html = Path(__file__).resolve().parent.parent / "topology-editor.html"
    return FileResponse(topo_html)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "docker": docker_available,
        "model_loaded": inference_engine.policy is not None if inference_engine else False,
    }


@app.post("/api/infer")
async def infer(topology: TopologyJSON) -> DeploymentResult:
    """Infer optimal routes from topology — no Docker required.

    Uses the network-rl model if a checkpoint is available,
    otherwise falls back to OSPF (hop-shortest) baseline.
    """
    job_id = uuid.uuid4().hex[:12]

    try:
        G, id_to_idx = build_nx_graph(topology)
        N = G.number_of_nodes()
        flows = generate_flows(N, seed=42)

        # Try model inference, fall back to OSPF baseline
        try:
            flow_results, edge_utils = inference_engine.infer(G, flows)
        except (FileNotFoundError, ImportError):
            print("[INFO] Model not available — using OSPF baseline")
            flow_results, edge_utils = _run_ospf_baseline(G, flows)

        result = _build_result(job_id, G, flow_results, edge_utils)

    except Exception as e:
        result = DeploymentResult(
            job_id=job_id,
            status="failed",
            error=str(e),
        )

    jobs[job_id] = result
    return result


@app.post("/api/deploy")
async def deploy(topology: TopologyJSON) -> DeploymentResult:
    """Deploy topology to Mininet, run traffic, infer optimal routes.

    Requires Docker and a running Mininet image.
    For a lighter alternative, use /api/infer.
    """
    if not docker_available:
        raise HTTPException(
            status_code=503,
            detail="Docker is not available. Use /api/infer for inference-only mode.",
        )

    global mn_manager
    from mininet.manager import MininetManager
    if mn_manager is None:
        mn_manager = MininetManager()  # lazily init + pull image on first deploy

    job_id = uuid.uuid4().hex[:12]

    jobs[job_id] = DeploymentResult(
        job_id=job_id,
        status="running",
        flows=[],
        topology_nodes=[],
        topology_edges=[],
    )

    container_id = None
    tmpdir = None

    try:
        G, id_to_idx = build_nx_graph(topology)
        N = G.number_of_nodes()
        flows = generate_flows(N, seed=42)

        mininet_flows = []
        for f in flows:
            mininet_flows.append({
                "flow_id": f["flow_id"],
                "src": f"h{f['src'] + 1}",
                "dst": f"h{f['dst'] + 1}",
                "bw_req": f["bw_req"],
                "duration": f["duration"],
            })

        container_id, exec_id, tmpdir = mn_manager.deploy(topology, mininet_flows)
        time.sleep(5)

        flow_results, edge_utils = inference_engine.infer(G, flows)
        result = _build_result(job_id, G, flow_results, edge_utils)

    except Exception as e:
        result = DeploymentResult(
            job_id=job_id,
            status="failed",
            error=str(e),
        )

    finally:
        if container_id is not None:
            mn_manager.stop_and_remove(container_id)
        if tmpdir is not None:
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
    if mn_manager is None:
        raise HTTPException(status_code=503, detail="Docker not available")
    mn_manager.stop_and_remove(container_id)
    return {"status": "removed", "container_id": container_id}


# ── Agent chat ────────────────────────────────────────────────────
agent = None  # AgentOrchestrator — initialized lazily when first used


@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Agent chat endpoint — natural language driven orchestration."""
    global agent, inference_engine

    if agent is None:
        try:
            from agent.orchestrator import AgentOrchestrator
            agent = AgentOrchestrator()
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Agent is not available. Install 'anthropic' package and set ANTHROPIC_API_KEY.",
            )

    def on_topology(topo):
        pass  # topology is returned in ChatResponse, frontend loads it

    def on_traffic(flows):
        pass  # returned in response

    def on_deploy(topo, traffic):
        # Run inference-only pipeline (no Docker needed)
        G, _ = build_nx_graph(topo)
        try:
            flow_results, edge_utils = inference_engine.infer(G, traffic)
        except (FileNotFoundError, ImportError):
            flow_results, edge_utils = _run_ospf_baseline(G, traffic)

        return _build_result("agent", G, flow_results, edge_utils)

    return agent.chat(
        request.message,
        topology=request.topology,
        on_topology=on_topology,
        on_traffic=on_traffic,
        on_deploy=on_deploy,
        api_key=request.api_key,
        base_url=request.base_url,
    )
