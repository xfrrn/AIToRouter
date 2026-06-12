# backend/main.py
"""AI Router — FastAPI backend."""

from __future__ import annotations

import uuid
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

    container_id = None
    tmpdir = None

    try:
        # 1. Convert topology to networkx
        G, id_to_idx = build_nx_graph(topology)
        N = G.number_of_nodes()

        # 2. Generate traffic flows
        flows = generate_flows(N, seed=42)

        # 3. Map flow src/dst from integer indices to Mininet hostnames
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
    mn_manager.stop_and_remove(container_id)
    return {"status": "removed", "container_id": container_id}


# ── Agent chat ────────────────────────────────────────────────────
from agent.orchestrator import AgentOrchestrator

agent: AgentOrchestrator | None = None


@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Agent chat endpoint — natural language driven orchestration."""
    global agent, inference_engine

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
